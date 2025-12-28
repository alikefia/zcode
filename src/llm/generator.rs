use std::path::PathBuf;

use anyhow::{Error as E, Result};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen2::ModelForCausalLM;
use tokenizers::Tokenizer;

const DEFAULT_SEED: u64 = 299792458;
const DEFAULT_TEMPERATURE: f64 = 0.5;
const DEFAULT_REPEAT_PENALTY: f32 = 1.1;
const DEFAULT_REPEAT_LAST_N: usize = 64;

pub(super) struct ModelFiles {
    pub config_file: PathBuf,
    pub tokenizer_file: PathBuf,
    pub weights_files: Vec<PathBuf>,
}

pub(super) struct Generator {
    device: Device,
    model: ModelForCausalLM,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos_tokens: Vec<u32>,
}

impl Generator {
    pub fn new(
        files: &ModelFiles,
        device: &Device,
        dtype: DType,
        seed: Option<u64>,
        temperature: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: Option<f32>,
        repeat_last_n: Option<usize>,
    ) -> Result<Self> {
        let config = serde_json::from_slice(&std::fs::read(files.config_file.to_owned())?)?;
        let tokenizer = Tokenizer::from_file(files.tokenizer_file.to_owned()).map_err(E::msg)?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&files.weights_files, dtype, device)? };
        let model = ModelForCausalLM::new(&config, vb)?;
        let logits_processor = LogitsProcessor::new(
            match seed {
                Some(v) => v,
                None => DEFAULT_SEED,
            },
            match temperature {
                Some(v) => Some(v),
                None => Some(DEFAULT_TEMPERATURE),
            },
            top_p,
        );
        let mut r = Self {
            device: device.clone(),
            model: model,
            tokenizer: tokenizer,
            logits_processor,
            repeat_penalty: match repeat_penalty {
                Some(v) => v,
                None => DEFAULT_REPEAT_PENALTY,
            },
            repeat_last_n: match repeat_last_n {
                Some(v) => v,
                None => DEFAULT_REPEAT_LAST_N,
            },
            eos_tokens: vec![],
        };

        if let Some(v) = r.get_token("<|endoftext|>") {
            r.eos_tokens.push(v);
        }
        if let Some(v) = r.get_token("<|im_end|>") {
            r.eos_tokens.push(v);
        }
        Ok(r)
    }

    pub fn new_with_defaults(files: &ModelFiles, device: &Device, dtype: DType) -> Result<Self> {
        Self::new(files, device, dtype, None, None, None, None, None)
    }

    fn get_token(&self, text: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(text).copied()
    }

    pub fn generate(&mut self, prompt: &str, max_new_tokens: usize) -> Result<()> {
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let tokens_in_len = tokens.len();
        let start_time = std::time::Instant::now();
        for index in 0..max_new_tokens {
            let ctx_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(ctx_size);
            let ctx = &tokens[start_pos..];
            let input = Tensor::new(ctx, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            if self.eos_tokens.contains(&next_token) {
                break;
            }
        }
        let generated = self
            .tokenizer
            .decode(&tokens[tokens_in_len..], true)
            .map_err(E::msg)?;
        println!("{}", generated);
        println!(
            "rate: {} t/s",
            (tokens.len() - tokens_in_len) as u64 / start_time.elapsed().as_secs()
        );
        Ok(())
    }
}
