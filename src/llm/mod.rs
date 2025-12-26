extern crate accelerate_src;

use anyhow::{Error as E, Result};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::ModelForCausalLM;
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;

mod code_generation;
use code_generation::CodeGeneration;

const MODEL_ID: &str = "Qwen/Qwen2.5-Coder-1.5B";
const MODEL_REV: &str = "df3ce67c0e24480f20468b6ef2894622d69eb73b";
const PROMPT: &str = r"
def slice(l: list[Any], *, n: int) -> Generator:
     ";
pub(crate) fn run() -> Result<()> {
    let start = std::time::Instant::now();
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        MODEL_ID.to_string(),
        RepoType::Model,
        MODEL_REV.to_string(),
    ));
    let tokenizer_file = repo.get("tokenizer.json")?;
    let weight_files = vec![repo.get("model.safetensors")?];
    println!("retrieved the files in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;
    let config_file = repo.get("config.json")?;
    let device = Device::new_metal(0)?;
    let dtype = DType::BF16;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)? };
    let model = ModelForCausalLM::new(&serde_json::from_slice(&std::fs::read(config_file)?)?, vb)?;
    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = CodeGeneration::new(&device, model, tokenizer, None, None, None, None, None);
    pipeline.run(PROMPT, None)
}
