extern crate accelerate_src;

use anyhow::Result;

use candle_core::{DType, Device};
use hf_hub::{Repo, RepoType, api::sync::Api};

use super::profiler::with_profiler;

mod generator;
use generator::{Generator, ModelFiles};

const MODEL_ID: &str = "Qwen/Qwen2.5-Coder-1.5B";
const MODEL_REV: &str = "df3ce67c0e24480f20468b6ef2894622d69eb73b";

const PROMPT: &str = r"
def slice(l: list[Any], *, n: int) -> Generator:
     ";
const DEFAULT_MAX_NEW_TOKENS: usize = 128;

fn get_repo_files(model_id: &str, model_rev: &str) -> Result<ModelFiles> {
    let repo = Api::new()?.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        model_rev.to_string(),
    ));
    return Ok(ModelFiles {
        config_file: repo.get("config.json")?,
        tokenizer_file: repo.get("tokenizer.json")?,
        weights_files: vec![repo.get("model.safetensors")?],
    });
}

pub(crate) fn run() -> Result<()> {
    let files = with_profiler("model get", || get_repo_files(MODEL_ID, MODEL_REV))?;
    let mut generator = with_profiler("model load", || {
        Generator::new_with_defaults(&files, &Device::new_metal(0)?, DType::BF16)
    })?;
    with_profiler("model infer", || {
        generator.generate(PROMPT, DEFAULT_MAX_NEW_TOKENS)
    })
}
