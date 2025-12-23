from time import time

from transformers import AutoModelForCausalLM, AutoTokenizer


code = """
def slice(l: list[Any], *, n: int) -> Generator:
    """
models = {
    "Qwen/Qwen2.5-Coder-0.5B": "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301",
    "Qwen/Qwen2.5-Coder-1.5B": "df3ce67c0e24480f20468b6ef2894622d69eb73b",
}


def run(model_name, model_rev, prompt):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_rev,
        local_files_only=True,
        dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_rev,
        local_files_only=True,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    start = time()
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=128)
    input_ids = model_inputs.input_ids[0]
    generated_ids = generated_ids[0]
    new_ids = generated_ids[len(input_ids) :]
    return (
        tokenizer.batch_decode([new_ids], skip_special_tokens=True)[0],
        len(new_ids) / (time() - start),
    )


def main():
    for model_name, model_rev in models.items():
        print(f"\n## {model_name} ##\n")
        res, rate = run(model_name, model_rev, code)
        print(res)
        print(f"\nrate: {rate:.2f}t/s\n")


if __name__ == "__main__":
    main()
