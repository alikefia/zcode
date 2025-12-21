from time import time

from transformers import AutoModelForCausalLM, AutoTokenizer


code = """
def slice(l: list[Any], *, n: int) -> Generator:
    
"""
models = {
    "Qwen/Qwen2.5-Coder-0.5B": "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301",
    "Qwen/Qwen2.5-Coder-1.5B": "df3ce67c0e24480f20468b6ef2894622d69eb73b",
    "Qwen/Qwen2.5-Coder-3B": "09d9bc5d376b0cfa0100a0694ea7de7232525803",
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
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ], time() - start


def main():
    for model_name, model_rev in models.items():
        print(f"\n## {model_name} ##\n")
        res, t = run(model_name, model_rev, code)
        print(res)
        print(f"\ndone in {t:.2f}\n")


if __name__ == "__main__":
    main()
