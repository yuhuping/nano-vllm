import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser(os.environ.get("NANOVLLM_MODEL", "~/Qwen3-0.6B/"))
    tokenizer = AutoTokenizer.from_pretrained(path)
    os.makedirs("log", exist_ok=True)
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=1,
        log_file="./log/nanovllm_scheduler.log",
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=32)
    prompts = [
        "Long request: explain the difference between prefill and decode in an LLM inference engine. Include batching, KV cache allocation, scheduler behavior, and why decode is usually one token at a time.",
        "Short request: say hi.",
        "Medium request: summarize what a scheduler does in nano-vllm in three sentences.",
        "Tiny request: ok?",
    ]
    for i, prompt in enumerate(prompts):
        token_ids = tokenizer.encode(prompt)
        print(f"input[{i}]: prompt_tokens={len(token_ids)}, text={prompt[:60]!r}")

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    open("./log/nanovllm_scheduler.log", "w").close()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
