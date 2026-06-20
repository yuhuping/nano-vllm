import argparse
import json
import os
import time
import uuid
from threading import Lock
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from nanovllm import LLM, SamplingParams


app = FastAPI(title="nano-vllm OpenAI-compatible API")
server: "OpenAIServer | None" = None


class OpenAIServer:
    def __init__(self, llm: LLM, model_name: str):
        self.llm = llm
        self.model_name = model_name
        self.lock = Lock()

    def create_completion(self, request: dict[str, Any]) -> dict[str, Any]:
        self._reject_common_unsupported(request)
        if request.get("logprobs") is not None:
            raise ValueError("logprobs is not supported by this simplified server")

        prompts = self._normalize_prompts(request.get("prompt"))
        sampling_params = self._sampling_params(request)

        with self.lock:
            outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        choices = []
        prompt_tokens = 0
        completion_tokens = 0
        for index, (prompt, output) in enumerate(zip(prompts, outputs)):
            out_token_ids = output["token_ids"]
            prompt_tokens += self._count_prompt_tokens(prompt)
            completion_tokens += len(out_token_ids)
            choices.append({
                "text": output["text"],
                "index": index,
                "logprobs": None,
                "finish_reason": self._finish_reason(out_token_ids, sampling_params.max_tokens),
            })

        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.get("model", self.model_name),
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def stream_completion(self, request: dict[str, Any]):
        self._reject_common_unsupported(request)
        if request.get("logprobs") is not None:
            raise ValueError("logprobs is not supported by this simplified server")

        prompts = self._normalize_prompts(request.get("prompt"))
        if len(prompts) != 1:
            raise ValueError("streaming completions only support one prompt")

        prompt = prompts[0]
        sampling_params = self._sampling_params(request)
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model = request.get("model", self.model_name)
        prompt_tokens = self._count_prompt_tokens(prompt)

        def events():
            completion_tokens = 0
            with self.lock:
                for output in self.llm.generate_stream(prompt, sampling_params):
                    completion_tokens = len(output["token_ids"])
                    if output["text"]:
                        yield self._sse({
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "text": output["text"],
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None,
                            }],
                        })
                    if output["finished"]:
                        yield self._sse({
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "text": "",
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": self._finish_reason(output["token_ids"], sampling_params.max_tokens),
                            }],
                            "usage": self._usage(prompt_tokens, completion_tokens),
                        })
                        yield "data: [DONE]\n\n"

        return events()

    def create_chat_completion(self, request: dict[str, Any]) -> dict[str, Any]:
        self._reject_common_unsupported(request)
        if request.get("tools") or request.get("tool_choice"):
            raise ValueError("tools/tool_choice are not supported by this simplified server")

        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty array")

        prompt = self.llm.tokenizer.apply_chat_template(
            [self._normalize_message(message) for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling_params = self._sampling_params(request)

        with self.lock:
            output = self.llm.generate([prompt], sampling_params, use_tqdm=False)[0]

        out_token_ids = output["token_ids"]
        prompt_tokens = len(self.llm.tokenizer.encode(prompt))
        completion_tokens = len(out_token_ids)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", self.model_name),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output["text"],
                },
                "logprobs": None,
                "finish_reason": self._finish_reason(out_token_ids, sampling_params.max_tokens),
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def stream_chat_completion(self, request: dict[str, Any]):
        self._reject_common_unsupported(request)
        if request.get("tools") or request.get("tool_choice"):
            raise ValueError("tools/tool_choice are not supported by this simplified server")

        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty array")

        prompt = self.llm.tokenizer.apply_chat_template(
            [self._normalize_message(message) for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling_params = self._sampling_params(request)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model = request.get("model", self.model_name)
        prompt_tokens = len(self.llm.tokenizer.encode(prompt))

        def chunk(delta: dict[str, Any], finish_reason: str | None = None, usage: dict[str, int] | None = None):
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }],
            }
            if usage is not None:
                payload["usage"] = usage
            return self._sse(payload)

        def events():
            completion_tokens = 0
            yield chunk({"role": "assistant", "content": ""})
            with self.lock:
                for output in self.llm.generate_stream(prompt, sampling_params):
                    completion_tokens = len(output["token_ids"])
                    if output["text"]:
                        yield chunk({"content": output["text"]})
                    if output["finished"]:
                        usage = self._usage(prompt_tokens, completion_tokens)
                        yield chunk({}, self._finish_reason(output["token_ids"], sampling_params.max_tokens), usage)
                        yield "data: [DONE]\n\n"

        return events()

    def list_models(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{
                "id": self.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nano-vllm",
            }],
        }

    def _reject_common_unsupported(self, request: dict[str, Any]):
        if int(request.get("n", 1)) != 1:
            raise ValueError("n values other than 1 are not supported by this simplified server")

    def _sampling_params(self, request: dict[str, Any]) -> SamplingParams:
        temperature = float(request.get("temperature", 1.0))
        if temperature <= 1e-10:
            temperature = 1e-5
        max_tokens = int(request.get("max_tokens", request.get("max_completion_tokens", 64)))
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        return SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            ignore_eos=bool(request.get("ignore_eos", False)),
        )

    def _normalize_prompts(self, prompt: Any) -> list[str | list[int]]:
        if prompt is None:
            raise ValueError("prompt is required")
        if isinstance(prompt, str):
            return [prompt]
        if self._is_token_list(prompt):
            return [prompt]
        if isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
            return prompt
        if isinstance(prompt, list) and all(self._is_token_list(item) for item in prompt):
            return prompt
        raise ValueError("prompt must be a string, token array, string array, or token-array array")

    def _normalize_message(self, message: Any) -> dict[str, str]:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = message.get("role")
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {role}")
        return {
            "role": role,
            "content": self._message_content_to_text(message.get("content", "")),
        }

    def _message_content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                else:
                    raise ValueError("only text message content parts are supported")
            return "".join(parts)
        if content is None:
            return ""
        raise ValueError("message content must be a string or text content parts")

    def _count_prompt_tokens(self, prompt: str | list[int]) -> int:
        if isinstance(prompt, str):
            return len(self.llm.tokenizer.encode(prompt))
        return len(prompt)

    def _finish_reason(self, token_ids: list[int], max_tokens: int) -> str:
        if len(token_ids) >= max_tokens:
            return "length"
        return "stop"

    def _usage(self, prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _sse(self, payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @staticmethod
    def _is_token_list(value: Any) -> bool:
        return isinstance(value, list) and all(isinstance(token, int) for token in value)


def get_server() -> OpenAIServer:
    if server is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return server


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    return get_server().list_models()


@app.post("/v1/completions")
def create_completion(request: dict[str, Any]):
    try:
        openai_server = get_server()
        if request.get("stream"):
            return StreamingResponse(
                openai_server.stream_completion(request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        return openai_server.create_completion(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/chat/completions")
def create_chat_completion(request: dict[str, Any]):
    try:
        openai_server = get_server()
        if request.get("stream"):
            return StreamingResponse(
                openai_server.stream_chat_completion(request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        return openai_server.create_chat_completion(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Run a simplified OpenAI-compatible nano-vllm server.")
    parser.add_argument("--model", default=os.environ.get("NANOVLLM_MODEL", "~/Qwen3-0.6B/"))
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main():
    global server

    args = parse_args()
    model_path = os.path.expanduser(args.model)
    served_model_name = args.served_model_name or os.path.basename(os.path.normpath(model_path))
    llm = LLM(
        model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        log_file=args.log_file,
    )
    server = OpenAIServer(llm, served_model_name)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
