from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import torch

try:
    from feedback_state.newarch_loader import apply_torch_fp8_shim

    apply_torch_fp8_shim()
except Exception:
    pass

from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    use_vllm: bool = False
    local_files_only: bool = False
    tokenizer_mode: str = "auto"
    config_format: str = "auto"
    load_format: str = "auto"
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    # Optional PEFT LoRA adapter directory. When set with the vllm backend, the
    # engine is started with enable_lora=True and every request carries a
    # LoRARequest, so the adapter is applied on top of the base weights.
    lora_path: str | None = None
    max_lora_rank: int = 16
    # vLLM: skip CUDA-graph capture. LoRA + spawn multiprocessing can hang during
    # graph capture on vLLM 0.8.5; eager mode avoids it at a small speed cost.
    enforce_eager: bool = False
    # When set, models whose name is a key here are served by a running vLLM
    # OpenAI-compatible server instead of being loaded in-process. The value is a
    # base_url (".../v1") or a list of base_urls (replicas, round-robined).
    endpoints: dict[str, Any] = field(default_factory=dict)
    # Max in-flight HTTP requests per generate() call when using a vLLM server.
    request_concurrency: int = 64


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(name).lower(), torch.bfloat16)


class TextGenerator:
    def __init__(self, model_name: str, config: GenerationConfig) -> None:
        self.model_name = model_name
        self.config = config
        self.backend = "transformers"
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.llm = None
        self.endpoints: list[str] = []
        self._endpoint_cursor = 0
        self._clients: list[Any] = []
        self._sem = None
        self._lora_request = None
        # vLLM OpenAI-compatible server backend: no weights in-process, only the
        # tokenizer (for render_instruction_prompt). Generation goes over HTTP.
        endpoint = config.endpoints.get(model_name) if config.endpoints else None
        if endpoint:
            self.endpoints = [endpoint] if isinstance(endpoint, str) else list(endpoint)
            self.backend = "vllm_server"
            from openai import AsyncOpenAI

            self._clients = [
                AsyncOpenAI(base_url=url, api_key="EMPTY", max_retries=3)
                for url in self.endpoints
            ]
            self._sem = asyncio.Semaphore(config.request_concurrency)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=config.local_files_only
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            return
        if config.use_vllm:
            try:
                from vllm import LLM

                if config.lora_path:
                    from vllm.lora.request import LoRARequest

                    vllm_kwargs = {}
                    if str(config.config_format).lower() != "auto":
                        vllm_kwargs["config_format"] = str(config.config_format)
                    if str(config.load_format).lower() != "auto":
                        vllm_kwargs["load_format"] = str(config.load_format)
                    if config.max_model_len is not None:
                        vllm_kwargs["max_model_len"] = int(config.max_model_len)
                    if config.gpu_memory_utilization is not None:
                        vllm_kwargs["gpu_memory_utilization"] = float(
                            config.gpu_memory_utilization
                        )
                    self.llm = LLM(
                        model=model_name,
                        tokenizer_mode=str(config.tokenizer_mode),
                        dtype=str(config.dtype),
                        enable_lora=True,
                        max_lora_rank=int(config.max_lora_rank),
                        enforce_eager=bool(config.enforce_eager),
                        **vllm_kwargs,
                    )
                    self._lora_request = LoRARequest("adapter", 1, config.lora_path)
                else:
                    vllm_kwargs = {}
                    if str(config.config_format).lower() != "auto":
                        vllm_kwargs["config_format"] = str(config.config_format)
                    if str(config.load_format).lower() != "auto":
                        vllm_kwargs["load_format"] = str(config.load_format)
                    if config.max_model_len is not None:
                        vllm_kwargs["max_model_len"] = int(config.max_model_len)
                    if config.gpu_memory_utilization is not None:
                        vllm_kwargs["gpu_memory_utilization"] = float(
                            config.gpu_memory_utilization
                        )
                    self.llm = LLM(
                        model=model_name,
                        tokenizer_mode=str(config.tokenizer_mode),
                        dtype=str(config.dtype),
                        enforce_eager=bool(config.enforce_eager),
                        **vllm_kwargs,
                    )
                try:
                    self.tokenizer = self.llm.get_tokenizer()
                    if self.tokenizer.pad_token_id is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.padding_side = "left"
                except Exception:
                    self.tokenizer = None
                self.backend = "vllm"
                return
            except Exception:
                self.llm = None
        load_kwargs = {
            "torch_dtype": dtype_from_name(config.dtype),
            "device_map": "auto" if str(config.device).startswith("cuda") else None,
            "local_files_only": config.local_files_only,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=config.local_files_only
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).eval()
        except ValueError:
            self.processor = AutoProcessor.from_pretrained(
                model_name, local_files_only=config.local_files_only
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, **load_kwargs
            ).eval()
        if not str(config.device).startswith("cuda"):
            self.model.to(config.device)

    def generate(self, prompts: list[str]) -> list[str]:
        if self.backend == "vllm_server":
            return asyncio.run(self.agenerate(prompts))
        if self.backend == "vllm":
            from vllm import SamplingParams

            params = SamplingParams(
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            if self._lora_request is not None:
                outputs = self.llm.generate(prompts, params, lora_request=self._lora_request)
            else:
                outputs = self.llm.generate(prompts, params)
            return [item.outputs[0].text for item in outputs]
        assert self.model is not None and self.tokenizer is not None
        if self.processor is not None:
            encoded = self.processor(text=prompts, return_tensors="pt", padding=True)
        else:
            encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0.0,
                temperature=max(self.config.temperature, 1e-6),
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = encoded["input_ids"].size(1)
        return [
            self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
            for row in outputs
        ]

    async def agenerate(self, prompts: list[str]) -> list[str]:
        """Async generation against vLLM OpenAI servers (vllm_server backend only).

        Prompts are already-rendered strings; we hit /v1/completions so the server
        does NOT re-apply a chat template (semantics match the in-process paths).
        Requests are dispatched concurrently (capped) and round-robined across replicas.
        """
        assert self.backend == "vllm_server", "agenerate requires the vllm_server backend"

        async def one(index: int, prompt: str) -> str:
            client = self._clients[index % len(self._clients)]
            async with self._sem:
                resp = await client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
            return resp.choices[0].text.strip()

        return await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.llm = None
        self._clients = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def render_instruction_prompt(tokenizer: Any, content: str) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except (TypeError, ValueError):
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except ValueError:
                pass
    return content
