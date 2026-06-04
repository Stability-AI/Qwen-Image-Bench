"""vLLM inference engine (via ms-swift's VllmEngine).

Drop-in alternative to backends.ms_swift_backend.MsSwiftJudge that runs the
judge model on vLLM instead of HF static batching. It exposes the same
``generate_batch(items)`` contract and reuses the identical RequestConfig and
``enable_thinking`` template setup, so judge outputs stay comparable to the
PtEngine path while gaining continuous batching + PagedAttention.

vLLM does its own scheduling, so callers should submit as many requests as
possible in a single ``generate_batch`` call and let ``max_num_seqs`` bound
concurrency (rather than pre-slicing into small static batches).

Requires ms-swift>=4.0.0 and a compatible vllm install.
"""

# VllmEngine import location has moved across ms-swift releases; try the modern
# path first, then the top-level alias used by the PtEngine backend.
try:
    from swift.llm import VllmEngine
except ImportError:  # pragma: no cover - depends on installed ms-swift layout
    from swift import VllmEngine

from swift import InferRequest, RequestConfig


class VllmJudge:
    def __init__(
        self,
        model_path,
        max_new_tokens=4096,
        max_num_seqs=256,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        max_model_len=None,
    ):
        engine_kwargs = dict(
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"image": 1},
        )
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = max_model_len

        self.engine = VllmEngine(model_path, **engine_kwargs)
        self.request_config = RequestConfig(
            max_tokens=max_new_tokens,
            temperature=0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.05,
            seed=42,
        )

        # Enable Qwen3 thinking mode on the engine's default template, mirroring
        # the PtEngine backend so the two paths stay comparable.
        try:
            self.engine.default_template.template_meta.template_kwargs = {
                "enable_thinking": True
            }
        except AttributeError:
            pass  # fall back to per-request template_inputs below

    def generate_batch(self, items):
        """
        Batch inference for multiple items.
        Each item: {"system_prompt": str, "user_text": str, "image": PIL.Image}
        Returns list of generated text strings.
        """
        infer_requests = []
        for item in items:
            messages = [
                {"role": "system", "content": item["system_prompt"]},
                {"role": "user", "content": item["user_text"]},
            ]
            infer_requests.append(
                InferRequest(messages=messages, images=[item["image"]])
            )

        resp_list = self.engine.infer(
            infer_requests,
            self.request_config,
        )

        return [r.choices[0].message.content for r in resp_list]
