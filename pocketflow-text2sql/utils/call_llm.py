import os
from functools import lru_cache
from pathlib import Path

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "llama-3-sqlcoder-0.5b.gguf"
DEFAULT_CHAT_MODEL_PATH = PROJECT_ROOT / "CURE-MED-1.5B.i1-Q4_K_M.gguf"
SQL_SYSTEM_PROMPT = (
    "You generate precise SQLite SQL. "
    "Follow the requested output format exactly and avoid extra commentary."
)
CHAT_SYSTEM_PROMPT = (
    "You are a grounded DHIS2 health assistant. "
    "Use your medical knowledge to answer clearly and accurately. "
    "When database evidence is provided, connect your explanation to that evidence explicitly. "
    "You may add concise general medical context, but do not invent dataset values, patient-specific facts, or unsupported findings. "
    "If the retrieved evidence is indirect, say that clearly and then explain what the concept usually means medically before summarizing what the data shows. "
    "Do not mention prompts, SQL, or models."
)
ANDROID_BRIDGE_ENV = "ANDROID_LLM_BRIDGE"
ANDROID_BRIDGE_CLASS = "com.uliza.healthworker.runtime.AndroidLlmBridge"
ANDROID_BRIDGE_ENABLED = {"1", "true", "yes", "on"}


def _resolve_model_path(env_var_name, default_path):
    model_path = Path(os.environ.get(env_var_name, default_path))
    return model_path.expanduser().resolve()


def _model_path():
    return _resolve_model_path("LLM_MODEL_PATH", DEFAULT_MODEL_PATH)


def _chat_model_path():
    return _resolve_model_path("CHAT_MODEL_PATH", DEFAULT_CHAT_MODEL_PATH)


def get_model_path():
    return _model_path()


def get_chat_model_path():
    return _chat_model_path()


def _use_android_bridge():
    return os.environ.get(ANDROID_BRIDGE_ENV, "").strip().lower() in ANDROID_BRIDGE_ENABLED


def _android_bridge():
    if not _use_android_bridge():
        return None

    try:
        from java import jclass
    except Exception as exc:
        raise RuntimeError("Android bridge requested but Java bridge is unavailable") from exc

    return jclass(ANDROID_BRIDGE_CLASS)


def _android_invoke(profile, prompt, system_prompt, max_tokens, temperature):
    bridge = _android_bridge()
    if bridge is None:
        raise RuntimeError("Android bridge is not enabled")

    result = bridge.generate(
        profile,
        prompt,
        system_prompt or "",
        int(max_tokens),
        float(temperature),
    )
    if result is None:
        raise RuntimeError(f"Android bridge returned no content for profile '{profile}'")
    return str(result)


def is_model_available():
    model_path = _model_path()
    if _use_android_bridge():
        return model_path.exists()
    return Llama is not None and model_path.exists()


def is_chat_model_available():
    model_path = _chat_model_path()
    if _use_android_bridge():
        return model_path.exists()
    return Llama is not None and model_path.exists()


def model_unavailable_reason():
    model_path = _model_path()
    if _use_android_bridge():
        if not model_path.exists():
            return f"GGUF model not found at: {model_path}"
        return None
    if Llama is None:
        return "llama_cpp is not installed or could not be imported"
    if not model_path.exists():
        return f"GGUF model not found at: {model_path}"
    return None


def chat_model_unavailable_reason():
    model_path = _chat_model_path()
    if _use_android_bridge():
        if not model_path.exists():
            return f"Chat GGUF model not found at: {model_path}"
        return None
    if Llama is None:
        return "llama_cpp is not installed or could not be imported"
    if not model_path.exists():
        return f"Chat GGUF model not found at: {model_path}"
    return None


@lru_cache(maxsize=1)
def _get_llm(model_path_str):
    model_path = Path(model_path_str)
    if Llama is None:
        raise RuntimeError("llama_cpp is not installed or could not be imported")
    if not model_path.exists():
        raise FileNotFoundError(f"GGUF model not found at: {model_path}")

    n_threads = int(os.environ.get("LLM_THREADS", max(1, os.cpu_count() or 1)))
    n_ctx = int(os.environ.get("LLM_CTX_SIZE", 4096))
    n_gpu_layers = int(os.environ.get("LLM_GPU_LAYERS", 0))

    return Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )


def _chat_messages(prompt, system_prompt):
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]


def _invoke_llm(model_path, prompt, system_prompt, max_tokens, temperature):
    llm = _get_llm(str(model_path))

    try:
        response = llm.create_chat_completion(
            messages=_chat_messages(prompt, system_prompt),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"]
    except Exception:
        fallback_prompt = (
            f"{system_prompt}\n\n"
            f"{prompt}"
        )
        response = llm(
            fallback_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["text"].strip()


def call_llm(prompt, system_prompt=None, max_tokens=None, temperature=None):
    resolved_max_tokens = max_tokens if max_tokens is not None else int(os.environ.get("LLM_MAX_TOKENS", 512))
    resolved_temperature = temperature if temperature is not None else float(os.environ.get("LLM_TEMPERATURE", 0.0))
    if _use_android_bridge():
        return _android_invoke(
            "sql",
            prompt,
            system_prompt or SQL_SYSTEM_PROMPT,
            resolved_max_tokens,
            resolved_temperature,
        )
    return _invoke_llm(
        _model_path(),
        prompt,
        system_prompt or SQL_SYSTEM_PROMPT,
        resolved_max_tokens,
        resolved_temperature,
    )


def call_chat_llm(prompt, system_prompt=None, max_tokens=None, temperature=None):
    resolved_max_tokens = max_tokens if max_tokens is not None else int(os.environ.get("CHAT_LLM_MAX_TOKENS", 384))
    resolved_temperature = temperature if temperature is not None else float(os.environ.get("CHAT_LLM_TEMPERATURE", 0.2))
    if _use_android_bridge():
        return _android_invoke(
            "chat",
            prompt,
            system_prompt or CHAT_SYSTEM_PROMPT,
            resolved_max_tokens,
            resolved_temperature,
        )
    return _invoke_llm(
        _chat_model_path(),
        prompt,
        system_prompt or CHAT_SYSTEM_PROMPT,
        resolved_max_tokens,
        resolved_temperature,
    )

# Example usage
if __name__ == "__main__":
    print(call_llm("Tell me a short joke")) 
