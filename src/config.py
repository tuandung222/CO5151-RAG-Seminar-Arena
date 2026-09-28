import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CORPUS_PATH = DATA_DIR / "corpus_labor.json"
TEST_CASES_PATH = DATA_DIR / "test_cases.json"
CORPUS_EMBEDDINGS_PATH = DATA_DIR / "corpus_embeddings.npy"
CORPUS_EMBEDDINGS_META_PATH = DATA_DIR / "corpus_embeddings_meta.json"
CACHED_BENCHMARK_PATH = DATA_DIR / "cached_benchmark_results.json"

# Auto-load .env if present (for local development)
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    try:
        with open(_env_file, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
    except Exception:
        pass

# Hugging Face Credentials (Loaded via Environment Variable or Space Secret)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
HF_EMBEDDING_MODEL = "BAAI/bge-m3"
HF_ROUTER_URL = f"https://router.huggingface.co/hf-inference/models/{HF_EMBEDDING_MODEL}"

# Default Inference Settings (Defaults to Hugging Face Provider)
DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "huggingface")
DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api-inference.huggingface.co/v1")
DEFAULT_API_KEY = HF_TOKEN
DEFAULT_MODEL = HF_DEFAULT_MODEL

# Local Ollama Defaults
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

# Local vLLM Defaults
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
