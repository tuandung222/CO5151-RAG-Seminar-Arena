import time
import urllib.request
import json
from typing import List, Dict, Any, Optional
import openai
import huggingface_hub

from .config import (
    DEFAULT_PROVIDER,
    DEFAULT_BASE_URL,
    DEFAULT_API_KEY,
    DEFAULT_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    USER_AGENT,
    HF_TOKEN,
)


def safe_log(msg: str):
    """Safely log messages without crashing on headless/detached stdout broken pipes."""
    try:
        print(msg, flush=True)
    except Exception:
        pass


class UnifiedLLM:
    """
    Pluggable, Multi-Provider LLM Client:
    1. Hugging Face Serverless Inference (via official huggingface_hub using User Token)
    2. Any OpenAI-Compatible endpoint (vLLM, LiteLLM, OpenRouter, Groq, FastChat)
    3. Local vLLM (e.g. http://localhost:8000/v1)
    4. Local Ollama (via native /api/generate or OpenAI-compatible /v1)
    """

    def __init__(
        self,
        provider: str = DEFAULT_PROVIDER,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        model_name: str = DEFAULT_MODEL,
    ):
        self.provider = provider
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key if api_key else "EMPTY"
        self.model_name = model_name

        self._init_client()

    def _init_client(self):
        """Initialize appropriate client based on selected provider."""
        if self.provider == "huggingface":
            self.hf_client = huggingface_hub.InferenceClient(token=self.api_key or HF_TOKEN)
        else:
            self.openai_client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                default_headers={"User-Agent": USER_AGENT},
                timeout=45.0,
            )

    def test_connection(self) -> Dict[str, Any]:
        """Test connection to the configured inference engine."""
        start_t = time.time()
        try:
            if self.provider == "huggingface":
                resp = self.hf_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5,
                )
                elapsed = round((time.time() - start_t) * 1000, 2)
                return {
                    "success": True,
                    "latency_ms": elapsed,
                    "message": f"Hugging Face Inference OK: {resp.choices[0].message.content.strip()}",
                }
            elif self.provider == "ollama-native":
                url = f"{self.base_url}/api/generate"
                payload = json.dumps({
                    "model": self.model_name,
                    "prompt": "ping",
                    "stream": False,
                }).encode("utf-8")
                req = urllib.request.Request(
                    url, data=payload, headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    elapsed = round((time.time() - start_t) * 1000, 2)
                    return {"success": True, "latency_ms": elapsed, "message": "Ollama Native OK"}
            else:
                resp = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5,
                )
                elapsed = round((time.time() - start_t) * 1000, 2)
                return {
                    "success": True,
                    "latency_ms": elapsed,
                    "message": f"{self.provider.upper()} OK: {resp.choices[0].message.content.strip()}",
                }
        except Exception as e:
            return {"success": False, "latency_ms": 0, "message": f"Connection failed: {e}"}

    def generate(
        self,
        prompt: str,
        system_prompt: str = "Bạn là chuyên gia pháp lý chính xác, trung thực và dựa trên bằng chứng.",
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> Dict[str, Any]:
        """Generate text from the configured backend."""
        start_t = time.time()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        if self.provider == "huggingface":
            try:
                resp = self.hf_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                elapsed_ms = round((time.time() - start_t) * 1000, 2)
                content = resp.choices[0].message.content or ""
                return {
                    "text": content.strip(),
                    "latency_ms": elapsed_ms,
                    "provider": "huggingface",
                    "model": self.model_name,
                    "success": True,
                }
            except Exception as e:
                safe_log(f"[LLM Warning] Hugging Face failed ({e}), attempting local Ollama fallback...")
                return self._call_ollama_native(prompt, system_prompt, temperature, max_tokens, start_t)

        elif self.provider == "ollama-native":
            return self._call_ollama_native(prompt, system_prompt, temperature, max_tokens, start_t)

        else:
            try:
                resp = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                elapsed_ms = round((time.time() - start_t) * 1000, 2)
                content = resp.choices[0].message.content or ""
                return {
                    "text": content.strip(),
                    "latency_ms": elapsed_ms,
                    "provider": self.provider,
                    "model": self.model_name,
                    "success": True,
                }
            except Exception as e:
                safe_log(f"[LLM Warning] Provider '{self.provider}' failed ({e}). Attempting fallback to local Ollama...")
                return self._call_ollama_native(prompt, system_prompt, temperature, max_tokens, start_t)

    def _call_ollama_native(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, start_t: float
    ) -> Dict[str, Any]:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                elapsed_ms = round((time.time() - start_t) * 1000, 2)
                return {
                    "text": data.get("response", "").strip(),
                    "latency_ms": elapsed_ms,
                    "provider": "ollama-fallback",
                    "model": OLLAMA_MODEL,
                    "success": True,
                }
        except Exception as err:
            return {
                "text": f"Lỗi gọi LLM: {err}",
                "latency_ms": round((time.time() - start_t) * 1000, 2),
                "provider": "failed",
                "model": "none",
                "success": False,
            }
