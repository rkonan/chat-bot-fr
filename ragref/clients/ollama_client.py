from __future__ import annotations
import os, json, requests
from typing import Any, Dict, Iterable, List, Optional

class OllamaClient:
    """Minimal Ollama client used by the engine and expanders.
    Compatible with /api/generate and /api/chat.
    """
    def __init__(self, model: str, host: Optional[str] = None, timeout: int = 300):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout
        self._gen_url = self.host.rstrip("/") + "/api/generate"
        self._chat_url = self.host.rstrip("/") + "/api/chat"

    def generate(self, prompt: str, stop: Optional[List[str]] = None,
                 max_tokens: Optional[int] = None, stream: bool = False,
                 raw: bool = False) -> str | Iterable[str]:
        payload: Dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": stream}
        if raw: payload["raw"] = True
        if stop: payload["stop"] = stop
        if max_tokens is not None: payload["num_predict"] = int(max_tokens)

        if stream:
            def _gen():
                with requests.post(self._gen_url, json=payload, stream=True, timeout=self.timeout) as r:
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if not line: continue
                        try: data = json.loads(line)
                        except Exception: continue
                        if "response" in data and not data.get("done"): yield data["response"]
                        if data.get("done"): break
            return _gen()

        r = requests.post(self._gen_url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("response", "")

    def chat(self, messages: list, stream: bool = False) -> Dict | Iterable[str]:
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": stream}
        if stream:
            def token_gen():
                with requests.post(self._chat_url, json=payload, stream=True, timeout=self.timeout) as r:
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if not line: continue
                        data = json.loads(line)
                        if "message" in data and not data.get("done"): yield data["message"]["content"]
                        if data.get("done"): break
            return token_gen()
        r = requests.post(self._chat_url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()