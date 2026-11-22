"""The one pluggable model function. Every model call in this study (write a note, write a query,
answer, judge) goes through call(prompt, system).

Backends (env MEM_LM):
  haiku          blank Claude CLI agent (no CLAUDE.md, memory, tools); subscription login, no paid API
  openai         any OpenAI-compatible chat endpoint, e.g. Ollama or vLLM serving a 7B-9B model:
                 MEM_LM=openai MEM_BASE_URL=http://localhost:11434/v1 MEM_MODEL=qwen2.5:7b-instruct
Results are cached in cache/<backend>-<model>.jsonl keyed by (system, prompt), so reruns are free and a
second backend never reuses the first backend's answers.
"""
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.request

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
BLANK = os.path.join(HERE, "..", "blank-agent", "blank_agent.sh")
BACKEND = os.environ.get("MEM_LM", "haiku")
MODEL = os.environ.get("MEM_MODEL", "haiku" if BACKEND == "haiku" else "qwen2.5:7b-instruct")
BASE = os.environ.get("MEM_BASE_URL", "http://localhost:11434/v1")
CACHE = os.path.join(HERE, "cache", f"{BACKEND}-{MODEL.replace('/', '_').replace(':', '_')}.jsonl")
_lock = threading.Lock()
_mem = None
STATS = {"calls": 0, "cached": 0, "tok_in": 0, "tok_out": 0}


def _key(system, prompt):
    return hashlib.sha256((system + "\x00" + prompt).encode()).hexdigest()


def _load():
    global _mem
    if _mem is None:
        _mem = {}
        if os.path.exists(CACHE):
            for line in open(CACHE, encoding="utf-8"):
                try:
                    r = json.loads(line)
                    _mem[r["k"]] = r
                except json.JSONDecodeError:
                    pass
    return _mem


def _haiku(prompt, system):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(system)
    j = None
    try:
        for _ in range(3):
            try:
                p = subprocess.run([BLANK, "-m", MODEL, "-s", f.name], input=prompt, capture_output=True,
                                   text=True, timeout=240)
                j = json.loads(p.stdout)
                if not j.get("is_error"):
                    break
            except (subprocess.TimeoutExpired, json.JSONDecodeError):
                j = None
            time.sleep(5)
    finally:
        os.unlink(f.name)
    if j is None or j.get("is_error"):
        raise RuntimeError("blank agent failed 3 times")
    u = j["usage"]
    tin = u["input_tokens"] + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0)
    return j["result"].strip(), tin, u["output_tokens"]


def _openai(prompt, system):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 2048,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(BASE.rstrip("/") + "/chat/completions", body,
                                 {"Content-Type": "application/json",
                                  "Authorization": "Bearer " + os.environ.get("MEM_API_KEY", "none")})
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.load(r)
    u = j.get("usage", {})
    return j["choices"][0]["message"]["content"].strip(), u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def call(prompt, system="You are a helpful assistant."):
    k = _key(system, prompt)
    with _lock:
        hit = _load().get(k)
    if hit:
        STATS["cached"] += 1
        return hit["out"]
    out, tin, tout = (_haiku if BACKEND == "haiku" else _openai)(prompt, system)
    with _lock:
        STATS["calls"] += 1
        STATS["tok_in"] += tin
        STATS["tok_out"] += tout
        _mem[k] = {"k": k, "out": out, "tin": tin, "tout": tout}
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "a", encoding="utf-8") as f:
            f.write(json.dumps(_mem[k]) + "\n")
    return out


def pmap(fn, items, workers=2):
    """Run fn over items with at most `workers` model calls at once (the 2-agent limit)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))
