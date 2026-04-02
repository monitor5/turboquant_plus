#!/usr/bin/env python3
"""TurboQuant WebUI — Open WebUI-compatible FastAPI backend.

Architecture mirrors Open WebUI:
  /api/chat/*        — chat completions proxy (OpenAI-compatible, SSE streaming)
  /api/models        — model list from running llama.cpp server
  /api/conversations — conversation CRUD (persisted to data/conversations.json)
  /api/control/*     — TurboQuant server/tests/bench management
  /api/status        — GPU + process status

Pipes (webapp/pipes/) add pre/post-processing middleware.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
CONV_FILE = DATA_DIR / "conversations.json"

VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
PYTHON_BIN = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="TurboQuant WebUI", docs_url=None, redoc_url=None)

# ---------------------------------------------------------------------------
# Config (runtime-mutable)
# ---------------------------------------------------------------------------
_cfg: dict = {
    "server_url": "http://localhost:8080",
    "default_cache_type": "turbo4",
    "system_prompt": "You are a helpful assistant.",
    "temperature": 0.7,
    "max_tokens": 2048,
}

# ---------------------------------------------------------------------------
# Conversation store
# ---------------------------------------------------------------------------
def _load_conversations() -> dict:
    if CONV_FILE.exists():
        try:
            return json.loads(CONV_FILE.read_text())
        except Exception:
            pass
    return {}

def _save_conversations(convs: dict) -> None:
    CONV_FILE.write_text(json.dumps(convs, ensure_ascii=False, indent=2))

_conversations: dict = _load_conversations()

# ---------------------------------------------------------------------------
# Process + log state (Open WebUI "Pipe" concept simplified)
# ---------------------------------------------------------------------------
_procs: dict[str, Optional[subprocess.Popen]] = {
    "server": None,
    "tests": None,
    "bench": None,
}
_logs: dict[str, list[str]] = {"server": [], "tests": [], "bench": []}
_MAX_LOG = 2000


def _tail_proc(proc: subprocess.Popen, key: str) -> None:
    """Background thread: drain stdout into log buffer."""
    assert proc.stdout
    buf = _logs[key]
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip()
        buf.append(line)
        if len(buf) > _MAX_LOG:
            del buf[: len(buf) - _MAX_LOG]
    proc.wait()


# ===========================================================================
# Pydantic models
# ===========================================================================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionReq(BaseModel):
    model: str = "model-turbo"
    messages: list[ChatMessage]
    stream: bool = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    chat_id: Optional[str] = None   # TurboQuant extension

class ConversationCreateReq(BaseModel):
    title: str = "새 대화"

class ConversationUpdateReq(BaseModel):
    title: Optional[str] = None
    messages: Optional[list[dict]] = None

class ConfigUpdateReq(BaseModel):
    server_url: Optional[str] = None
    default_cache_type: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class ServerStartReq(BaseModel):
    model_path: str
    cache_type: str = "turbo4"
    ctx_size: int = 32768
    port: int = 8080
    ngl: int = 999

class TestRunReq(BaseModel):
    mode: str = "quick"

class BenchRunReq(BaseModel):
    model_path: str
    llama_dir: str = ""
    no_ref: bool = False


# ===========================================================================
# Static files + SPA
# ===========================================================================
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((static_dir / "index.html").read_text())


# ===========================================================================
# /api/config
# ===========================================================================
@app.get("/api/config")
async def get_config():
    return _cfg

@app.patch("/api/config")
async def update_config(req: ConfigUpdateReq):
    for k, v in req.model_dump(exclude_none=True).items():
        _cfg[k] = v
    return _cfg


# ===========================================================================
# /api/models  (mirrors Open WebUI model list)
# ===========================================================================
@app.get("/api/models")
async def list_models():
    """Query running llama.cpp server; fall back to cache-type stubs."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{_cfg['server_url']}/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                # Annotate with TurboQuant metadata
                for m in data.get("data", []):
                    m.setdefault("owned_by", "turboquant-llama")
                return data
    except Exception:
        pass

    # Server offline — return placeholder
    return {
        "object": "list",
        "data": [
            {"id": "model-turbo", "object": "model", "owned_by": "turboquant",
             "description": "Start the LLM server to load a model"},
        ],
    }


# ===========================================================================
# /api/chat/completions  (Open WebUI-compatible, SSE streaming proxy)
# ===========================================================================
@app.post("/api/chat/completions")
async def chat_completions(req: ChatCompletionReq, raw_request: Request):
    """Proxy to llama.cpp OpenAI-compatible server with SSE streaming."""
    payload = {
        "model": req.model,
        "messages": [m.model_dump() for m in req.messages],
        "stream": req.stream,
        "temperature": req.temperature if req.temperature is not None else _cfg["temperature"],
        "max_tokens": req.max_tokens if req.max_tokens is not None else _cfg["max_tokens"],
    }

    if req.stream:
        async def stream_gen():
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=5)) as client:
                    async with client.stream(
                        "POST",
                        f"{_cfg['server_url']}/v1/chat/completions",
                        json=payload,
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk
            except httpx.ConnectError:
                err = json.dumps({"error": {"message": "LLM 서버에 연결할 수 없습니다. 서버를 먼저 시작하세요.", "type": "connection_error"}})
                yield f"data: {err}\n\ndata: [DONE]\n\n".encode()
            except Exception as e:
                err = json.dumps({"error": {"message": str(e), "type": "server_error"}})
                yield f"data: {err}\n\ndata: [DONE]\n\n".encode()

        return StreamingResponse(stream_gen(), media_type="text/event-stream",
                                  headers={"X-Accel-Buffering": "no"})
    else:
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    f"{_cfg['server_url']}/v1/chat/completions", json=payload
                )
                return JSONResponse(resp.json(), status_code=resp.status_code)
        except httpx.ConnectError:
            return JSONResponse({"error": "LLM 서버에 연결할 수 없습니다."}, status_code=503)


# ===========================================================================
# /api/conversations  (Open WebUI conversation CRUD)
# ===========================================================================
@app.get("/api/conversations")
async def get_conversations():
    items = [
        {"id": cid, "title": c["title"], "updated_at": c.get("updated_at", 0)}
        for cid, c in _conversations.items()
    ]
    items.sort(key=lambda x: x["updated_at"], reverse=True)
    return items

@app.post("/api/conversations")
async def create_conversation(req: ConversationCreateReq):
    cid = str(uuid.uuid4())
    _conversations[cid] = {
        "title": req.title,
        "messages": [],
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    _save_conversations(_conversations)
    return {"id": cid, **_conversations[cid]}

@app.get("/api/conversations/{cid}")
async def get_conversation(cid: str):
    if cid not in _conversations:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"id": cid, **_conversations[cid]}

@app.patch("/api/conversations/{cid}")
async def update_conversation(cid: str, req: ConversationUpdateReq):
    if cid not in _conversations:
        return JSONResponse({"error": "not found"}, status_code=404)
    if req.title is not None:
        _conversations[cid]["title"] = req.title
    if req.messages is not None:
        _conversations[cid]["messages"] = req.messages
    _conversations[cid]["updated_at"] = time.time()
    _save_conversations(_conversations)
    return {"id": cid, **_conversations[cid]}

@app.delete("/api/conversations/{cid}")
async def delete_conversation(cid: str):
    _conversations.pop(cid, None)
    _save_conversations(_conversations)
    return {"ok": True}


# ===========================================================================
# /api/status  (GPU + process status)
# ===========================================================================
@app.get("/api/status")
async def get_status():
    def pstat(key: str) -> dict:
        p = _procs.get(key)
        if p is None:
            return {"running": False, "pid": None, "exit_code": None}
        rc = p.poll()
        return {"running": rc is None, "pid": p.pid, "exit_code": rc}

    gpus: list[dict] = []
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 6:
                mu, mt = int(p[2]), int(p[3])
                gpus.append({
                    "id": p[0], "name": p[1],
                    "mem_used_mib": mu, "mem_total_mib": mt,
                    "mem_pct": round(mu / mt * 100, 1) if mt else 0,
                    "util_pct": int(p[4]),
                    "temp_c": int(p[5]),
                })
    except Exception:
        pass

    # Check if llama server is actually responding
    server_ok = False
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r2 = await client.get(f"{_cfg['server_url']}/health")
            server_ok = r2.status_code == 200
    except Exception:
        pass

    return {
        "processes": {k: pstat(k) for k in _procs},
        "llm_server_reachable": server_ok,
        "llm_server_url": _cfg["server_url"],
        "gpus": gpus,
    }


# ===========================================================================
# /api/control/server
# ===========================================================================
@app.post("/api/control/server/start")
async def server_start(req: ServerStartReq):
    p = _procs["server"]
    if p and p.poll() is None:
        return JSONResponse({"ok": False, "error": "서버가 이미 실행 중입니다."})
    if not Path(req.model_path).is_file():
        return JSONResponse({"ok": False, "error": f"모델 파일 없음: {req.model_path}"})

    env = os.environ.copy()
    env.update({
        "CACHE_TYPE": req.cache_type,
        "CTX_SIZE": str(req.ctx_size),
        "PORT": str(req.port),
        "NGL": str(req.ngl),
    })
    # Update server URL to match configured port
    _cfg["server_url"] = f"http://localhost:{req.port}"

    _logs["server"].clear()
    proc = subprocess.Popen(
        ["bash", str(SCRIPTS_DIR / "launch_server.sh"), "-m", req.model_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, preexec_fn=os.setsid,
    )
    _procs["server"] = proc
    threading.Thread(target=_tail_proc, args=(proc, "server"), daemon=True).start()
    return {"ok": True, "pid": proc.pid}

@app.post("/api/control/server/stop")
async def server_stop():
    p = _procs["server"]
    if p is None or p.poll() is not None:
        return {"ok": False, "error": "실행 중인 서버가 없습니다."}
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    _procs["server"] = None
    _logs["server"].append("[WebUI] 서버 종료됨.")
    return {"ok": True}


# ===========================================================================
# /api/control/tests
# ===========================================================================
@app.post("/api/control/tests/run")
async def tests_run(req: TestRunReq):
    p = _procs["tests"]
    if p and p.poll() is None:
        return {"ok": False, "error": "테스트가 이미 실행 중입니다."}

    flag_map = {"quick": "--quick", "unit": "--unit", "full": ""}
    flag = flag_map.get(req.mode, "--quick")
    cmd = ["bash", str(SCRIPTS_DIR / "run_tests.sh")] + ([flag] if flag else [])

    _logs["tests"].clear()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT), preexec_fn=os.setsid,
    )
    _procs["tests"] = proc
    threading.Thread(target=_tail_proc, args=(proc, "tests"), daemon=True).start()
    return {"ok": True, "pid": proc.pid}


# ===========================================================================
# /api/control/bench
# ===========================================================================
@app.post("/api/control/bench/run")
async def bench_run(req: BenchRunReq):
    p = _procs["bench"]
    if p and p.poll() is None:
        return {"ok": False, "error": "벤치마크가 이미 실행 중입니다."}
    if not Path(req.model_path).is_file():
        return JSONResponse({"ok": False, "error": f"모델 파일 없음: {req.model_path}"})

    cmd = ["bash", str(SCRIPTS_DIR / "turbo-quick-bench.sh")]
    if req.no_ref:
        cmd.append("--no-ref")
    cmd.append(req.model_path)
    if req.llama_dir:
        cmd.append(req.llama_dir)

    _logs["bench"].clear()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT), preexec_fn=os.setsid,
    )
    _procs["bench"] = proc
    threading.Thread(target=_tail_proc, args=(proc, "bench"), daemon=True).start()
    return {"ok": True, "pid": proc.pid}

@app.post("/api/control/bench/stop")
async def bench_stop():
    p = _procs["bench"]
    if p is None or p.poll() is not None:
        return {"ok": False, "error": "실행 중인 벤치마크가 없습니다."}
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    _procs["bench"] = None
    return {"ok": True}


# ===========================================================================
# /api/logs/{key}  (polling-based log retrieval)
# ===========================================================================
@app.get("/api/logs/{key}")
async def get_logs(key: str, since: int = 0):
    if key not in _logs:
        return JSONResponse({"error": "unknown key"}, status_code=404)
    p = _procs.get(key)
    lines = _logs[key]
    return {
        "lines": lines[since:],
        "total": len(lines),
        "running": p is not None and p.poll() is None,
        "exit_code": p.poll() if p else None,
    }


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    print(f"\n  TurboQuant WebUI → http://localhost:{port}\n")
    uvicorn.run("app:app", host="0.0.0.0", port=port,
                reload=False, log_level="warning",
                app_dir=str(Path(__file__).parent))
