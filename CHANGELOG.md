# Changelog

All notable changes to TurboQuant are documented here.

---

## [Unreleased]

### Added — TurboQuant WebUI (`webapp/`)

Open WebUI-inspired web control panel for TurboQuant.  
Architecture mirrors Open WebUI: FastAPI backend · SSE streaming · Pipe middleware.

#### `webapp/app.py` — FastAPI backend
- `/api/chat/completions` — OpenAI-compatible chat proxy with SSE streaming to llama.cpp server
- `/api/models` — model list from running server (falls back to stub when offline)
- `/api/conversations` — conversation CRUD (persisted to `webapp/data/conversations.json`)
- `/api/config` GET/PATCH — runtime config (server URL, temperature, system prompt, max tokens)
- `/api/status` — GPU info (nvidia-smi), process status, llama server health check
- `/api/control/server/start|stop` — launch/terminate llama-server via `launch_server.sh`
- `/api/control/tests/run` — run pytest suite via `run_tests.sh`
- `/api/control/bench/run|stop` — run/stop `turbo-quick-bench.sh`
- `/api/logs/{key}?since=N` — polling-based log retrieval for server/tests/bench

#### `webapp/pipes/turboquant_pipe.py` — TurboQuant Pipe
- `inlet()` — injects cache-type metadata into system prompt before sending to LLM
- `outlet()` — optional stats footer on assistant responses (cache type, tok/s)
- Follows Open WebUI's inlet/outlet middleware pattern for future plugin-loader compatibility

#### `webapp/static/index.html` — Single-Page App
- Chat interface with markdown rendering (marked.js) and syntax highlighting (highlight.js)
- Streaming response display via Fetch API + ReadableStream
- Conversation sidebar with create/load/delete
- Right-side control drawer with 5 tabs:
  - **서버** — model path, cache type, ctx size, port, NGL; start/stop; live log
  - **테스트** — mode select (quick/unit/full); live pytest log with pass/fail coloring
  - **벤치** — turbo-quick-bench runner with --no-ref toggle; live log; stop button
  - **GPU** — per-GPU VRAM bar, utilization bar, temperature (auto-refresh 3s)
  - **설정** — server URL, system prompt, temperature, max tokens
- Top bar: model selector, server status dot (green/red), GPU mini-bars
- Fully offline-capable for chat settings (only CDN: marked.js, highlight.js)

#### `scripts/start_webui.sh` — WebUI launcher
- Auto-installs FastAPI + uvicorn + httpx if missing
- `--dev` flag for hot-reload mode
- Port configurable via argument or `WEBUI_PORT` env

#### `scripts/run_tests.sh` — Test runner shell script (added 2026-04-02)
- Modes: `--quick` (5 core files), `--unit` (all unit tests), `--full` (including hw/niah)
- Pass/fail exit codes, `--cov` for coverage report
- `PYTEST_ARGS` env passthrough

#### `pyproject.toml`
- Added `[project.optional-dependencies] web` group: `fastapi>=0.110`, `uvicorn[standard]>=0.27`, `httpx>=0.27`
- Install: `pip install -e ".[web]"`

### Usage

```bash
# 빠른 시작 (의존성 자동 설치)
bash scripts/start_webui.sh

# 개발 모드 (핫 리로드)
bash scripts/start_webui.sh --dev

# 포트 지정
bash scripts/start_webui.sh 8888

# 직접 설치 후 실행
pip install -e ".[web]"
cd webapp && python app.py
```

브라우저에서 `http://localhost:7860` 접속.

---

## [0.1.0] — 2026-03-xx (initial)

- `turboquant/` — core library: PolarQuant, QJL, TurboQuant, TurboQuantMSE, KVCacheCompressor
- `tests/` — pytest suite (13 test files)
- `scripts/launch_server.sh` — multi-GPU llama-server launcher with VRAM-aware tensor split
- `scripts/turbo-quality-gate.sh` — PPL + context-scaling regression gate
- `scripts/turbo-quick-bench.sh` — rapid benchmark (PPL, decode speed, NIAH)
- `scripts/turbo-realworld-bench.sh` — full real-world benchmark suite
- `scripts/turbo_hardware_diag.py` — hardware diagnostic tool (zip output)
- `benchmarks/` — benchmark scripts and results
- `docs/` — research notes, ablation logs, paper drafts
