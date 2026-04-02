#!/usr/bin/env bash
# start_webui.sh — TurboQuant WebUI 런처
#
# 사용법:
#   bash scripts/start_webui.sh           # 기본 포트 7860
#   bash scripts/start_webui.sh 8888      # 포트 지정
#   bash scripts/start_webui.sh --dev     # 핫 리로드 모드
#
# Env 오버라이드:
#   WEBUI_PORT    — WebUI 포트 (default: 7860)
#   WEBUI_HOST    — 바인딩 주소 (default: 0.0.0.0)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 옵션 파싱
# ---------------------------------------------------------------------------
WEBUI_PORT="${WEBUI_PORT:-7860}"
WEBUI_HOST="${WEBUI_HOST:-0.0.0.0}"
DEV_MODE=0

for arg in "$@"; do
  case "$arg" in
    --dev)  DEV_MODE=1 ;;
    [0-9]*) WEBUI_PORT="$arg" ;;
    *) echo "알 수 없는 옵션: $arg"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# 파이썬 인터프리터 탐색
# ---------------------------------------------------------------------------
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
  PIP="$REPO_ROOT/.venv/bin/pip"
elif command -v python3 &>/dev/null; then
  PYTHON="python3"
  PIP="python3 -m pip"
else
  echo "ERROR: 파이썬을 찾을 수 없습니다."
  exit 1
fi

echo "Python: $($PYTHON --version 2>&1)"

# ---------------------------------------------------------------------------
# 의존성 확인 & 설치
# ---------------------------------------------------------------------------
echo "의존성 확인 중..."

MISSING=()
$PYTHON -c "import fastapi"  2>/dev/null || MISSING+=("fastapi>=0.110")
$PYTHON -c "import uvicorn"  2>/dev/null || MISSING+=("uvicorn[standard]>=0.27")
$PYTHON -c "import httpx"    2>/dev/null || MISSING+=("httpx>=0.27")

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo ""
  echo "  설치 필요: ${MISSING[*]}"
  echo "  설치 중..."
  $PIP install --quiet "${MISSING[@]}"
  echo "  완료."
fi

# ---------------------------------------------------------------------------
# data 디렉토리 생성
# ---------------------------------------------------------------------------
mkdir -p webapp/data

# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')

echo ""
echo "========================================================"
echo "  TurboQuant WebUI"
echo "========================================================"
echo "  로컬    : http://localhost:${WEBUI_PORT}"
if [[ -n "$LAN_IP" ]]; then
  echo "  LAN     : http://${LAN_IP}:${WEBUI_PORT}  ← 다른 PC에서 이 주소 사용"
fi
echo "  Host : ${WEBUI_HOST}"
echo "  Mode : $([ $DEV_MODE -eq 1 ] && echo 'dev (hot reload)' || echo 'production')"
echo "========================================================"
echo ""

if [[ $DEV_MODE -eq 1 ]]; then
  cd webapp
  exec $PYTHON -m uvicorn app:app \
    --host "$WEBUI_HOST" --port "$WEBUI_PORT" \
    --reload --reload-dir . \
    --log-level info
else
  cd webapp
  exec $PYTHON app.py "$WEBUI_PORT"
fi
