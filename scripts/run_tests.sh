#!/usr/bin/env bash
# run_tests.sh — TurboQuant 테스트 러너
#
# 사용법:
#   bash scripts/run_tests.sh           # 전체 테스트
#   bash scripts/run_tests.sh --quick   # 핵심 모듈만 빠르게
#   bash scripts/run_tests.sh --unit    # 단위 테스트만 (hw/niah 제외)
#   bash scripts/run_tests.sh --file tests/test_turboquant.py  # 특정 파일
#   bash scripts/run_tests.sh --cov     # 전체 + 커버리지 리포트
#
# Env 오버라이드:
#   PYTHON    — 사용할 파이썬 인터프리터 (default: .venv/bin/python or python3)
#   PYTEST_ARGS — pytest에 직접 전달할 추가 인자

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 인터프리터 탐색
# ---------------------------------------------------------------------------
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo "ERROR: 파이썬을 찾을 수 없습니다. PYTHON 환경변수를 설정하세요."
    exit 1
fi

echo "Python: $($PYTHON_BIN --version 2>&1)"

# pytest 존재 확인
if ! $PYTHON_BIN -m pytest --version &>/dev/null; then
    echo "ERROR: pytest가 설치되지 않았습니다."
    echo "  pip install pytest"
    exit 1
fi

# ---------------------------------------------------------------------------
# 옵션 파싱
# ---------------------------------------------------------------------------
MODE="full"
SPECIFIC_FILE=""
COVERAGE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)   MODE="quick"; shift ;;
        --unit)    MODE="unit";  shift ;;
        --cov|--coverage) COVERAGE=1; shift ;;
        --file)    SPECIFIC_FILE="$2"; MODE="file"; shift 2 ;;
        -*)        echo "알 수 없는 옵션: $1"; exit 1 ;;
        *)         SPECIFIC_FILE="$1"; MODE="file"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# 테스트 대상 설정
# ---------------------------------------------------------------------------
# 핵심 모듈 테스트 (외부 의존성 없음)
CORE_TESTS=(
    "tests/test_polar_quant.py"
    "tests/test_qjl.py"
    "tests/test_rotation.py"
    "tests/test_codebook.py"
    "tests/test_turboquant.py"
    "tests/test_turbo4.py"
    "tests/test_kv_cache.py"
    "tests/test_outlier.py"
    "tests/test_distortion.py"
    "tests/test_utils.py"
)

# 단위 테스트 (hw_replay, niah 제외)
UNIT_TESTS=("${CORE_TESTS[@]}")

# 전체 테스트
ALL_TESTS=(
    "${CORE_TESTS[@]}"
    "tests/test_hw_replay.py"
    "tests/test_niah.py"
    "tests/test_turbo_hardware_diag.py"
)

case "$MODE" in
    quick)
        TARGET_TESTS=("tests/test_turboquant.py" "tests/test_turbo4.py" "tests/test_kv_cache.py" "tests/test_polar_quant.py" "tests/test_qjl.py")
        LABEL="핵심 5개 파일 (빠른 검증)"
        ;;
    unit)
        TARGET_TESTS=("${UNIT_TESTS[@]}")
        LABEL="단위 테스트 (hw/niah 제외)"
        ;;
    file)
        TARGET_TESTS=("$SPECIFIC_FILE")
        LABEL="파일 지정: $SPECIFIC_FILE"
        ;;
    full|*)
        TARGET_TESTS=("${ALL_TESTS[@]}")
        LABEL="전체 테스트"
        ;;
esac

# ---------------------------------------------------------------------------
# 커버리지 플래그
# ---------------------------------------------------------------------------
PYTEST_EXTRA_ARGS=()
if [[ $COVERAGE -eq 1 ]]; then
    if $PYTHON_BIN -m pytest --co -q 2>/dev/null | grep -q "no tests ran" 2>/dev/null || \
       $PYTHON_BIN -c "import pytest_cov" 2>/dev/null; then
        PYTEST_EXTRA_ARGS+=(--cov=turboquant --cov-report=term-missing --cov-report=html)
    else
        echo "WARNING: pytest-cov 미설치. 커버리지 없이 실행합니다."
        echo "  pip install pytest-cov"
    fi
fi

# 존재하는 테스트 파일만 필터링
VALID_TESTS=()
for f in "${TARGET_TESTS[@]}"; do
    if [[ -f "$f" ]]; then
        VALID_TESTS+=("$f")
    else
        echo "WARNING: 파일 없음, 건너뜀: $f"
    fi
done

if [[ ${#VALID_TESTS[@]} -eq 0 ]]; then
    echo "ERROR: 실행할 테스트 파일이 없습니다."
    exit 1
fi

# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  TurboQuant 테스트 러너"
echo "========================================================"
echo "  모드      : $LABEL"
echo "  대상 파일 : ${#VALID_TESTS[@]}개"
echo "  커버리지  : $([ $COVERAGE -eq 1 ] && echo 활성화 || echo 비활성화)"
echo "========================================================"
echo ""

$PYTHON_BIN -m pytest \
    -v \
    --tb=short \
    --no-header \
    -p no:warnings \
    "${PYTEST_EXTRA_ARGS[@]+"${PYTEST_EXTRA_ARGS[@]}"}" \
    ${PYTEST_ARGS:-} \
    "${VALID_TESTS[@]}"

STATUS=$?

echo ""
echo "========================================================"
if [[ $STATUS -eq 0 ]]; then
    echo "  PASS — 모든 테스트 통과"
else
    echo "  FAIL — 테스트 실패 (exit code: $STATUS)"
fi
echo "========================================================"

exit $STATUS
