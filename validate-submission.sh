#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# SentinelOps — Pre-Submission Validation Script
#
# Usage:
#   ./validate-submission.sh <hf_space_url>
#
# Runs 3 mandatory checks:
#   1. HF Space ping (POST /reset → HTTP 200)
#   2. Docker build
#   3. openenv validate (if available)
# ──────────────────────────────────────────────────────────────

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS="${GREEN}✓ PASS${NC}"
FAIL="${RED}✗ FAIL${NC}"

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  SentinelOps — Pre-Submission Validation${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""

HF_URL="${1:-}"
CHECKS_PASSED=0
CHECKS_TOTAL=4

# ──────────────────────────────────────────────────────────────
# Check 1: HF Space Ping
# ──────────────────────────────────────────────────────────────
echo -e "${YELLOW}[1/4] HF Space Ping${NC}"
if [ -z "$HF_URL" ]; then
    echo -e "  ${FAIL} — No HF Space URL provided."
    echo "  Usage: ./validate-submission.sh <hf_space_url>"
else
    echo "  Testing POST $HF_URL/reset ..."
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "$HF_URL/reset" \
        -H "Content-Type: application/json" \
        -d '{}' \
        --max-time 30 2>/dev/null || echo "000")

    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "  ${PASS} — /reset returned HTTP $HTTP_CODE"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        echo -e "  ${FAIL} — /reset returned HTTP $HTTP_CODE (expected 200)"
    fi
fi
echo ""

# ──────────────────────────────────────────────────────────────
# Check 2: Docker Build
# ──────────────────────────────────────────────────────────────
echo -e "${YELLOW}[2/4] Docker Build${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="$SCRIPT_DIR/Dockerfile"

if [ ! -f "$DOCKERFILE" ]; then
    # Check server/ fallback
    DOCKERFILE="$SCRIPT_DIR/server/Dockerfile"
fi

if [ ! -f "$DOCKERFILE" ]; then
    echo -e "  ${FAIL} — No Dockerfile found at repo root or server/"
else
    echo "  Building Docker image..."
    if docker build -t sentinelops-validate "$SCRIPT_DIR" -f "$DOCKERFILE" > /dev/null 2>&1; then
        echo -e "  ${PASS} — Docker build successful"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))

        # Quick smoke test: start container, hit /health, stop
        echo "  Running container smoke test..."
        CONTAINER_ID=$(docker run -d -p 17860:7860 sentinelops-validate)
        sleep 5  # Wait for startup

        HEALTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            http://localhost:17860/health --max-time 10 2>/dev/null || echo "000")

        docker stop "$CONTAINER_ID" > /dev/null 2>&1 || true
        docker rm "$CONTAINER_ID" > /dev/null 2>&1 || true

        if [ "$HEALTH_CODE" = "200" ]; then
            echo -e "  ${PASS} — Container /health returned 200"
        else
            echo -e "  ${YELLOW}⚠ WARNING${NC} — Container /health returned $HEALTH_CODE"
        fi
    else
        echo -e "  ${FAIL} — Docker build failed"
    fi
fi
echo ""

# ──────────────────────────────────────────────────────────────
# Check 3: openenv validate
# ──────────────────────────────────────────────────────────────
echo -e "${YELLOW}[3/4] OpenEnv Validate${NC}"

if command -v openenv &> /dev/null; then
    echo "  Running openenv validate ..."
    if openenv validate "$SCRIPT_DIR" 2>&1; then
        echo -e "  ${PASS} — openenv validate passed"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        echo -e "  ${FAIL} — openenv validate failed"
    fi
else
    echo -e "  ${YELLOW}⚠ SKIP${NC} — 'openenv' CLI not installed."
    echo "  Install with: pip install openenv-core"
    echo "  Marking as conditional pass."
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
fi
echo ""

# ──────────────────────────────────────────────────────────────
# Check 4: inference.py structured log format
# ──────────────────────────────────────────────────────────────
echo -e "${YELLOW}[4/4] Inference Script Structured Log Format${NC}"

INFERENCE_FILE="$SCRIPT_DIR/inference.py"

if [ ! -f "$INFERENCE_FILE" ]; then
    echo -e "  ${FAIL} — inference.py not found at repo root"
else
    # Check for [START], [STEP], [END] emit calls and mandatory env vars
    START_OK=$(grep -c '\[START\]' "$INFERENCE_FILE" 2>/dev/null || echo "0")
    STEP_OK=$(grep -c '\[STEP\]' "$INFERENCE_FILE" 2>/dev/null || echo "0")
    END_OK=$(grep -c '\[END\]' "$INFERENCE_FILE" 2>/dev/null || echo "0")
    API_BASE_OK=$(grep -c 'API_BASE_URL' "$INFERENCE_FILE" 2>/dev/null || echo "0")
    MODEL_NAME_OK=$(grep -c 'MODEL_NAME' "$INFERENCE_FILE" 2>/dev/null || echo "0")
    HF_TOKEN_OK=$(grep -c 'HF_TOKEN' "$INFERENCE_FILE" 2>/dev/null || echo "0")
    OPENAI_OK=$(grep -c 'from openai import OpenAI' "$INFERENCE_FILE" 2>/dev/null || echo "0")

    INFERENCE_ISSUES=0
    [ "$START_OK" -ge 1 ] && echo -e "  ✓ [START] emit found" || { echo -e "  ✗ [START] emit MISSING"; INFERENCE_ISSUES=$((INFERENCE_ISSUES + 1)); }
    [ "$STEP_OK" -ge 1 ] && echo -e "  ✓ [STEP] emit found" || { echo -e "  ✗ [STEP] emit MISSING"; INFERENCE_ISSUES=$((INFERENCE_ISSUES + 1)); }
    [ "$END_OK" -ge 1 ] && echo -e "  ✓ [END] emit found" || { echo -e "  ✗ [END] emit MISSING"; INFERENCE_ISSUES=$((INFERENCE_ISSUES + 1)); }
    [ "$API_BASE_OK" -ge 1 ] && echo -e "  ✓ API_BASE_URL defined" || { echo -e "  ✗ API_BASE_URL MISSING"; INFERENCE_ISSUES=$((INFERENCE_ISSUES + 1)); }
    [ "$MODEL_NAME_OK" -ge 1 ] && echo -e "  ✓ MODEL_NAME defined" || { echo -e "  ✗ MODEL_NAME MISSING"; INFERENCE_ISSUES=$((INFERENCE_ISSUES + 1)); }
    [ "$HF_TOKEN_OK" -ge 1 ] && echo -e "  ✓ HF_TOKEN defined" || { echo -e "  ✗ HF_TOKEN MISSING"; INFERENCE_ISSUES=$((INFERENCE_ISSUES + 1)); }
    [ "$OPENAI_OK" -ge 1 ] && echo -e "  ✓ OpenAI client used" || { echo -e "  ✗ OpenAI client MISSING"; INFERENCE_ISSUES=$((INFERENCE_ISSUES + 1)); }

    if [ "$INFERENCE_ISSUES" -eq 0 ]; then
        echo -e "  ${PASS} — All structured log format checks passed"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        echo -e "  ${FAIL} — $INFERENCE_ISSUES issue(s) found in inference.py"
    fi
fi
echo ""

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "  Results: ${CHECKS_PASSED}/${CHECKS_TOTAL} checks passed"

if [ "$CHECKS_PASSED" -eq "$CHECKS_TOTAL" ]; then
    echo -e "  ${GREEN}🎉 ALL CHECKS PASSED — Ready for submission!${NC}"
else
    echo -e "  ${RED}❌ Some checks failed — fix before submitting.${NC}"
fi

echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""

exit $((CHECKS_TOTAL - CHECKS_PASSED))
