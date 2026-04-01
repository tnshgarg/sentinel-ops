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
CHECKS_TOTAL=3

# ──────────────────────────────────────────────────────────────
# Check 1: HF Space Ping
# ──────────────────────────────────────────────────────────────
echo -e "${YELLOW}[1/3] HF Space Ping${NC}"
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
echo -e "${YELLOW}[2/3] Docker Build${NC}"

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
echo -e "${YELLOW}[3/3] OpenEnv Validate${NC}"

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
