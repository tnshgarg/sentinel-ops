# Fix OpenEnv Validation Failures & Hackathon Compliance Gaps

## Background

Ran `openenv validate` against both local directory and running server. Found **5 critical failures** that will cause the submission to be rejected. Also identified 2 additional issues.

## Diagnosis Summary

### `openenv validate` (local)
| Check | Status | Issue |
|-------|--------|-------|
| `multi_mode_deployment_readiness` | ❌ FAIL | Missing `pyproject.toml` |

### `openenv validate --url http://localhost:7860` (runtime)
| Check | Status | Issue |
|-------|--------|-------|
| `openapi_version_available` | ✅ PASS | — |
| [health_endpoint](file:///Users/tanish/Documents/SentinalOps/tests/test_api.py#43-49) | ❌ FAIL | Returns `"status": "ok"`, expects `"status": "healthy"` |
| `metadata_endpoint` | ❌ FAIL | `GET /metadata` returns 404 (endpoint missing) |
| `schema_endpoint` | ❌ FAIL | `GET /schema` returns 404 (endpoint missing) |
| `mcp_endpoint` | ❌ FAIL | `POST /mcp` returns 404 (endpoint missing) |
| `mode_endpoint_consistency` | ✅ PASS | — |

### Additional issues
- [requirements.txt](file:///Users/tanish/Documents/SentinalOps/requirements.txt) pins `openai>=1.12.0,<2.0.0` but `openenv-core` installed `openai 2.30.0` — the upper bound is wrong
- [requirements.txt](file:///Users/tanish/Documents/SentinalOps/requirements.txt) is missing `python-dotenv` which [inference.py](file:///Users/tanish/Documents/SentinalOps/inference.py) imports

---

## Proposed Changes

### Server (app.py)

#### [MODIFY] [app.py](file:///Users/tanish/Documents/SentinalOps/server/app.py)

1. **Fix `/health`**: Change `"status": "ok"` → `"status": "healthy"` to match OpenEnv spec
2. **Add `GET /metadata`**: Return `name`, `description` (and optionally `version`) from our openenv.yaml
3. **Add `GET /schema`**: Return JSON schemas for [action](file:///Users/tanish/Documents/SentinalOps/grader.py#74-77), [observation](file:///Users/tanish/Documents/SentinalOps/env.py#494-535), and [state](file:///Users/tanish/Documents/SentinalOps/inference.py#127-131) using Pydantic's `.model_json_schema()`
4. **Add `POST /mcp`**: Add a minimal MCP (Model Context Protocol) endpoint that responds with JSON-RPC 2.0 format

---

### Project Config

#### [NEW] [pyproject.toml](file:///Users/tanish/Documents/SentinalOps/pyproject.toml)

Create a standard `pyproject.toml` with project metadata, dependencies mirroring [requirements.txt](file:///Users/tanish/Documents/SentinalOps/requirements.txt), and entry points. This is required by `openenv validate`.

---

### Dependencies

#### [MODIFY] [requirements.txt](file:///Users/tanish/Documents/SentinalOps/requirements.txt)

- Change `openai>=1.12.0,<2.0.0` → `openai>=1.12.0` (remove upper bound since `openenv-core` requires v2.x)
- Add `python-dotenv>=1.0.0`

---

### Tests

#### [MODIFY] [test_api.py](file:///Users/tanish/Documents/SentinalOps/tests/test_api.py)

- Update health test: expect `"healthy"` instead of `"ok"`
- Add tests for new endpoints: `/metadata`, `/schema`, `/mcp`

---

## Verification Plan

### Automated Tests

```bash
# 1. Run full test suite (should be 57+ tests, all pass)
cd /Users/tanish/Documents/SentinalOps && python -m pytest tests/ -v --tb=short

# 2. Run openenv validate (local — should pass)
cd /Users/tanish/Documents/SentinalOps && openenv validate -v

# 3. Restart server and run runtime validation
# (kill existing uvicorn, restart, then validate)
cd /Users/tanish/Documents/SentinalOps && openenv validate --url http://localhost:7860 --json
```

### Manual Verification

```bash
# 4. Verify /health returns "healthy"
curl -s http://localhost:7860/health | python3 -m json.tool

# 5. Verify /metadata returns name and description
curl -s http://localhost:7860/metadata | python3 -m json.tool

# 6. Verify /schema returns action, observation, state schemas
curl -s http://localhost:7860/schema | python3 -m json.tool

# 7. Verify /mcp returns JSON-RPC 2.0 response
curl -s -X POST http://localhost:7860/mcp -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","method":"ping","id":1}' | python3 -m json.tool
```
