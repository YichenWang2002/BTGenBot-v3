# OpenCode Zen / MiniMax M2.5

This guide shows how to run the OpenAI-compatible LLM repair benchmark with
OpenCode Zen and MiniMax M2.5.

## Environment

Set the required provider variables:

```bash
export OPENAI_BASE_URL="https://opencode.ai/zen/v1"
export OPENAI_API_KEY="..."
export OPENAI_MODEL="minimax-m2.5-free"
export OPENAI_PROVIDER_PROFILE="opencode-zen"
export OPENAI_DISABLE_RESPONSE_FORMAT="1"
export OPENAI_MAX_RETRIES="3"
export OPENAI_DEBUG_REQUEST="1"
```

Optional model fallbacks:

```bash
export OPENAI_MODEL_FALLBACKS="minimax-m2.5-free,opencode/minimax-m2.5-free,minimax-m2-5-free"
```

## Why Disable `response_format`

MiniMax M2.5 and some OpenAI-compatible endpoints may reject
`response_format=json_object` or JSON schema parameters. This project therefore
uses prompt instructions, the output parser, and the schema validator to enforce
structured JSON output.

For OpenCode Zen / MiniMax, keep:

```bash
export OPENAI_PROVIDER_PROFILE="opencode-zen"
export OPENAI_DISABLE_RESPONSE_FORMAT="1"
```

## Why Retry Is Needed

The LLM repair prompts and responses can be long. Some providers may return
network-level failures such as `IncompleteRead`, connection reset, timeout, or
temporary 5xx/429 errors. The backend retries transient failures and records
provider failures as `provider_error` rows in `raw.jsonl` instead of aborting the
whole suite.

## Recommended Run Order

First verify the endpoint with a minimal request:

```bash
bash scripts/test_opencode_zen_curl.sh
```

Then run a 2-run debug benchmark:

```bash
bash scripts/run_opencode_debug.sh
```

If that works, run a 10-run debug pass manually:

```bash
python -m src.repair.refinement_loop \
  --suite configs/experiment_suites/llm_repair_small.yaml \
  --backend openai-compatible \
  --max-iters 3 \
  --max-runs 10 \
  --output-dir results/llm_repair_small_opencode_debug10
```

Finally run the full 60-run small benchmark:

```bash
bash scripts/run_opencode_small.sh
```

## Outputs

Debug run:

- `results/llm_repair_small_opencode_debug/raw.jsonl`
- `results/llm_repair_small_opencode_debug/summary.csv`
- `results/llm_repair_small_opencode_debug/summary.md`
- `results/llm_repair_small_opencode_debug/runs/`

Full small run:

- `results/llm_repair_small_opencode_minimax/raw.jsonl`
- `results/llm_repair_small_opencode_minimax/summary.csv`
- `results/llm_repair_small_opencode_minimax/summary.md`
- `results/llm_repair_small_opencode_minimax/runs/`

When `OPENAI_DEBUG_REQUEST=1`, provider debug files are written to:

- `results/debug/openai_request_last.json`
- `results/debug/openai_response_last.txt`
- `results/debug/openai_error_last.txt`

These files are sanitized by the backend and must not contain the API key.

## Safety

- Do not commit `OPENAI_API_KEY` to git.
- Do not paste API keys into logs, issues, paper artifacts, or shared result
  bundles.
- Avoid shell tracing such as `set -x` while running provider scripts.
- If a key is exposed, revoke it immediately and create a new key.
