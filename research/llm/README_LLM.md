## RxInferLLM — how to run with OpenAI key

1. Copy `.env.example` to `.env` and set your `OPENAI_KEY` there (do NOT commit `.env`).

2. Run locally (example):

```bash
export OPENAI_KEY="sk-..."
julia --project=. -e 'using Pkg; Pkg.instantiate(); using RxInferLLM; RxInferLLM.main()'
```

Or use the helper script:

```bash
./scripts/run_with_key.sh <OPENAI_KEY>
```

Tests that require a live LLM will be skipped when `OPENAI_KEY` is not set. This keeps CI deterministic.


