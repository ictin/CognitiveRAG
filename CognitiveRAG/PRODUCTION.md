# CognitiveRAG-production deployment recipe

This is the stable production copy for the memory MVP.

## Source branch / commit
- Branch: `feature/redesign-memory-clean`
- Use the current reviewed commit on that branch as the source of truth.
- Promote it into a separate production folder, not the dev checkout.

## Target folder
- `~/.openclaw/workspace/CognitiveRAG-production`

## Clone / checkout
```bash
cd ~/.openclaw/workspace
rm -rf CognitiveRAG-production

git clone https://github.com/ictin/CognitiveRAG.git CognitiveRAG-production
cd CognitiveRAG-production
git checkout feature/redesign-memory-clean
```

## Lightweight environment
```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install fastapi requests httpx starlette networkx rank_bm25 python-multipart pytest
```

## Run
```bash
. .venv/bin/activate
export COGNITIVERAG_SKIP_KB=1
export PYTHONPATH=./CognitiveRAG
python -m uvicorn CognitiveRAG.main_server:app --host 0.0.0.0 --port 8000
```

## OpenClaw wiring point
Point the live OpenClaw installation at the production server base URL and use these endpoints:
- `POST /session_append_message`
- `POST /session_append_message_part`
- `POST /session_upsert_context_item`
- `POST /promote_session`

Use `CognitiveRAG.client` as the write shim.

## Minimal smoke path
1. Write a turn via `mirror_agent_event` or `mirror_turn`.
2. Confirm the session write endpoints succeed.
3. Call `POST /promote_session` for the session.
4. Verify the response contains `promoted_count` and `promoted_pattern_ids`.
