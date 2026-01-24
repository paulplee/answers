# CB Nano R&D AI Assistant Service — OpenWebUI Playbook

Last updated: 2026-01-24

This playbook covers the **next steps after Web Search is already working** (SearXNG integration complete), focusing on: (1) building a useful internal knowledge collection quickly, (2) role-based assistants, (3) small “tooling” upgrades for scientist productivity, and (4) a practical rollout plan.

## Current baseline (for reference)

- `pegasus` runs OpenWebUI (published port 3000 → container 8080) and Ollama (11434). [file:1]
- `ae86` hosts the Corpus stack (Postgres + pgvector + Airflow + Scrapyd + MCP server) and SearXNG (published port 8180 → container 8080). [file:2][file:4]
- External access (when needed) is typically routed via `zima` using Nginx Proxy Manager. [file:4]

---

## Step 1 — Create a persistent “Lab Knowledge” collection

Goal: Scientists can chat with internal PDFs/notes in OpenWebUI immediately, and you can evolve this into more structured RAG later.

### 1.1 Decide the canonical storage location (source of truth)

Choose a stable directory on `ae86` (ZFS-backed) as the canonical knowledge store, for example:

- `/tank/ai-kb/internal/` (your internal docs, slides, SOPs)
- `/tank/ai-kb/standards/` (standards, datasheets, safety)
- `/tank/ai-kb/papers/` (curated PDFs you want “in the lab KB”)

Suggested commands (on `ae86`):

```bash
sudo mkdir -p /tank/ai-kb/{internal,standards,papers,dropbox}
sudo chown -R $USER:$USER /tank/ai-kb
