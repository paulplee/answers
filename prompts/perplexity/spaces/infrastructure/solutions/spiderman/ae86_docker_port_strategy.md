# ae86 Docker Port Strategy & Service Inventory

## Current State Analysis

Based on `inventory_ae86.json`, ae86 is running MULTIPLE docker compose projects with ports getting increasingly chaotic. Let me map what's currently running:

### Active Projects on ae86

| Project | Container | Port | Protocol | Purpose |
|---------|-----------|------|----------|---------|
| **corpus-postgres** | postgres | 5432 | TCP | Postgres database (Corpus) |
| **corpus-postgres** | corpus-airflow | 8080 | TCP | Airflow UI |
| **corpus-postgres** | corpus-scrapy | 6800 | TCP | Scrapyd spider API |
| **corpus-postgres** | corpus-mcp | 3001 | TCP | MCP Server (LLM interface) |
| **labrador** | labrador-backend-archive | 8000 | TCP | Lab automation backend |
| **labrador** | labrador-frontend | 3000 | TCP | Lab automation frontend |
| **samba** | samba | 139, 445 | TCP | File sharing (SMB) |
| **stirling-pdf** | stirling-pdf | 8081 | TCP | PDF processing tool |
| **super-productivity** | super-productivity | 8082 | TCP | Task manager UI |
| **super-productivity** | super-productivity-sync | 8083 | TCP | WebDAV sync |
| **obsidian-livesync** | couchdb | 5984 | TCP | CouchDB (Obsidian sync) |

**Issues:**
- Ports are scattered: 3000, 3001, 5432, 5984, 6800, 8000, 8080, 8081, 8082, 8083
- Multiple projects claiming similar ports (8000, 3000/3001)
- Not scalable if we add more services (future ML microservices, etc.)
- No clear strategy for internal vs external ports
- All binding to `0.0.0.0` (accessible from entire network)

---

## Proposed Port Strategy

### Philosophy: RFC 6335 + Internal Consistency

**Separate ranges by function:**
- **1000-2999**: Internal infrastructure (databases, message queues, caches)
- **3000-4999**: Application APIs (REST services)
- **5000-5999**: Development/Admin UIs and special services
- **6000-6999**: Batch/Worker services (Scrapyd, Celery, etc.)
- **8000-8999**: Third-party services (user-facing tools)

**Binding Rule:**
- **Internal services**: Bind to `127.0.0.1:port` (only accessible from localhost/Docker network)
- **Public services**: Bind to `0.0.0.0:port` (accessible from LAN, proxied via zima)

---

### Proposed ae86 Port Allocation

#### Database & Storage Layer (1000-2999)

| Port | Service | Container | Bind | Notes |
|------|---------|-----------|------|-------|
| 1000 | Postgres | corpus-postgres | `127.0.0.1` | PRIMARY database. No external access needed |
| 1001 | Redis (future) | - | `127.0.0.1` | Cache layer for APIs |
| 1002 | Elasticsearch (future) | - | `127.0.0.1` | Full-text search on papers |

**Rationale:**
- Postgres doesn't need external access; apps talk via Docker DNS
- Reserve for future caching/search infrastructure
- All bind to localhost only (zero exposure)

#### API Layer (3000-4999)

| Port | Service | Container | Bind | Notes |
|------|---------|-----------|------|-------|
| 3100 | corpus-api | corpus-api | `127.0.0.1` | Paper search, download orchestration |
| 3101 | corpus-web (internal) | corpus-web | `127.0.0.1` | Static frontend (served via nginx internally) |
| 3102 | labrador-api | labrador-backend | `127.0.0.1` | Lab automation API |
| 3103 | (reserved) | - | - | Future internal API |
| 3200 | mcp-server | corpus-mcp | `127.0.0.1` | Model Context Protocol (internal for LLM clients) |

**Rationale:**
- All internal APIs bind to `127.0.0.1` (Docker network only)
- 3000s reserved for internal microservices
- Clear naming: `31xx` = corpus, `32xx` = mcp, etc.

#### Admin UIs (5000-5999)

| Port | Service | Container | Bind | Notes |
|------|---------|-----------|------|-------|
| 5100 | Airflow UI | corpus-airflow | `127.0.0.1` | Accessed via zima proxy only |
| 5101 | Postgres Admin (pgAdmin future) | - | `127.0.0.1` | Database management (via proxy) |
| 5102 | Scrapyd Web UI | corpus-scrapy | `127.0.0.1` | Accessible via proxy for monitoring |

**Rationale:**
- Admin tools don't need direct network access
- All proxied through zima for security
- 5100s reserved for orchestration/monitoring

#### Worker & Scheduler Services (6000-6999)

| Port | Service | Container | Bind | Notes |
|------|---------|-----------|------|-------|
| 6800 | Scrapyd API | corpus-scrapy | `127.0.0.1` | Spider scheduling (internal only) |
| 6801 | (reserved) | - | - | Future job queue (Celery, RQ) |

**Rationale:**
- Worker services don't accept user requests directly
- Bind to localhost; orchestration tools access via Docker DNS
- 6000s for background processing

#### External/Third-Party Services (8000-8999)

These are user-facing tools proxied through zima; their internal ports don't matter much but should be consistent:

| Port | Service | Container | Bind | Public URL | Notes |
|------|---------|-----------|------|------------|-------|
| 8001 | labrador-frontend | labrador-frontend | `127.0.0.1` | `labrador.marina.cbnano.com` | Lab UI (proxied) |
| 8002 | stirling-pdf | stirling-pdf | `127.0.0.1` | `pdf.marina.cbnano.com` | PDF tools (proxied) |
| 8003 | super-productivity | super-productivity | `127.0.0.1` | `tasks.marina.cbnano.com` | Task manager (proxied) |

**Rationale:**
- All third-party tools are accessed ONLY via zima reverse proxy
- Internal ports are arbitrary; external users see HTTPS URLs only
- Cleaner, more secure surface

#### SMB/File Sharing (Special)

| Port | Service | Container | Bind | Notes |
|------|---------|-----------|------|-------|
| 139, 445 | Samba | samba | `0.0.0.0` | Keep as-is (network shares for file access) |

**Rationale:**
- Samba needs network access for SMB protocol
- Access controlled via credentials, not firewall
- Keep binding to `0.0.0.0` for LAN file access

#### Future: ML/Chat Services (will be on separate GPU machine)

| Port | Service | Machine | Bind | Notes |
|------|---------|---------|------|-------|
| 5000 | corpus-chat API | pegasus/RTX5090 | `127.0.0.1` | LLM inference (internal) |
| 5001 | Ollama (future) | pegasus/RTX5090 | `127.0.0.1` | Model management API |

---

## Updated docker-compose.yml for corpus-postgres

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: corpus-postgres
    restart: unless-stopped
    env_file: .env
    ports:
      # OLD: - "${POSTGRES_BIND_IP}:${POSTGRES_PORT}:5432"
      # NEW: Bind only to localhost, apps access via Docker DNS
      - "127.0.0.1:1000:5432"
    volumes:
      - ${MNT_DOCKERVOLUMESDB}/corpus-postgres/pgdata:/var/lib/postgresql/data
      - ./initdb:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB"]
      interval: 10s
      timeout: 5s
      retries: 10
    networks:
      - corpus-net

  airflow:
    image: apache/airflow:2.10.3-python3.11
    container_name: corpus-airflow
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
    env_file: .env
    environment:
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__CORE__LOAD_EXAMPLES=False
      - AIRFLOW_CONN_SCRAPYD_CONN=http://scrapy:6800  # Docker DNS
    volumes:
      - ./airflow_dags:/opt/airflow/dags
      - ${MNT_CORPUS_RAW}:/data/raw
      - ${MNT_CORPUS_CLEAN}:/data/clean
    ports:
      # Admin UI: bind to localhost, access via proxy
      - "127.0.0.1:5100:8080"
    command: standalone
    networks:
      - corpus-net

  scrapy:
    build:
      context: ./scrapy
      dockerfile: Dockerfile
    container_name: corpus-scrapy
    restart: unless-stopped
    env_file: .env
    environment:
      - POSTGRES_HOST=postgres  # Docker DNS
      - POSTGRES_PORT=5432
      - CORPUS_RAW_ROOT=/data/raw
    volumes:
      - ${MNT_DOCKERVOLUMESDB}/corpus-postgres/scrapy_jobs:/var/lib/scrapyd
      - ${MNT_CORPUS_RAW}:/data/raw
    ports:
      # Scrapyd API: internal only
      - "127.0.0.1:6800:6800"
    networks:
      - corpus-net

  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    container_name: corpus-api
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
    env_file: .env
    environment:
      - DATABASE_URL=postgresql://corpus_admin:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - SCRAPYD_URL=http://scrapy:6800  # Docker DNS
      - CHAT_SERVICE_URL=http://corpus-chat:5000  # (future, on different host)
    volumes:
      - ${MNT_CORPUS_RAW}:/tank/corpus_raw:ro
    ports:
      # Search API: internal only (accessed via reverse proxy)
      - "127.0.0.1:3100:8000"
    networks:
      - corpus-net

  web:
    build:
      context: ./web
      dockerfile: Dockerfile
    container_name: corpus-web
    restart: unless-stopped
    volumes:
      - ./web/public:/app/public:ro
    ports:
      # Frontend: internal only (served via nginx proxy)
      - "127.0.0.1:3101:80"
    depends_on:
      - api
    networks:
      - corpus-net

  mcp-server:
    build:
      context: ./mcp_server
      dockerfile: Dockerfile
    container_name: corpus-mcp
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
    env_file: .env
    environment:
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
    ports:
      # MCP API: internal only (for LLM clients on other machines)
      - "127.0.0.1:3200:3000"
    networks:
      - corpus-net

networks:
  corpus-net:
    driver: bridge
    name: corpus-network
```

---

## zima (Nginx Proxy Manager) Configuration

All external access goes through zima with TLS. Add to NPM:

```
Hostname: corpus.marina.cbnano.com
Forward to: http://10.10.1.237:3101  (corpus-web)

Hostname: corpus-api.marina.cbnano.com
Forward to: http://10.10.1.237:3100  (corpus-api)
  [Optional, for testing APIs directly]

Hostname: airflow.marina.cbnano.com
Forward to: http://10.10.1.237:5100  (corpus-airflow)

Hostname: scrapyd.marina.cbnano.com
Forward to: http://10.10.1.237:6800  (corpus-scrapy UI)

Hostname: couchdb.marina.cbnano.com
Forward to: http://10.10.1.237:5984  (obsidian-livesync CouchDB)

[... existing proxies for labrador, stirling, super-productivity ...]
```

---

## Benefits of This Strategy

1. **Clarity**: Port ranges have semantic meaning (1000s = DB, 3000s = APIs, 5000s = Admin)
2. **Security**: All internal services bind to `127.0.0.1` (zero external exposure without proxy)
3. **Scalability**: Plenty of room in each range for future microservices
4. **Consistency**: Every future service follows the same pattern
5. **Maintainability**: New team members understand port allocation at a glance
6. **No collisions**: Reserved ranges prevent port conflicts
7. **Future-proof**: GPU compute machines (pegasus, RTX 5090) follow same pattern (5000s for AI)

---

## Migration Checklist

- [ ] Update `docker-compose.yml` with new port bindings
- [ ] Test Docker DNS resolution from containers (`docker exec <container> getent hosts postgres`)
- [ ] Update `.env` if any hardcoded ports exist
- [ ] Update zima proxy rules to point to new internal ports
- [ ] Test all services still communicate (Airflow → Scrapyd, API → Postgres, etc.)
- [ ] Verify external access via zima still works (https://corpus.marina.cbnano.com)
- [ ] Document in README
- [ ] No manual port tunneling needed anymore (everything goes through zima)

---

## Long-term Extensibility

### If adding new GPU compute node (RTX 5090):

```yaml
# On RTX 5090 machine (separate docker-compose)
services:
  corpus-chat:
    # Inference service
    ports:
      - "127.0.0.1:5000:5000"  # Internal LLM API
  
  # ae86 corpus-api would reach it via:
  # CHAT_SERVICE_URL=http://10.10.1.xxx:5000
  # (or via reverse tunnel if not on same LAN)
```

### If adding Celery/Redis for async jobs:

```yaml
# ae86 docker-compose additions
redis:
  ports:
    - "127.0.0.1:1001:6379"  # Cache layer

celery-worker:
  # No external port (backgroundtask)
  environment:
    - REDIS_URL=redis://redis:6379
```

This keeps everything consistent and predictable.
