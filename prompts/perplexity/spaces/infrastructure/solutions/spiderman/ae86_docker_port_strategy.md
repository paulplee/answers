# ae86 Docker Port Strategy & Service Inventory

## Current State Analysis

Based on `inventory_ae86.json`, ae86 is running MULTIPLE docker compose projects with ports getting increasingly chaotic. Let me map what's currently running:

### Active Projects on ae86

| Project | Container | Current Port | Protocol | Purpose |
|---------|-----------|--------------|----------|---------|
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

### Philosophy: RFC 6335 + Hierarchical Spacing

**Separate ranges by function:**
- **1000-1999**: Internal infrastructure (databases, message queues, caches)
- **3000-3999**: Application APIs (REST services)
- **5000-5999**: Development/Admin UIs and special services
- **6000-6999**: Batch/Worker services (Scrapyd, Celery, etc.)
- **8000-8999**: Third-party services (user-facing tools)

**Port Spacing Rule (New):**
Each project gets a 10-digit block. Within that block, related services stay together:
- **X100-X109**: Main services
- **X110-X119**: Secondary services
- **X120-X129**: Related services
- **X130-X139**: Admin/monitoring
- etc.

**Binding Rule:**
- **Internal services**: Bind to `127.0.0.1:port` (only accessible from localhost/Docker network)
- **Public services**: Bind to `0.0.0.0:port` (accessible from LAN, proxied via zima)

This approach allows **easy visual hierarchy** and **room for expansion within project boundaries**.

---

### Proposed ae86 Port Allocation

#### Database & Storage Layer (1000-1999)

| Port | Service | Container | Bind | Notes |
|------|---------|-----------|------|-------|
| 1000 | Postgres | corpus-postgres | `127.0.0.1` | PRIMARY database. No external access needed |
| 1001 | Redis (future) | - | `127.0.0.1` | Cache layer for APIs |
| 1002 | Elasticsearch (future) | - | `127.0.0.1` | Full-text search on papers |

**Rationale:**
- Postgres doesn't need external access; apps talk via Docker DNS
- Reserve next 9 ports for future database infrastructure
- All bind to localhost only (zero exposure)

---

#### API Layer (3000-3999)

Projects are assigned 10-port blocks within 3000-3999:

##### Corpus Project (3100-3109)

| Port | Service | Container | Bind | Notes |
|------|---------|-----------|------|-------|
| 3100 | corpus-api | corpus-api | `127.0.0.1` | Paper search, download orchestration |
| 3101 | corpus-web | corpus-web | `127.0.0.1` | Static frontend (served via nginx internally) |
| 3102-3109 | (reserved) | - | - | Future corpus-related APIs |

##### Labrador Project (3110-3119)

| Port | Service | Container | Bind | Notes |
|------|---------|-----------|------|-------|
| 3110 | labrador-api | labrador-backend | `127.0.0.1` | Lab automation API |
| 3111 | labrador-web | labrador-frontend | `127.0.0.1` | Lab automation UI |
| 3112-3119 | (reserved) | - | - | Future labrador-related services |

##### MCP & LLM Integration (3120-3129)

| Port | Service | Container | Bind | Notes |
|------|---------|-----------|------|-------|
| 3120 | mcp-server | corpus-mcp | `127.0.0.1` | Model Context Protocol (LLM interface) |
| 3121 | corpus-rag-api | (future) | `127.0.0.1` | RAG retrieval for chat |
| 3122-3129 | (reserved) | - | - | Future LLM-related APIs |

##### Future API Projects (3130-3999)

Reserved for future microservices:
- **3130-3139**: Future project A
- **3140-3149**: Future project B
- **3150-3159**: Future project C
- etc.

---

#### Admin UIs & Monitoring (5000-5999)

Projects are assigned 10-port blocks within 5000-5999:

##### Corpus Project Admin (5100-5109)

| Port | Service | Container | Bind | Notes |
|------|---------|-----------|------|-------|
| 5100 | Airflow UI | corpus-airflow | `127.0.0.1` | Accessed via zima proxy only |
| 5101 | pgAdmin (future) | - | `127.0.0.1` | Database management (via proxy) |
| 5102 | Scrapyd Web UI | corpus-scrapy | `127.0.0.1` | Spider monitoring (via proxy) |
| 5103-5109 | (reserved) | - | - | Future corpus admin tools |

##### Labrador Admin (5110-5119)

| Port | Service | Container | Bind | Notes |
|------|---------|-----------|------|-------|
| 5110 | labrador-admin | (future) | `127.0.0.1` | Lab experiment logs/management |
| 5111-5119 | (reserved) | - | - | Future labrador monitoring |

##### System Monitoring (5120-5129)

| Port | Service | Container | Bind | Notes |
|------|---------|-----------|------|-------|
| 5120 | Prometheus (future) | - | `127.0.0.1` | Metrics collection |
| 5121 | Grafana (future) | - | `127.0.0.1` | Dashboards (via proxy) |
| 5122-5129 | (reserved) | - | - | Future monitoring tools |

---

#### Worker & Scheduler Services (6000-6999)

Projects are assigned 10-port blocks within 6000-6999:

##### Corpus Project Workers (6100-6109)

| Port | Service | Container | Bind | Notes |
|------|---------|-----------|------|-------|
| 6100 | Scrapyd API | corpus-scrapy | `127.0.0.1` | Spider scheduling (internal only) |
| 6101 | Celery Worker (future) | - | `127.0.0.1` | Async job processing |
| 6102-6109 | (reserved) | - | - | Future corpus workers |

##### Labrador Workers (6110-6119)

| Port | Service | Container | Bind | Notes |
|------|---------|-----------|------|-------|
| 6110 | Lab Experiment Queue | (future) | `127.0.0.1` | Equipment control queuing |
| 6111-6119 | (reserved) | - | - | Future lab workers |

##### System Workers (6120-6129)

| Port | Service | Container | Bind | Notes |
|------|---------|-----------|------|-------|
| 6120 | Message Queue (future) | - | `127.0.0.1` | Redis PubSub or RabbitMQ |
| 6121-6129 | (reserved) | - | - | Future async infrastructure |

---

#### External/Third-Party Services (8000-8999)

These are user-facing tools proxied through zima; we assign 10-port blocks:

##### Productivity Tools (8100-8109)

| Port | Service | Container | Bind | Public URL | Notes |
|------|---------|-----------|------|------------|-------|
| 8100 | super-productivity | super-productivity | `127.0.0.1` | `tasks.marina.cbnano.com` | Task manager (proxied) |
| 8101 | super-productivity-sync | super-productivity-sync | `127.0.0.1` | (via 8100) | WebDAV sync backend |
| 8102-8109 | (reserved) | - | - | - | Future productivity apps |

##### Document Tools (8110-8119)

| Port | Service | Container | Bind | Public URL | Notes |
|------|---------|-----------|------|------------|-------|
| 8110 | stirling-pdf | stirling-pdf | `127.0.0.1` | `pdf.marina.cbnano.com` | PDF tools (proxied) |
| 8111-8119 | (reserved) | - | - | - | Future document processing |

##### Data Tools (8120-8129)

| Port | Service | Container | Bind | Public URL | Notes |
|------|---------|-----------|------|------------|-------|
| 8120 | obsidian-livesync (CouchDB) | couchdb | `127.0.0.1` | `couchdb.marina.cbnano.com` | Note sync backend |
| 8121-8129 | (reserved) | - | - | - | Future data sync tools |

##### Lab Tools (8130-8139)

| Port | Service | Container | Bind | Public URL | Notes |
|------|---------|-----------|------|------------|-------|
| 8130 | labrador-web (public) | labrador-frontend | `127.0.0.1` | `lab.marina.cbnano.com` | Lab UI (proxied) |
| 8131-8139 | (reserved) | - | - | - | Future lab public interfaces |

---

#### SMB/File Sharing (Special Range)

| Port | Service | Container | Bind | Notes |
|------|---------|-----------|------|-------|
| 139 | Samba NetBIOS | samba | `0.0.0.0` | Keep on special ports (protocol requirement) |
| 445 | Samba SMB | samba | `0.0.0.0` | Keep on standard SMB port |

**Rationale:**
- Samba needs network access for SMB protocol
- Access controlled via credentials, not firewall
- Standard ports (139, 445) are required by Windows clients
- Keep binding to `0.0.0.0` for LAN file access

---

#### Future: ML/Chat Services (on separate GPU machine)

When Phase 2 GPU node joins:

##### GPU Machine API Layer (3200-3209)

| Port | Service | Machine | Bind | Notes |
|------|---------|---------|------|-------|
| 3200 | corpus-chat-api | pegasus/RTX5090 | `127.0.0.1` | LLM inference (internal) |
| 3201 | (reserved) | - | - | Future chat APIs |

##### GPU Machine Admin (5200-5209)

| Port | Service | Machine | Bind | Notes |
|------|---------|---------|------|-------|
| 5200 | Ollama UI (future) | pegasus/RTX5090 | `127.0.0.1` | Model management (via proxy) |
| 5201-5209 | (reserved) | - | - | Future GPU monitoring |

---

## Benefits of This Strategy

1. **Visual Clarity**: Port number immediately tells you: range (1000s=DB, 3000s=API, 5000s=admin, 6000s=worker, 8000s=public) AND project (X100=corpus, X110=labrador, X120=mcp, etc.)
2. **Expansion Room**: Each project has 10 ports for growth
3. **Security**: All internal services bind to `127.0.0.1` (zero external exposure without proxy)
4. **Scalability**: Easy to add new projects (just pick next 10-port block)
5. **Consistency**: Every future service follows the same pattern
6. **No collisions**: Reserved ranges prevent accidental port conflicts
7. **Future-proof**: GPU machines, new microservices all fit seamlessly

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
      # Database: 1000 (only accessible via Docker DNS internally)
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
      - AIRFLOW_CONN_SCRAPYD_CONN=http://scrapy:6100  # Docker DNS
    volumes:
      - ./airflow_dags:/opt/airflow/dags
      - ${MNT_CORPUS_RAW}:/data/raw
      - ${MNT_CORPUS_CLEAN}:/data/clean
    ports:
      # Admin UI: 5100 (bind to localhost, access via proxy)
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
      # Scrapyd API: 6100 (internal only)
      - "127.0.0.1:6100:6800"
      # Scrapyd Web UI: 5102 (internal, proxied for monitoring)
      - "127.0.0.1:5102:6800"
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
      - SCRAPYD_URL=http://scrapy:6100  # Docker DNS, uses Scrapyd API port
      - CHAT_SERVICE_URL=http://corpus-chat:3200  # (future GPU machine)
    volumes:
      - ${MNT_CORPUS_RAW}:/tank/corpus_raw:ro
    ports:
      # Search API: 3100 (internal only, proxied via zima)
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
      # Frontend: 3101 (internal only, proxied via zima)
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
      # MCP API: 3120 (internal only, for LLM clients on other machines)
      - "127.0.0.1:3120:3000"
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
SSL: Let's Encrypt

Hostname: api.corpus.marina.cbnano.com
Forward to: http://10.10.1.237:3100  (corpus-api)
  [Optional, for API testing]

Hostname: airflow.marina.cbnano.com
Forward to: http://10.10.1.237:5100  (corpus-airflow)

Hostname: scrapyd.marina.cbnano.com
Forward to: http://10.10.1.237:5102  (corpus-scrapy Web UI)

Hostname: couchdb.marina.cbnano.com
Forward to: http://10.10.1.237:8120  (obsidian-livesync CouchDB)

Hostname: tasks.marina.cbnano.com
Forward to: http://10.10.1.237:8100  (super-productivity)

Hostname: pdf.marina.cbnano.com
Forward to: http://10.10.1.237:8110  (stirling-pdf)

Hostname: lab.marina.cbnano.com
Forward to: http://10.10.1.237:8130  (labrador frontend)
```

---

## Port Reference Quick Guide

### Quick Lookup by Range

```
1000-1999: Databases
  1000: Postgres
  1001: Redis (future)
  1002: Elasticsearch (future)

3000-3999: APIs
  3100-3109: Corpus APIs (api @ 3100, web @ 3101)
  3110-3119: Labrador APIs (api @ 3110, web @ 3111)
  3120-3129: MCP/LLM (mcp @ 3120)
  3200-3209: GPU Machine APIs (corpus-chat @ 3200)
  3130-3199, 3210+: Reserved for future projects

5000-5999: Admin UIs
  5100-5109: Corpus Admin (airflow @ 5100, pgAdmin @ 5101, scrapyd UI @ 5102)
  5110-5119: Labrador Admin
  5120-5129: System Monitoring (prometheus, grafana future)
  5200-5209: GPU Machine Admin (ollama UI @ 5200)

6000-6999: Workers
  6100-6109: Corpus Workers (scrapyd @ 6100, celery future @ 6101)
  6110-6119: Labrador Workers
  6120-6129: System Workers (message queue future @ 6120)

8000-8999: Public Tools (via zima proxy)
  8100-8109: Productivity (super-productivity @ 8100)
  8110-8119: Documents (stirling-pdf @ 8110)
  8120-8129: Data (couchdb @ 8120)
  8130-8139: Lab (labrador @ 8130)

139, 445: Samba (special)
```

---

## Migration Path

### Phase 1: Plan (No downtime)
1. Document current ports for all services
2. Create mapping of old → new ports
3. Generate new docker-compose.yml files for each project

### Phase 2: Test (Offline)
1. Build new images with updated port bindings
2. Test on development machine (if available)
3. Verify Docker DNS resolution: `docker exec <container> getent hosts postgres`

### Phase 3: Deploy (Rolling, low-risk)
Since ae86 hosts multiple docker-compose projects, migrate one at a time:

**Step 1: corpus-postgres project** (most critical)
```bash
cd /tank/docker/compose/corpus-postgres
# Update docker-compose.yml with new port strategy
# Update .env if needed
docker compose down
docker compose up -d
# Test: curl http://localhost:3100/api/papers/categories
# Test: curl http://localhost:5100 (Airflow)
```

**Step 2: labrador project** (secondary)
```bash
cd /tank/docker/compose/labrador
# Update docker-compose.yml with new port strategy
docker compose down
docker compose up -d
# Test: curl http://localhost:3110/api/...
```

**Step 3: Update other projects** (stirling-pdf, super-productivity, obsidian-livesync)
```bash
# Each project, one at a time
# Update ports to new ranges
# Test each service
```

**Step 4: Update zima proxy rules**
```
Update all forward-to ports in Nginx Proxy Manager
Test external access via https://corpus.marina.cbnano.com, etc.
```

### Phase 4: Verify (Full testing)
- [ ] All internal Docker DNS resolution works
- [ ] All services can reach each other (airflow → scrapyd, api → postgres, etc.)
- [ ] External access via zima works (curl https://corpus.marina.cbnano.com)
- [ ] No conflicts or port binding errors
- [ ] Restart ae86 and verify all services come back up

---

## Benefits of This Strategy

1. **Clarity**: Port number immediately tells you: range (1000s=DB, 3000s=API, 5000s=admin, 6000s=worker, 8000s=public) AND project (X100=corpus, X110=labrador, X120=mcp, etc.)
2. **Expansion Room**: Each project has 10 ports for growth within its block
3. **Security**: All internal services bind to `127.0.0.1` (zero external exposure without proxy)
4. **Scalability**: Easy to add new projects (pick next 10-port block)
5. **Consistency**: Every future service follows the same pattern
6. **No collisions**: Reserved ranges prevent accidental port conflicts
7. **Future-proof**: GPU machines, new microservices fit seamlessly
8. **Visual organization**: Grouping by 10s makes it easy to scan and understand at a glance

---

## Example: Adding a New Project (e.g., Web Scraper)

When you add a new web scraper project in the future:

```
API tier (3000-3999):
  Port 3130: scraper-api
  Port 3131: scraper-dashboard
  Port 3132-3139: (reserved)

Admin tier (5000-5999):
  Port 5130: scraper-logs
  Port 5131-5139: (reserved)

Worker tier (6000-6999):
  Port 6130: scraper-queue
  Port 6131-6139: (reserved)

Public tier (8000-8999):
  Port 8140: scraper-results-viewer
  Port 8141-8149: (reserved)
```

Everything follows the same pattern. No conflicts, no confusion.

---

## Summary

| Project | API | Web | Admin | Workers |
|---------|-----|-----|-------|---------|
| **Corpus** | 3100 | 3101 | 5100 | 6100 |
| **Labrador** | 3110 | 3111 | 5110 | 6110 |
| **MCP/LLM** | 3120 | — | 5120 | — |
| **GPU (future)** | 3200 | — | 5200 | — |
| **Future Project 4** | 3130 | 3131 | 5130 | 6130 |
| **Future Project 5** | 3140 | 3141 | 5140 | 6140 |

This is a **complete, extensible port allocation strategy** that scales cleanly to 10+ projects while remaining easy to understand and maintain.
