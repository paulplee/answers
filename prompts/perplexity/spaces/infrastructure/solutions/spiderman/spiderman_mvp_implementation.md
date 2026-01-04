# Spiderman MVP Implementation Plan (Q1 2026)

## Refocused Scope: Web Portal + Search API Only

Given pegasus is being repurposed for CAD and GPU/LLM work is on the backburner, here's the tightly focused MVP:

---

## 1. What We're Building (MVP)

### ✅ Included

- **corpus-web**: Static frontend portal
  - Grid/list view of papers
  - Search bar with autocomplete
  - Sidebar filters (category, year, relevance, PDF status)
  - Paper detail cards with download buttons
  - Zero external JS dependencies (vanilla HTML/CSS/JS)

- **corpus-api**: FastAPI backend
  - `/api/papers/search` — Full-text + vector search
  - `/api/papers/{id}` — Paper details
  - `/api/papers/{id}/download` — Trigger Scrapyd PDF download
  - `/api/download/{job_id}/status` — Poll download progress
  - `/api/papers/{id}/file` — Serve downloaded PDF
  - `/api/papers/categories` — List available categories
  - Error handling, rate limiting, input validation

- **Docker deployment**: Both as containers on ae86
  - Use new port strategy (3100 for API, 3101 for web)
  - Reverse proxy via zima for external access
  - All bind to `127.0.0.1` (internal only)

- **Corpus metadata UI**: Corpus statistics footer
  - Total papers, PDFs downloaded
  - Last ingestion time
  - Average relevance score

### ❌ NOT Included (Future/Backlog)

- ❌ Chat UI (no LLM yet)
- ❌ User accounts/auth (simple IP-based optional logging only)
- ❌ Paper upload/annotation
- ❌ Advanced analytics
- ❌ Multi-language support (English only for MVP)
- ❌ PDF inline viewer (just download buttons)
- ❌ Real-time notifications

---

## 2. Architecture (MVP Version)

```
┌─────────────────────────────────────┐
│  zima (Nginx Proxy Manager)         │
│  corpus.marina.cbnano.com (HTTPS)   │
└────────────────────┬────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
  ┌──────────┐            ┌──────────┐
  │corpus-web│ (3101)     │corpus-api│ (3100)
  │HTML/CSS/JS           │FastAPI
  │No deps               │ 
  │Read-only             │Postgres queries
  └──────────┘           │Scrapyd triggers
                         │File serving
                         └────────┬─────────┘
                                  │
                        ┌─────────┴──────────┐
                        │                    │
                        ▼                    ▼
                   ┌──────────┐         ┌──────────┐
                   │corpus-   │         │corpus-   │
                   │postgres  │         │scrapy    │
                   │(1000)    │         │(6800)    │
                   └──────────┘         └──────────┘
```

**Key**: Everything on ae86, nothing on pegasus for MVP.

---

## 3. Detailed File Structure

```
/tank/docker/compose/corpus-web-mvp/
├── docker-compose.yml          # Defines both web + api containers
├── .env                        # DB credentials, paths
├── .env.example
├── README.md
│
├── api/                        # FastAPI backend
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pyproject.toml         # (optional, for uv)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI app, routes
│   │   ├── database.py        # asyncpg pool
│   │   ├── models.py          # Data models (Pydantic)
│   │   ├── routes/
│   │   │   ├── papers.py      # /api/papers/* endpoints
│   │   │   └── downloads.py   # /api/download/* endpoints
│   │   └── utils.py           # Helpers (search, scrapyd, etc.)
│
├── web/                       # Static frontend
│   ├── Dockerfile
│   ├── public/
│   │   ├── index.html         # Single-page app entry
│   │   ├── css/
│   │   │   └── style.css      # Design system colors
│   │   ├── js/
│   │   │   ├── main.js        # App initialization
│   │   │   ├── api.js         # fetch() wrappers
│   │   │   ├── ui.js          # DOM rendering
│   │   │   └── state.js       # Client state management
│   │   └── images/
│   │       ├── logo.svg
│   │       └── icons/
│
└── docs/
    ├── API.md                 # Endpoint documentation
    ├── SETUP.md               # Deployment steps
    └── ARCHITECTURE.md        # High-level overview
```

---

## 4. Core API Endpoints

### Search & Browse

```
GET /api/papers/search
  Query params:
    - q: string                 # Full-text search
    - category: string[]        # Filter by categories
    - year_from: int            # Year range
    - year_to: int
    - relevance_min: float      # 0.0-1.0
    - sort_by: "date"|"relevance"|"title"
    - page: int                 # Pagination
    - limit: int                # 50 default
  
  Response:
  {
    "papers": [
      {
        "arxiv_id": "2401.12345",
        "title": "...",
        "authors": ["Name1", "Name2"],
        "abstract": "...",
        "published_date": "2024-01-15",
        "relevance_score": 0.87,
        "paper_status": "pdf_downloaded"|"metadata_only",
        "categories": ["cond-mat.mtrl-sci", "cs.AI"]
      }
    ],
    "total": 1234,
    "page": 1,
    "pages": 25
  }
```

### Paper Details

```
GET /api/papers/{arxiv_id}
  Response:
  {
    "arxiv_id": "2401.12345",
    "title": "...",
    "authors": [...],
    "abstract": "...",
    "published_date": "2024-01-15",
    "updated_date": "2024-02-01",
    "categories": [...],
    "paper_status": "pdf_downloaded",
    "pdf_path": "/tank/corpus_raw/pdfs/2401.12345.pdf",
    "relevance_score": 0.87
  }
```

### Download Management

```
POST /api/papers/{arxiv_id}/download
  Response:
  {
    "job_id": "abc123xyz",
    "status": "pending",
    "eta_seconds": 120
  }

GET /api/download/{job_id}/status
  Response:
  {
    "status": "downloading|complete|failed",
    "progress": 45,  # 0-100
    "error": null
  }

GET /api/papers/{arxiv_id}/file
  Response: Binary PDF file (Content-Type: application/pdf)
```

### Metadata

```
GET /api/papers/categories
  Response:
  {
    "categories": [
      "cs.AI",
      "cond-mat.mtrl-sci",
      "cs.MA",
      ...
    ]
  }

GET /api/stats
  Response:
  {
    "total_papers": 15234,
    "papers_with_pdf": 8932,
    "last_ingestion": "2026-01-04T12:34:56Z",
    "average_relevance": 0.72,
    "categories_count": 45
  }
```

---

## 5. Frontend Architecture (Vanilla JS)

### Key Principles

- **No build tool**: Everything runs in browser as-is
- **No framework**: React, Vue, etc. are overkill for this UI
- **No external dependencies**: Only native Web APIs
- **Modular JS**: Separate concerns (api, ui, state)
- **CSS variables**: Design system colors, easy to theme

### Main.js Flow

```javascript
// Lifecycle:
1. Load app state (filters, last query) from localStorage
2. Fetch categories from API
3. Render sidebar filters
4. Load initial papers (most recent)
5. Attach event listeners
6. Debounce search input
7. Lazy-load paper cards as user scrolls
```

### Example Components

```javascript
// api.js
export async function fetchPapers(filters, page = 1) {
  const params = new URLSearchParams({
    q: filters.search,
    category: filters.categories.join(','),
    year_from: filters.yearFrom,
    year_to: filters.yearTo,
    relevance_min: filters.relevanceMin,
    sort_by: filters.sortBy,
    page, limit: 50
  });
  
  const response = await fetch(`/api/papers/search?${params}`);
  if (!response.ok) throw new Error(`API error: ${response.status}`);
  return response.json();
}

// ui.js
export function renderPaperCard(paper) {
  const card = document.createElement('article');
  card.className = 'paper-card';
  
  const statusLabel = paper.paper_status === 'pdf_downloaded' 
    ? '📄 PDF' 
    : '⬇️ Request';
  
  card.innerHTML = `
    <h3>${escapeHtml(paper.title)}</h3>
    <p class="authors">${escapeHtml(paper.authors.join(', ').substring(0, 100))}...</p>
    <p class="date">${new Date(paper.published_date).toLocaleDateString()}</p>
    <p class="abstract">${escapeHtml(paper.abstract.substring(0, 300))}...</p>
    <div class="meta">
      <span class="relevance">Relevance: ${(paper.relevance_score * 100).toFixed(0)}%</span>
      <span class="status ${paper.paper_status}">${statusLabel}</span>
    </div>
    <div class="actions">
      ${paper.paper_status === 'pdf_downloaded'
        ? `<a href="/api/papers/${paper.arxiv_id}/file" class="btn btn-primary" download>Download PDF</a>`
        : `<button class="btn btn-secondary" data-arxiv="${paper.arxiv_id}">Request PDF</button>`
      }
      <button class="btn btn-outline" data-arxiv="${paper.arxiv_id}">More Info</button>
    </div>
  `;
  
  return card;
}

// state.js
const state = {
  filters: {
    search: '',
    categories: [],
    yearFrom: 2020,
    yearTo: new Date().getFullYear(),
    relevanceMin: 0.5,
    sortBy: 'date'
  },
  papers: [],
  currentPage: 1,
  totalPages: 1,
  downloadingJobs: {},
  allCategories: []
};
```

---

## 6. Docker Setup

### docker-compose.yml

```yaml
version: '3.8'

services:
  # Frontend
  web:
    build:
      context: ./web
      dockerfile: Dockerfile
    container_name: corpus-web
    restart: unless-stopped
    ports:
      - "127.0.0.1:3101:80"
    volumes:
      - ./web/public:/app/public:ro
    depends_on:
      - api
    networks:
      - corpus-net

  # API
  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    container_name: corpus-api
    restart: unless-stopped
    env_file: .env
    environment:
      - DATABASE_URL=postgresql://corpus_admin:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - SCRAPYD_URL=http://scrapy:6800
    depends_on:
      postgres:
        condition: service_healthy
    ports:
      - "127.0.0.1:3100:8000"
    volumes:
      - ${MNT_CORPUS_RAW}:/tank/corpus_raw:ro
    networks:
      - corpus-net

networks:
  corpus-net:
    driver: bridge
    name: corpus-network
```

### Dockerfile (API)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Using uv for fast dependency resolution
RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv pip install --system -r pyproject.toml

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Dockerfile (Web)

```dockerfile
FROM caddy:latest

COPY web/public /srv

EXPOSE 80
```

---

## 7. Reverse Proxy (zima/NPM)

Add to Nginx Proxy Manager:

```
Hostname: corpus.marina.cbnano.com
Forward to: http://10.10.1.237:3101
SSL: Let's Encrypt
```

Optional (for API testing):
```
Hostname: api.corpus.marina.cbnano.com
Forward to: http://10.10.1.237:3100
SSL: Let's Encrypt
```

---

## 8. Development Workflow

### On Your Machine

```bash
# Clone repo
git clone <repo> corpus-web-mvp
cd corpus-web-mvp

# Test API locally (requires Postgres access)
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Set DATABASE_URL env var pointing to ae86 Postgres
uvicorn app.main:app --reload

# Test frontend locally
cd ../web
# Open public/index.html in browser (CORS will fail locally, expected)
# Or run simple HTTP server:
python -m http.server --directory public 8000
```

### Deploy to ae86

```bash
# SSH to ae86
ssh paulplee@ae86.marina

# Go to docker compose dir
cd /tank/docker/compose

# Create new directory
mkdir corpus-web-mvp
cd corpus-web-mvp

# Copy files
git clone <repo> .

# Build & launch
docker compose up -d

# Verify
docker compose logs -f api
docker compose ps
curl http://localhost:3100/api/papers/categories
```

---

## 9. Testing Checklist

### API Tests

```python
# api/tests/test_search.py

async def test_search_papers():
    """Search with multiple filters"""
    response = await client.get("/api/papers/search?q=graphene&year_from=2023")
    assert response.status_code == 200
    assert len(response.json()["papers"]) > 0

async def test_paper_details():
    """Retrieve full paper details"""
    arxiv_id = "2401.12345"
    response = await client.get(f"/api/papers/{arxiv_id}")
    assert response.status_code == 200
    assert response.json()["arxiv_id"] == arxiv_id

async def test_download_trigger():
    """Trigger PDF download via Scrapyd"""
    arxiv_id = "2401.12345"
    response = await client.post(f"/api/papers/{arxiv_id}/download")
    assert response.status_code == 200
    assert "job_id" in response.json()
```

### Frontend Tests

```javascript
// web/tests/ui.test.js

describe("Paper Card Rendering", () => {
  test("renders paper card with all fields", () => {
    const paper = {
      arxiv_id: "2401.12345",
      title: "Test Paper",
      authors: ["Author 1", "Author 2"],
      abstract: "Abstract text",
      published_date: "2024-01-15",
      relevance_score: 0.87,
      paper_status: "pdf_downloaded"
    };
    
    const card = renderPaperCard(paper);
    expect(card.querySelector('h3').textContent).toContain("Test Paper");
    expect(card.querySelector('a').href).toContain("2401.12345/file");
  });

  test("shows Request PDF button for metadata-only papers", () => {
    const paper = { ...mockPaper, paper_status: "metadata_only" };
    const card = renderPaperCard(paper);
    expect(card.querySelector('button').textContent).toContain("Request PDF");
  });
});
```

---

## 10. Deliverables (MVP)

By end of Q1 2026:

- ✅ `corpus-web` Docker container (production-ready)
- ✅ `corpus-api` Docker container (production-ready)
- ✅ Updated `docker-compose.yml` for corpus-postgres project
- ✅ Port strategy documentation
- ✅ Reverse proxy configuration (zima)
- ✅ README with setup instructions
- ✅ API documentation (OpenAPI/Swagger auto-generated)
- ✅ Frontend code (vanilla JS, zero external deps)
- ✅ Test suite (API + frontend)
- ✅ Deployment checklist

---

## 11. Future Work (Post-MVP)

Once MVP is stable and being used:

1. **Add basic auth** (API keys for rate limiting)
2. **User preferences** (save filter states to localStorage)
3. **Advanced search** (boolean operators, proximity search)
4. **Metrics dashboard** (corpus growth, trending topics)
5. **Chat feature** (when GPU hardware is ready)
6. **Export features** (BibTeX, CSV)
7. **Paper annotations** (user notes, highlights)

But these are all **FUTURE**. Focus on MVP shipping first.

---

## 12. Success Criteria

### Technical

- ✅ All API endpoints respond <500ms (cached queries)
- ✅ Frontend loads in <3 seconds (no external JS deps)
- ✅ Search returns results within 2 seconds
- ✅ PDF download queued within 1 second
- ✅ Zero JavaScript console errors in browser
- ✅ Responsive design (mobile + desktop)
- ✅ WCAG 2.1 AA accessibility compliant
- ✅ 100% Lighthouse performance score

### User Experience

- ✅ Non-technical users can use portal without training
- ✅ Filters intuitive and discoverable
- ✅ Download workflow clear (request → progress → link)
- ✅ Error messages helpful (not technical jargon)
- ✅ No 404s or broken links

---

## Conclusion

**This MVP is lean, focused, and ships fast.** No chat, no complex features, just a clean interface for browsing and downloading papers from your corpus. Once this is live and stable, you'll have a solid foundation for adding the chat layer when GPUs are ready.
