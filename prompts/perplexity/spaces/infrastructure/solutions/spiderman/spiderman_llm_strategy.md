# Spiderman AI Chat Layer: LLM Research & Multi-Model Strategy

## Executive Summary

For a scientific "chat with papers" system optimized for heat dissipation, materials science, and PUE-focused research, we recommend a **multi-model strategy** rather than a single 70B monolith:

**Phase 1 (Now - MVP, backburner)**: No chat yet. Focus on corpus-web + corpus-api.

**Phase 2 (Future, when GPU hardware ready)**: 
- **Tier 1 (Lightweight, always-on)**: Qwen 2.5 7B Instruct or Mistral 7B
- **Tier 2 (For deep reasoning)**: Qwen 2.5 70B or Llama 3.3 70B
- **Tier 3 (Specialized tasks)**: Domain-specific fine-tuned model on RTX 5090

---

## 1. Open-Source LLM Landscape (2025)

### Top Contenders for Science/Reasoning

| Model | Parameters | License | Strengths | Weaknesses | Context | Hardware Needed |
|-------|-----------|---------|-----------|-----------|---------|-----------------|
| **Llama 3.3 70B Instruct** | 70B | Meta (Llama3 License) | Code generation (80.5% HumanEval), long context (128K), efficient tokenizer, -15% tokens | Limited to 8 languages, custom commercial license | Dec 2023 | 2x A100 / RTX 5090 |
| **Qwen 2.5 72B** | 72B | Apache 2.0 | Math reasoning (83.1% MATH), multilingual (29 langs), structured data, large context (128K) | Slightly verbose output, slower inference than Llama | April 2024 | 2x A100 / RTX 5090 |
| **Mixtral 8x22B (MoE)** | 46.7B active / 141B total | Apache 2.0 | 6x faster inference, excellent math, multilingual (30+ langs), efficient (only 12.9B active) | MoE routing overhead on edge hardware, slightly lower knowledge benchmarks | Jan 2024 | 1x A100 / RTX 5090 with optimization |
| **Mistral 7B v0.3** | 7B | Apache 2.0 | Fast inference, small footprint, great for edge, good reasoning for size | Less sophisticated reasoning than 70B, smaller knowledge base | Jan 2024 | Single RTX 3090 / consumer GPU |
| **Qwen 2.5 7B Instruct** | 7B | Apache 2.0 | Best 7B for structured data, 128K context, good code (79.8% HumanEval) | Limited reasoning depth | April 2024 | RTX 4070 / 5060 Ti |
| **DeepSeek V3** | 671B | Commercial/Research | SOTA reasoning, beats GPT-4o on some benchmarks | Massive resource requirement, not suitable for on-prem | Oct 2024 | Enterprise-grade cluster |
| **Phi 3.5 Mini** | 3.8B | MIT License | Excellent output quality for size, 128K context, unrestricted commercial use | Very small knowledge base | Sept 2024 | Laptop-grade hardware |

### Science Domain Specialists

| Model | Base | Parameters | License | Specialization | Notes |
|-------|------|-----------|---------|-----------------|-------|
| **Me-LLaMA 70B** | Llama 2 | 70B | Proprietary | Medical/Biomedical | Outperforms OSS medical LLMs; not ideal for materials science |
| **Meditron 70B** | Llama 2 | 70B | Research | Medical | Similar to Me-LLaMA; broader medical focus |
| **Qwen 2.5 Math** | Qwen 2.5 | 7B, 32B | Apache 2.0 | Math-heavy reasoning | Custom-trained for STEM tasks; excellent for materials/physics calculations |
| **Qwen 2.5 Coder** | Qwen 2.5 | 7B, 14B, 32B | Apache 2.0 | Code generation + analysis | Strong for interpreting simulation scripts, CFD analysis |

---

## 2. Hardware Timeline & Constraints

### Current Situation
- **ae86**: Dual Xeon (no GPU, for orchestration)
- **pegasus**: RTX 5060 Ti 16GB (now for CAD, soon available for AI)
- **Incoming**: 
  - RTX 5090 machine (Intel Core Ultra 7, 64GB DDR5, 24GB VRAM) — Q1 2026
  - Potentially: 1-2x RTX Pro 6000 96GB (if budget approved)
  - Potentially: Ryzen Strix Halo 365+ 128GB (if budget approved)

### Memory/Inference Requirements

| Model | FP32 (unquantized) | BF16 (half precision) | Int4 Quantized | Inference Speed (H100) | Inference Speed (RTX 5090) |
|-------|-------------------|----------------------|----------------|-----------------------|---------------------------|
| **Llama 3.3 70B** | 280 GB | 140 GB | 18 GB | 100-200 tokens/sec | 30-50 tokens/sec |
| **Qwen 2.5 72B** | 288 GB | 144 GB | 18-20 GB | 80-150 tokens/sec | 25-45 tokens/sec |
| **Mixtral 8x22B** | 176 GB total | 88 GB total | 12 GB | 200+ tokens/sec (6x faster) | 60-100 tokens/sec |
| **Mistral 7B** | 28 GB | 14 GB | 4 GB | 500+ tokens/sec | 150-250 tokens/sec |
| **Qwen 2.5 7B** | 28 GB | 14 GB | 4 GB | 500+ tokens/sec | 150-250 tokens/sec |

**Key insight**: RTX 5090 (24GB VRAM) can handle:
- ✅ 7B models at full precision + room for context
- ✅ 70B models at 4-bit quantization (Int4, like `bitsandbytes` or `GPTQ`)
- ✅ Mixtral 8x22B at 4-bit (actually better than dense 70B due to MoE efficiency)

---

## 3. Recommended Multi-Model Strategy for CB Nano

### Why Multi-Model?

**Single 70B monolith problems:**
- Always uses max compute (slow, expensive, wasteful for simple queries)
- One failure point
- Inefficient for quick lookups (user asks "what is PUE?")
- Harder to fine-tune (you lose general capabilities)

**Multi-model advantages:**
- **Fast tier** (7B): Handles 80% of queries (definitions, quick lookups, paper retrieval)
- **Deep tier** (70B): Complex reasoning, novel analysis, cross-domain synthesis
- **Specialized tier**: Fine-tuned on CB Nano corpus (best for your specific domain)
- **Fallback**: If one model fails, route to another
- **Cost**: 7B inference is 10x cheaper (faster, less power)

---

## 4. Proposed 3-Tier Architecture

### Tier 1: Fast Responder (7B Model) — Always-On

**Model**: **Qwen 2.5 7B Instruct**

**Why Qwen 2.5 7B vs Mistral 7B:**
- Better at structured data handling (perfect for paper metadata queries)
- Stronger mathematical reasoning (materials science, PUE calculations)
- 128K context (can ingest entire papers)
- Multilingual (29 languages, good for international research)
- Apache 2.0 license (unrestricted commercial use)

**Hardware**: RTX 5060 Ti (pegasus, now) or RTX 5090 (dedicated machine, future)

**Inference**: ~150-250 tokens/sec with 4-bit quantization

**Use cases:**
```
User: "What is PUE?"
Tier 1: Instant response (definition + examples)

User: "Find papers on water cooling"
Tier 1: Vector search → Summarize + list papers

User: "Which papers mention immersion cooling?"
Tier 1: Vector search → Return top 5 with abstracts
```

**Deployment**:
```bash
# On pegasus or RTX 5090 machine
ollama pull qwen2.5:7b-instruct-q4_K_M  # 4-bit quantized
# or
python -m llama_cpp.server \
  --model qwen2.5-7b-instruct-q4.gguf \
  --n_gpu_layers 50  # Offload to VRAM \
  --port 5000
```

---

### Tier 2: Deep Reasoner (70B Model) — On-Demand

**Model**: **Mixtral 8x22B (MoE)** OR **Qwen 2.5 70B**

**Mixtral 8x22B vs Qwen 2.5 70B:**

| Aspect | Mixtral 8x22B | Qwen 2.5 70B |
|--------|----------------|-------------|
| **Speed** | 6x faster (only 12.9B active) | ~40-60% slower (full 70B active) |
| **Math Reasoning** | Excellent (tied with Llama 70B) | Slightly better (83.1% MATH vs 82%) |
| **Code Gen** | Strong | Slightly stronger (HumanEval ~80%) |
| **Multilingual** | 30+ languages | 29 languages |
| **Memory @ Int4** | ~12-15 GB | ~18-20 GB |
| **Inference Time** | 50-80 tokens/sec (RTX 5090) | 25-45 tokens/sec (RTX 5090) |
| **Best For** | Fast, complex tasks; efficiency | Maximum knowledge; slower queries |

**Recommendation**: **Start with Mixtral 8x22B** (better speed/quality tradeoff)

**Use cases:**
```
User: "Compare thermal management approaches in datacenters"
Tier 1 response too shallow → Route to Tier 2
Mixtral synthesizes 5-10 papers, generates comparison matrix

User: "Design a novel liquid cooling system for x architecture"
Tier 2 generates design concepts with citations to 3-4 papers
```

**Deployment**:
```bash
ollama pull mixtral:8x22b-instruct-v0.1-q4_K_M  # 4-bit quantized
# or vLLM for batched inference
python -m vllm.entrypoints.openai.api_server \
  --model mixtral-8x22b-instruct-v0.1-q4 \
  --tensor-parallel-size 1 \
  --max-model-len 4096
```

---

### Tier 3: Domain Expert (Fine-tuned, Custom)

**Model Base**: Mixtral 8x22B or Qwen 2.5 7B (fine-tuned)

**Training Data**: CB Nano corpus (when 10k+ papers collected)

**Fine-tuning objective**:
- **Task**: Generate research summaries grounded in CB Nano papers
- **Data**: Generate ~5k SFT pairs from corpus
  - Prompts: "Analyze thermal efficiency of X technology"
  - Responses: Model-generated answer citing papers

**Timeline**: Post-MVP, Q2-Q3 2026 when corpus is mature

**Use cases**:
```
User: "What do we know about graphene thermal conductivity?"
Tier 3 responds: "Based on our papers, X leads to Y improvement..."
  Better domain knowledge than generic model
  Automatically cites CB Nano papers (not external sources)
```

---

## 5. Multi-Model Routing Logic

```python
# In corpus-api (on ae86)

async def route_to_chat_tier(query: str, context: dict) -> str:
    """
    Smart routing: Determine which LLM tier can best handle this query
    """
    
    # Complexity scoring (heuristic)
    complexity = analyze_query_complexity(query)
    
    # Paper retrieval
    papers = await retrieve_papers(query, top_k=20)
    
    # Routing logic
    if complexity < 0.3:  # Simple query ("Define PUE")
        # Use fast Tier 1
        response = await call_tier1_llm(
            query=query,
            papers=papers[:3],  # Just top 3
            model="qwen2.5:7b-instruct",
            timeout=5.0  # Fast timeout
        )
    
    elif complexity < 0.7:  # Moderate query
        # Use Tier 1, but with full paper context
        response = await call_tier1_llm(
            query=query,
            papers=papers[:10],
            model="qwen2.5:7b-instruct",
            timeout=10.0,
            extended_reasoning=True
        )
    
    else:  # Complex query ("Design novel cooling approach")
        # Use Tier 2 (deeper reasoning)
        response = await call_tier2_llm(
            query=query,
            papers=papers[:15],  # More context
            model="mixtral:8x22b-instruct",
            timeout=30.0,
            generate_analysis=True
        )
    
    return response

def analyze_query_complexity(query: str) -> float:
    """
    Heuristic complexity scoring (0-1)
    """
    indicators = {
        "define|explain|what is": 0.1,
        "compare|analyze|design": 0.6,
        "novel|innovative|synthesis": 0.9,
    }
    
    score = 0.3  # default
    for pattern, value in indicators.items():
        if re.search(pattern, query, re.I):
            score = max(score, value)
    
    return min(score, 1.0)
```

---

## 6. Hardware Acquisition Recommendation

### Priority Order (if budget constrained)

1. **Tier A (Must-have for MVP backburner)**: Nothing right now. Focus on corpus-web + corpus-api.

2. **Tier B (Essential for Phase 2 chat launch)**:
   - RTX 5090 machine (Intel Core Ultra 7, 64GB DDR5)
   - Reason: Handles both 7B (always-on) + 70B (on-demand) with 4-bit quantization
   - Cost: ~$2,500 (GPU) + ~$800 (CPU/mobo/RAM)

3. **Tier C (If budget allows)**:
   - **Option A**: 1x RTX Pro 6000 96GB
     - Pro: 96GB VRAM = can run 70B models in FP16 (no quantization needed, higher quality)
     - Con: Professional GPU pricing (expensive), overkill for your use case
   
   - **Option B**: Ryzen Strix Halo 365+ 128GB machine
     - Pro: Cheap, 128GB unified memory (great for CPU inference of large models)
     - Con: CPU-only inference is 10x slower than GPU (not suitable for interactive chat)
     - Verdict: **NOT recommended** for interactive chat; good for batch processing

4. **Tier D (Future, after Phase 2 success)**:
   - 2x RTX 5090 (for serving multiple concurrent users)
   - Or: 1x RTX 6000 Ada + 1x RTX 5090 (hybrid setup)

---

## 7. Implementation Plan for MVP (No Chat Yet)

### This Quarter (Q1 2026)

**Goal**: corpus-web + corpus-api fully operational, zero chat features

```yaml
# docker-compose.yml (ae86)
services:
  postgres: 1000
  corpus-api: 3100
  corpus-web: 3101
  airflow: 5100
  scrapy: 6800
  mcp-server: 3200  # EXISTS but not connected to any LLM yet
```

**Key**: MCP server is ready to accept LLM connections, but no LLM running.

---

## 8. Implementation Plan for Phase 2 (Chat Launch)

### When RTX 5090 is ready (Q2 2026)

**Step 1: Set up inference layer (RTX 5090 machine)**

```dockerfile
# Dockerfile (on RTX 5090 machine)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    cmake build-essential libopenblas-dev

# Install inference frameworks
RUN pip install ollama llama-cpp-python vllm transformers torch

# Copy custom inference server
COPY inference_server.py /app/

EXPOSE 5000

CMD ["python3", "inference_server.py"]
```

**Step 2: Deploy both Tier 1 + Tier 2 models**

```bash
# On RTX 5090
ollama pull qwen2.5:7b-instruct-q4_K_M    # Tier 1: ~4 GB
ollama pull mixtral:8x22b-instruct-q4     # Tier 2: ~13 GB

# Or use vLLM for better concurrency
python -m vllm.entrypoints.openai.api_server \
  --model mixtral-8x22b-instruct-v0.1-q4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --port 5000
```

**Step 3: Update corpus-api to use LLM**

```python
# corpus-api/routes/chat.py (ae86)

@app.post("/api/chat/message")
async def chat_message(query: str, history: list = None):
    """
    Smart routing to Tier 1 or Tier 2 LLM on RTX 5090 machine
    """
    
    # Vector search in pgvector
    papers = await vector_search(query)
    
    # Route based on complexity
    tier = determine_tier(query)
    
    # Call LLM on RTX 5090
    response = await httpx.post(
        f"http://10.10.1.xxx:5000/v1/chat/completions",  # RTX 5090 IP
        json={
            "model": tier.model_name,
            "messages": [
                {"role": "system", "content": f"Context: {papers_to_text(papers)}"},
                *history,
                {"role": "user", "content": query}
            ],
            "stream": True
        }
    )
    
    return response
```

**Step 4: Connect to MCP server**

```python
# corpus-mcp/server.py

from mcp.server import Server
from mcp.server.stdio import StdioServerTransport

# Register tools for LLM to call
server.define_tool("search_papers", search_papers_handler)
server.define_tool("get_paper_details", get_paper_details_handler)
server.define_tool("download_paper", download_paper_handler)

# MCP server runs on ae86 (port 3200)
# LLM on RTX 5090 calls MCP endpoints to interact with corpus
```

---

## 9. Model Selection Summary Table

### For Your Use Case (Scientific Chat, Heat Dissipation Domain)

| Phase | Tier 1 (Fast) | Tier 2 (Deep) | Tier 3 (Expert) |
|-------|---------------|---------------|-----------------|
| **Phase 1** | None (backlog) | None (backlog) | None (backlog) |
| **Phase 2 MVP** | **Qwen 2.5 7B Instruct** | **Mixtral 8x22B** | — |
| **Phase 2 v1.1** | Qwen 2.5 7B | Mixtral 8x22B | Fine-tuned Mixtral 7B |
| **Phase 3 (future)** | Qwen 2.5 7B | Qwen 2.5 70B | CB-Nano-Mixtral-7B-SFT |

**Why this combination?**
- **Qwen 2.5 7B**: Best balance of speed + structured data handling
- **Mixtral 8x22B**: 6x faster than dense 70B, but better reasoning than 7B
- **Fine-tuned variant**: When you have enough training data

---

## 10. Testing & Evaluation Plan

### Phase 2 Launch Benchmarking

```python
# Test suite: corpus_chat_tests.py

test_cases = [
    # Tier 1 (fast, should <2 sec)
    ("What is PUE?", "simple", "tier1", expected_latency=1.0),
    ("Find papers on water cooling", "retrieval", "tier1", expected_latency=2.0),
    
    # Tier 2 (deep, can take 5-30 sec)
    ("Compare immersion vs air cooling", "analysis", "tier2", expected_latency=10.0),
    ("Design a novel hybrid cooling system", "creative", "tier2", expected_latency=20.0),
]

# Run tests and benchmark against baseline (no LLM)
for query, category, expected_tier, latency_budget in test_cases:
    start = time.time()
    response = await chat_message(query)
    elapsed = time.time() - start
    
    assert elapsed < latency_budget, f"Latency {elapsed}s > budget {latency_budget}s"
    assert response["citations_count"] > 0, "No papers cited"
    assert len(response["text"]) > 50, "Response too short"
```

---

## 11. Budget & Timeline

### Rough Estimates

| Item | Cost | Timeline | Notes |
|------|------|----------|-------|
| **RTX 5090** | $2,000-2,500 | Q1 2026 | Priority #1 |
| **Intel Core Ultra 7 + 64GB RAM** | $800-1,200 | Q1 2026 | Complete machine build |
| **RTX Pro 6000 (optional Tier C)** | $7,000-9,000 | Q2 2026 if budget | Professional workstation |
| **Fine-tuning setup (compute + storage)** | ~$1,000 (one-time) | Q2 2026 | Training infrastructure |

**Total for MVP Phase 2**: ~$3,000-5,000 (just RTX 5090 machine)

---

## Conclusion

**For your "chat with scientists" use case:**

1. **Don't wait for 405B or 70B to be necessary**. Start with smart routing (7B + 70B).
2. **Mixtral 8x22B is the sweet spot** for your domain (math + reasoning + speed).
3. **Fine-tuning on your corpus** will be more valuable than any SOTA closed-source model.
4. **RTX 5090 is the right GPU** (single card can handle Tier 1 + Tier 2 with quantization).
5. **Multi-model approach is more resilient** (fallback if one model fails).

**Action**: Build MVP (corpus-web + corpus-api) now. In Q2 2026 when RTX 5090 arrives, flip on Phase 2 chat with confidence that you've chosen the right models.
