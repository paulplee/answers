# Comprehensive R&D AI Infrastructure Plan for CB Nano
## Strategic Framework for Materials Science & Chemical Engineering AI Solutions

**Document Version:** 1.0  
**Date:** December 29, 2025  
**Prepared For:** CB Nano Materials R&D Team  

***

## Executive Summary

This plan outlines a phased, cost-efficient approach to building a competitive AI infrastructure for materials science R&D. Rather than attempting all five components simultaneously, the strategy prioritizes **immediate value generation** with a minimum viable system, then scales intelligently based on team learnings and ROI.

**Key Principle:** Start with RAG + strategic inference infrastructure, fine-tune domain-specific models iteratively, and build internal knowledge systems as you learn what works.

***

## Part 1: Current Infrastructure Assessment

### Existing Hardware Inventory

| Machine | CPU | GPU | RAM | Storage | Primary Use |
|---------|-----|-----|-----|---------|------------|
| **atom** (Mac) | M4 Pro (14c) | M4 Pro GPU | 64GB | 1TB SSD | Workstation, Ollama inference |
| **pegasus** | i7-14700K | RTX 5060 Ti (16GB VRAM) | Variable | 2TB NVMe | Training/inference primary |
| **ae86** | Xeon X5650 | None | 128GB | 14TB ZFS | Data storage, backend services |
| **zima** | Celeron N3450 | Intel iGPU | 4GB | 30GB eMMC | Small inference, Docker |
| **lab** (RPi4) | ARM Cortex-A72 | None | 8GB | Small | IoT/edge experiments |

### Current Capability Analysis

✅ **Strengths:**
- Distributed infrastructure across multiple systems
- Docker containerization already in place (Zima, Pegasus)
- High-capacity NAS with ZFS (ae86: 14TB usable)
- Diverse compute profiles (Mac, Linux, ARM)
- Existing network infrastructure at marina.cbnano.com

⚠️ **Gaps:**
- GPU capacity insufficient for production-scale training (RTX 5060 Ti ~ 7-8 TFLOPS per process)
- Limited unified ML framework deployment
- No centralized vector database for RAG
- No specialized inference optimization
- Python/ML stack scattered across systems

***

## Part 2: Recommended Hardware Investment Strategy

### Phase 1 (Immediate): Core Inference + Fine-Tuning Node
**Budget: ~$25,000-35,000 | Timeline: 1-2 months**

**Primary Recommendation: Dell PowerEdge XE8640 or Supermicro GPU System with RTX 6000 Ada 48GB**

**Why This Over Mac Studio M3 Ultra?**
- NVIDIA RTX 6000 Ada (48GB HBM2e) offers:
  - 2.5x better performance/$ than M3 Ultra
  - Superior compatibility with PyTorch ecosystem
  - Established Linux/containerization tooling
  - Better multi-GPU scaling path
  - Lower vendor lock-in for future upgrades
  
**Alternative: Dual RTX 5090 System**
- More parallelism for concurrent fine-tuning tasks
- Better throughput for batch inference
- ~$32k for both cards + system

**Specification:**
```
- CPU: AMD Ryzen 9 7950X3D or Intel Xeon W9-3495X (32 cores, $3-4k)
- GPU: RTX 6000 Ada 48GB ($7,500 retail)
- RAM: 256GB DDR5 ECC ($6-8k)
- Storage: 4TB NVMe + network attachment to ae86 ZFS
- PSU: 2000W+ modular
- Network: 10GbE connection to marina.cbnano.com
```

**Total System Cost: ~$28-32k**

### Phase 2 (Months 3-6): Secondary Inference Optimization
**Budget: ~$8,000-12,000**

Upgrade **pegasus** or create new inference node:
- Keep RTX 5060 Ti for development
- Add: 64GB+ RAM, faster NVMe
- Purpose: Parallel inference, smaller model serving, continuous fine-tuning checkpoints

### Phase 3 (Months 6-12): Storage & Orchestration
**Budget: ~$10,000-15,000**

- Expand ae86 ZFS capacity to 30TB+ (model weights, training data, checkpoints)
- Deploy Kubernetes cluster across Pegasus + new node + ae86
- Establish centralized logging/monitoring

***

## Part 3: Software Architecture

### Layered AI Stack (Recommended)

```
┌─────────────────────────────────────────────────┐
│         Scientist Interface Layer                │
│    (Web UI, CLI, Jupyter Notebooks)             │
│     Streamlit/FastAPI Frontend                   │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         Application Logic Layer                  │
│  Orchestration + Chain-of-Thought Agent Logic   │
│  (LangChain/LlamaIndex + Function Calling)      │
└─────────────────────────────────────────────────┘
                      ↓
┌──────────────────┬──────────────────┬───────────┐
│   RAG Engine     │  LLM Interface   │  Tools    │
│  (Vector DB +    │  (External APIs  │ Integration
│   Retrieval)     │   + Local LLMs)  │           │
└──────────────────┴──────────────────┴───────────┘
                      ↓
┌──────────────────┬──────────────────┬───────────┐
│ Domain-Specific  │  Base LLM Server │  Science  │
│ Fine-Tuned Models│  (DeepSeek 70B   │  Tools    │
│ (Chemistry, etc) │   or Llama 405B) │ (RDKit,   │
│                  │  (vLLM/TensorRT) │  ORCA)    │
└──────────────────┴──────────────────┴───────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│    Infrastructure Layer (Kubernetes)            │
│  GPU Orchestration + Monitoring + Networking    │
└─────────────────────────────────────────────────┘
```

### Component Details

#### 1. **Inference Server** (Priority: IMMEDIATE)
**Selected: vLLM on RTX 6000 Ada**

```bash
# vLLM deployment for local inference
pip install vllm

# Example: DeepSeek-V3.2 70B serving
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-V3.2-70B-Instruct \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

**Why vLLM:**
- Native support for continuous batching → 5-10x throughput vs naive inference
- Token streaming for better UX
- Efficient KV cache management
- Built for RTX 6000 performance optimization

#### 2. **Vector Database for RAG** (Priority: MONTHS 1-2)
**Selected: Milvus (self-hosted) or Weaviate**

**Milvus Setup:**
```yaml
# docker-compose on zima for lightweight RAG
version: '3'
services:
  milvus:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
      - "9091:9091"
    environment:
      COMMON_STORAGETYPE: local
    volumes:
      - /data/milvus:/var/lib/milvus  # NFS mount to ae86
```

**Use Case:** Index material science papers, internal R&D notes, chemical literature
- Start: ~500k documents (papers + internal docs)
- Embedding model: All-MiniLM-L6-v2 (lightweight, good domain coverage)
- Query latency target: <100ms for 10k nearest neighbors

#### 3. **Fine-Tuning Pipeline** (Priority: MONTHS 2-3)
**Framework: Hugging Face + PEFT + Axolotl**

**Domain-Specific Fine-Tuning Roadmap:**

| Discipline | Base Model | Data Source | Timeline |
|-----------|-----------|-------------|----------|
| **Organic Chemistry** | DeepSeek-V3.2-32B or Llama 405B | USPTO, PubChem, ChemRxiv | Month 2-3 |
| **Chemical Engineering** | Llama-70B | Engineering handbooks, process flows | Month 3-4 |
| **Computational Materials** | DeepSeek-70B | ICSD, Materials Project data | Month 4-5 |
| **Thermal Dynamics** | Mistral-32B or Qwen-32B | ThermoData, NIST databases | Month 5-6 |

**Efficient Fine-Tuning Strategy (QLoRA):**

```python
# Using PEFT for parameter-efficient fine-tuning
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit quantization for RTX 6000 efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V3.2-70B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA configuration: only ~5-10M trainable params
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

**Expected Outcomes:**
- Reduce training memory from 80GB → 20-25GB
- Reduce training time from weeks → 2-4 days per domain on RTX 6000
- Fine-tuned model weights: ~200-400MB per LoRA adapter

#### 4. **Tool Integration Layer** (Priority: MONTHS 2-6)
**Pattern: Function Calling API**

```python
# LangChain agent with tool access
from langchain.agents import AgentExecutor, Tool
from langchain.chat_models import ChatOpenAI  # or local vLLM
import rdkit
from rdkit import Chem

# Define domain-specific tools
tools = [
    Tool(
        name="molecular_property_predictor",
        func=predict_molecular_properties,
        description="Predicts HOMO-LUMO gap, solubility, toxicity given SMILES"
    ),
    Tool(
        name="synthesis_route_planner",
        func=plan_synthesis,
        description="Generates synthetic routes to target molecule"
    ),
    Tool(
        name="literature_search",
        func=search_literature_rag,
        description="Searches internal knowledge base + arXiv/PubChem"
    ),
    Tool(
        name="simulation_runner",
        func=queue_orca_simulation,
        description="Submits ORCA DFT calculations to compute cluster"
    )
]

# Execute multi-step chemistry reasoning
agent = AgentExecutor.from_agent_and_tools(
    agent=OpenAIFunctionsAgent.from_llm_and_tools(llm, tools),
    tools=tools,
    verbose=True
)

result = agent.run("Design a new organic photocatalyst with high stability")
```

#### 5. **Data Pipelines & Web Crawling** (Priority: MONTHS 3-6)
**Recommended Tools:**

```python
# Web crawling and document ingestion
import scrapy
from unstructured.partition.pdf import partition_pdf
from langchain.document_loaders import PyPDFLoader, ArxivLoader

# Periodic crawling of relevant sources
class ChemLiteratureCrawler(scrapy.Spider):
    """Crawl materials science papers and databases"""
    
    def start_requests(self):
        urls = [
            'https://arxiv.org/list/cond-mat.mtrl-sci/recent',
            'https://www.nature.com/nmat/articles/',
            'https://pubs.acs.org/journal/macmdc'  # example: materials chem
        ]
        for url in urls:
            yield scrapy.Request(url, callback=self.parse)

# Batch processing pipeline
from airflow import DAG
from airflow.operators.python import PythonOperator

# Weekly ETL: crawl → parse → embed → index
with DAG('chemistry_literature_etl') as dag:
    crawl_task = PythonOperator(
        task_id='crawl_literature',
        python_callable=run_web_crawlers
    )
    parse_task = PythonOperator(
        task_id='parse_pdfs',
        python_callable=extract_text_and_metadata
    )
    embed_task = PythonOperator(
        task_id='generate_embeddings',
        python_callable=batch_embed_documents
    )
    index_task = PythonOperator(
        task_id='update_milvus',
        python_callable=update_vector_db
    )
    
    crawl_task >> parse_task >> embed_task >> index_task
```

#### 6. **Internal Knowledge Database** (Priority: MONTHS 1-6, Ongoing)
**Platform: Structured MongoDB + vector embeddings**

```javascript
// MongoDB schema for internal R&D knowledge
{
  _id: ObjectId(),
  type: "experiment_result",  // or "protocol", "material", "reaction"
  title: "Synthesis of ZnO nanoparticles via sol-gel",
  authors: ["Dr. Chen", "Lab Tech A"],
  date_conducted: ISODate("2025-01-15"),
  content: "...",  // full text
  metadata: {
    temperature_range: [150, 250],  // Celsius
    yield: 82.5,  // %
    material_produced: "ZnO",
    crystal_structure: "wurtzite",
    particle_size: { min: 10, max: 50, unit: "nm" }
  },
  embedding: [0.123, 0.456, ...],  // vector for semantic search
  tags: ["nanomaterials", "oxides", "synthesis"],
  success_rating: 4,  // 1-5 scale
  conditions: {
    precursor: "zinc acetate",
    solvent: "ethanol",
    pH: 7.2,
    stirring_rpm: 500
  },
  related_publications: ["doi:10.1021/xxx", "..."],
  created_at: ISODate("2025-01-20")
}
```

***

## Part 4: Phased Implementation Roadmap

### Month 1-2: Foundation (Inference + RAG)

**Hardware:**
- ✅ Order RTX 6000 Ada system
- ✅ Set up 10GbE network to ae86

**Software:**
- [ ] Deploy vLLM on Pegasus (initial) / new system (eventual)
- [ ] Set up Milvus on Zima for vector indexing
- [ ] Implement document ingestion pipeline (papers → embeddings)
- [ ] Build basic FastAPI interface for RAG queries

**Deliverable:** Scientists can query chemistry literature + internal docs via web UI

**Success Metrics:**
- RAG query latency < 150ms
- Embedding accuracy on internal test set > 85%
- 500k documents indexed

***

### Month 3-4: Fine-Tuning + Domain Specialization

**Hardware:**
- ✅ RTX 6000 Ada operational, integrated with cluster

**Software:**
- [ ] Create fine-tuning pipeline (Axolotl + PEFT)
- [ ] Curate chemistry training data (50k-100k examples from USPTO, PubChem, internal)
- [ ] Fine-tune DeepSeek-V3.2-32B on organic chemistry (2-3 days)
- [ ] Build chemistry agent with RDKit + SMILES parsing
- [ ] Evaluate against domain benchmarks

**Deliverable:** Domain-specific LLM for organic chemistry R&D assistance

**Success Metrics:**
- Fine-tuned model accuracy on chemistry tasks +15-20% vs base model
- Synthesis planning success rate > 75%
- Training time < 48 hours for 50k examples

***

### Month 5-6: Tool Integration + Automation

**Software:**
- [ ] Integrate ORCA/Gaussian runners (DFT calculations)
- [ ] Implement molecular property prediction tools
- [ ] Build chemistry workflow orchestrator
- [ ] Deploy second domain-specific model (chemical engineering)
- [ ] Set up continuous monitoring/logging

**Deliverable:** Multi-agent AI system for complex chemistry R&D workflows

**Success Metrics:**
- Agent success rate on multi-step tasks > 70%
- Average task completion time < 1 hour
- No GPU utilization bottleneck at concurrent usage

***

### Month 6-12: Scaling & Optimization

**Hardware:**
- Optional: Secondary GPU or upgrade existing
- Expand storage to 30TB+ as models grow

**Software:**
- [ ] Fine-tune remaining domain models (materials, thermal)
- [ ] Deploy Kubernetes for autoscaling
- [ ] Implement A/B testing framework for model improvements
- [ ] Build scientist feedback loop into training pipeline
- [ ] Create cost tracking + ROI analysis

**Deliverable:** Production-grade AI research platform

***

## Part 5: Technology Stack Summary

### Core Infrastructure
| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **GPU Compute** | NVIDIA RTX 6000 Ada (48GB) | Best price/performance, enterprise support |
| **Inference** | vLLM + TensorRT-LLM | 5-10x faster than default, continuous batching |
| **Fine-Tuning** | Hugging Face + PEFT (QLoRA) | Memory efficient, wide community support |
| **Vector DB** | Milvus (self-hosted) | Open-source, low latency, good scaling |
| **Orchestration** | LangChain + FastAPI | Flexible, well-documented, active community |
| **Data Pipeline** | Apache Airflow | Scheduling, monitoring, reproducibility |
| **Storage** | ZFS (ae86) + NVMe (compute) | Redundancy, snapshots, fast access |
| **Monitoring** | Prometheus + Grafana | Lightweight, self-hosted |

### Model Selection

**Base Models (Inference/Fine-Tuning):**
1. **DeepSeek-V3.2-32B** (Start here for fine-tuning)
   - Excellent reasoning, multilingual
   - ~16GB VRAM quantized → fits RTX 5060 Ti
   - MIT license (commercial use OK)

2. **Llama 3.2-70B** (For heavier tasks)
   - Strong chemistry benchmarks
   - Well-supported tooling
   - 35GB VRAM quantized on RTX 6000

3. **Mistral-32B** (Fallback for efficiency)
   - Fastest inference
   - Good for lightweight agents

### Embedding Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **All-MiniLM-L6-v2** | 22MB | ~2ms | High | General RAG, internal docs |
| **BioLORD** | 500MB | ~15ms | Very High | Specialized chemistry/materials |
| **ChemBERTa** | 400MB | ~12ms | High | Chemical text, SMILES |

***

## Part 6: Cost Analysis

### Hardware Investment Summary

```
Year 1 Budget:

Phase 1 (Immediate):
  - RTX 6000 Ada GPU: $7,500
  - Server system (CPU/RAM/Storage): $6,500-8,000
  - Networking (10GbE, cables): $2,000
  - Subtotal: ~$28,000-32,000

Phase 2 (Months 3-6):
  - Upgrade Pegasus RAM to 64GB: $2,000
  - Additional NVMe storage: $1,500
  - Subtotal: ~$3,500

Phase 3 (Months 6-12):
  - ZFS expansion modules: $4,000-6,000
  - Kubernetes infrastructure: $2,000-3,000
  - Subtotal: ~$6,000-9,000

Total Year 1 Hardware: ~$37,500-44,000

Ongoing (Year 2+):
  - Power + cooling: ~$2,000/year
  - Network maintenance: ~$500/year
  - Storage expansion: ~$3,000/year (as needed)
```

### Cost Comparison vs. Alternatives

| Approach | Initial Cost | Monthly OpEx | Pros | Cons |
|----------|-------------|--------------|------|------|
| **Self-Hosted (Recommended)** | $40k | $300-500 | Full control, no lockup, best ROI | Maintenance burden |
| **AWS SageMaker** | $0 | $5-10k | Managed, scale instantly | Expensive, latency, data egress costs |
| **Hugging Face Inference** | $0 | $500-1500 | Easy, managed | Limited customization, rate limits |
| **Modal/Fireworks** | $0 | $1-3k | Serverless, pay-per-use | No fine-tuning, pricing at scale |
| **Mac Studio M3 Ultra** | $8k | $300 | Integrated, low power | GPU limited, lesser ecosystem |

**Recommendation:** Self-hosted approach has best 3-year ROI (~$0.15/inference after breakeven)

***

## Part 7: Implementation Best Practices

### 1. Data Strategy
- **Start Small:** Begin with 10-20k curated chemistry examples
- **Quality > Quantity:** One expert-validated example > 10 noisy ones
- **Active Learning:** Use model uncertainty to prioritize labeling high-value examples
- **Version Everything:** DVC (Data Version Control) for reproducibility

### 2. Experiment Tracking
```python
# Use Weights & Biases or MLflow for tracking
import wandb

wandb.init(project="chemistry-llm-fine-tuning")
wandb.log({
    "epoch": epoch,
    "loss": train_loss,
    "chemistry_benchmark_accuracy": acc,
    "gpu_memory_gb": gpu_mem
})
```

### 3. Safety & Governance
- **Prompt Guardrails:** Implement safety classifier (e.g., Perspective API) to catch:
  - Requests for dangerous synthesis procedures
  - Toxic chemical handling without proper context
  - Misuse scenarios
  
- **Audit Trail:** Log all model outputs for regulatory compliance
- **Access Control:** Role-based access (read-only vs. execute simulations)

### 4. Scientist Feedback Loop
```
Suggestion → Manual Validation → Feedback → Retraining
(weekly check-in with chemistry team on model suggestions)
```

***

## Part 8: Success Metrics & KPIs

### Technical KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Inference latency (p95)** | < 2 sec for 1k token context | vLLM metrics |
| **GPU utilization** | 70-85% during peak hours | nvidia-smi logs |
| **RAG accuracy** | > 85% on retrieval tests | Internal evaluation set |
| **Fine-tuned model accuracy** | +15-25% vs base on domain tasks | Benchmark dataset |
| **System uptime** | 99.5% | Monitoring alerts |

### Business KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Time-to-insight** | 50% reduction | Scientist survey |
| **Experiment ideation throughput** | 2-3x increase | Project tracking system |
| **Failed experiments prevented** | > 20% | Internal validation |
| **Cost per experiment** | 30-40% reduction | Finance tracking |
| **Scientist adoption rate** | 80%+ by month 6 | Usage logs |

***

## Part 9: Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **GPU shortage/supply delays** | Medium | High | Order now, plan months 2-3 as buffer |
| **Fine-tuning data quality** | High | Medium | Start with small curated set, iterate |
| **Model hallucination on chemistry** | High | High | Implement fact-checking, human review loop |
| **Integration with lab systems** | Medium | Medium | Early prototyping with IT, API design |
| **Team skill gap** | Medium | Medium | Hire/train ML engineer by month 2 |

***

## Part 10: Recommended Next Steps (This Week)

1. **⚡ Hardware:** 
   - [ ] Obtain RTX 6000 Ada quote + lead time (typically 4-6 weeks)
   - [ ] Identify hosting location (on-site cooling/power adequate?)

2. **📊 Data Preparation:**
   - [ ] Audit internal R&D documentation
   - [ ] Identify 3-5 chemistry/materials domain experts for labeling
   - [ ] Start curating initial 5k example dataset (next 2 weeks)

3. **👨‍💻 Team & Hiring:**
   - [ ] Identify internal ML champion (part-time, month 1)
   - [ ] Scope full-time ML engineer (hire by month 2)
   - [ ] Plan chemistry domain expert consultation (0.5 FTE)

4. **🏗️ Infrastructure:**
   - [ ] Set up development environment on Pegasus
   - [ ] Document current marina.cbnano.com network topology
   - [ ] Plan 10GbE upgrade if needed

5. **📚 Proof of Concept:**
   - [ ] Week 1-2: Simple RAG demo (Llama-2-13B + Milvus)
   - [ ] Week 3-4: Show chemistry Q&A capability to stakeholders
   - [ ] Week 5-6: Begin fine-tuning on internal chemistry data

***

## Conclusion

This infrastructure plan balances **pragmatism with ambition**. Rather than building everything at once, it creates a foundation (RAG + inference) that delivers immediate value, then systematically adds specialized capabilities through domain-specific fine-tuning.

**Key Success Factors:**
1. Invest in the right GPU first (RTX 6000 Ada, not Mac)
2. Start RAG implementation immediately (low barrier to entry)
3. Pair hardware with strong ML engineering (hire early)
4. Build tight feedback loop with chemistry team
5. Iterate on fine-tuning with real internal data

**Expected Outcomes by Month 6:**
- ✅ Production RAG system indexing 500k+ documents
- ✅ Domain-specific chemistry LLM deployed internally
- ✅ Multi-step AI agents handling 70%+ of routine ideation tasks
- ✅ 20-30% reduction in R&D cycle time
- ✅ Foundation for scaling to additional domains (materials, thermal, etc.)

**Estimated ROI: 18-24 months** (accounting for engineering time and hardware costs)

***

**Document Prepared By:** AI Infrastructure Strategy Analysis  
**Contact:** [Your ML Engineering Lead]  
**Last Updated:** December 29, 2025

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_97d29e25-0cef-4c8d-9bcd-8e65e9554e42/42f5492f-6514-4bb0-aab7-368729ea0f83/inventory-atom.json)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_97d29e25-0cef-4c8d-9bcd-8e65e9554e42/07b449a3-53b2-4be1-a718-2eaa094238e1/README.md)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_97d29e25-0cef-4c8d-9bcd-8e65e9554e42/49446ff3-1474-4d01-a038-4095cd4d474e/netprobe.json)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_97d29e25-0cef-4c8d-9bcd-8e65e9554e42/b59b503c-b284-40dd-9d36-0da4ac54ae96/inventory-zima.json)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_97d29e25-0cef-4c8d-9bcd-8e65e9554e42/362d68e1-3440-4b37-a363-56d584491f66/inventory-pegasus.json)
[6](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_97d29e25-0cef-4c8d-9bcd-8e65e9554e42/e2f424a8-f5d4-4441-9aad-779c36bc48b1/inventory-lab.json)
[7](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_97d29e25-0cef-4c8d-9bcd-8e65e9554e42/3523d818-5c12-40f8-9737-0f0d9e6e3ee3/inventory-ae86.json)
[8](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_97d29e25-0cef-4c8d-9bcd-8e65e9554e42/6184e7d3-d304-4708-ab26-12a271231577/ae86-NAS-ZFS-Samba-summary.md)
[9](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_97d29e25-0cef-4c8d-9bcd-8e65e9554e42/6cf1d3be-4ff4-4049-ab19-edc72d959902/marina.cbnano.com-Intranet-Setup.md)
[10](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_97d29e25-0cef-4c8d-9bcd-8e65e9554e42/cf7eb508-01e2-4b3c-95b5-50d437a0a455/Inventory-Home.md)
[11](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_97d29e25-0cef-4c8d-9bcd-8e65e9554e42/99d3d5c6-2c8c-426e-aba1-7f90ef8edcb8/Inventory-CB-Nano-Marina.md)
