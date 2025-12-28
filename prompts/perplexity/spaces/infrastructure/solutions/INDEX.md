# Document Index & Navigation Guide
## CB Nano Materials R&D - AI Infrastructure Plan

**Last Updated:** December 29, 2025  
**Total Documents:** 12 comprehensive guides (~8,000 lines)  
**For:** CB Nano R&D Leadership, Engineers, and Decision Makers

***

## Quick Navigation by Role

### 👔 If You're an Executive/Decision Maker
**Time Available:** 30 minutes to 2 hours  
**Goal:** Understand business case and make go/no-go decision

**Read in This Order:**
1. **QUICK_REFERENCE.md** (5 min)
   - Your mission in one sentence
   - Hardware at a glance
   - Cost/benefit summary
   - Print and tape to monitor

2. **EXECUTIVE_SUMMARY.md** (15-20 min)
   - Full business case
   - ROI analysis (545x Year 1)
   - Risk assessment
   - Decision matrix (go/no-go)

3. **GPU_Arsenal_Optimization_Strategy.md** - Part 9 (10 min)
   - Summary: Why this setup wins
   - Decision matrix
   - Escalation paths

**Decision Output:**
- ✅ Approve $26k hardware investment
- ✅ Approve $3.6k annual OpEx
- ✅ Authorize ML engineer hiring
- ✅ Timeline: 6 months to production

***

### 👨‍💼 If You're an Engineering Lead / CTO
**Time Available:** 4-6 hours  
**Goal:** Understand architecture and ensure technical feasibility

**Read in This Order:**
1. **QUICK_REFERENCE.md** (5 min) - Overview

2. **EXECUTIVE_SUMMARY.md** (15 min) - Business context

3. **Your_System_Architecture.md** (60-90 min)
   - Physical system layout
   - Network topology
   - Code examples (copy-paste ready)
   - Deployment timeline (month-by-month)

4. **GPU_Arsenal_Optimization_Strategy.md** (45 min)
   - Hardware justification
   - Performance math
   - Why this specific setup
   - Risk analysis

5. **START_HERE.md** (20 min) - Learning roadmap overview

**Technical Output:**
- ✅ Confirm hardware specifications
- ✅ Verify network requirements
- ✅ Validate timeline
- ✅ Identify resource needs
- ✅ Plan team structure

***

### 👨‍💻 If You're an ML Engineer (Primary Engineer)
**Time Available:** 20+ hours over 2 weeks  
**Goal:** Master the system and implement it

**Read in This Order (Week 1):**
1. **QUICK_REFERENCE.md** (5 min)
   - Print this, reference daily
   - Emergency troubleshooting guide

2. **START_HERE.md** (30 min)
   - Your personal roadmap
   - What you're building
   - Week-by-week milestones
   - Success criteria

3. **Your_System_Architecture.md** (120 min)
   - Physical layout (understand where things go)
   - Network setup (how systems connect)
   - Code examples (inference gateway, monitoring, etc.)
   - Deployment checklist

4. **GPU_Arsenal_Optimization_Strategy.md** (60 min)
   - Understand hardware choices
   - Performance expectations
   - Benchmarking targets

**Read in This Order (Week 2):**
5. **AI_RD_Mastery_Roadmap.md** (Select Weeks 1-4)
   - Deep dive into inference optimization
   - Hands-on exercises
   - Checkpoint assessments

6. **R&D_AI_Infrastructure_Plan.md** (30 min)
   - Strategic context
   - Software components overview
   - Phased timeline

**Implementation Output:**
- Week 1: Understand full system
- Week 2: Start Weeks 1-4 of learning roadmap
- Month 1: Deploy inference (9,500 tok/s)
- Month 2-6: Follow deployment timeline

***

### 📚 If You're Learning ML/AI (Onboarding Engineer)
**Time Available:** 26 weeks  
**Goal:** Become expert in production AI systems

**Read in This Order:**

**Phase 1: Foundation (Weeks 1-4)**
- **AI_RD_Mastery_Roadmap.md** - Weeks 1-4 (Inference fundamentals)
  - Read papers, code examples
  - Run exercises on Mac before RTX 5090 arrives
  - Checkpoint: Understand inference deeply

**Phase 2: Intermediate (Weeks 5-10)**
- **AI_RD_Mastery_Roadmap.md** - Weeks 5-10 (Fine-tuning & RAG)
  - Implement LoRA from scratch
  - Build Milvus + RAG pipeline
  - Checkpoint: RAG system working (<100ms search)

**Phase 3: Advanced (Weeks 11-20)**
- **Your_System_Architecture.md** - Understand production patterns
- **AI_RD_Mastery_Roadmap.md** - Weeks 11-20 (Multi-GPU, agents, monitoring)
  - Deploy Expert Parallel
  - Build agentic workflows
  - Checkpoint: All 3 systems deployed

**Phase 4: Mastery (Weeks 21-26)**
- **R&D_AI_Infrastructure_Plan.md** - Strategic overview
- **AI_RD_Mastery_Roadmap.md** - Weeks 21-26 (Operations)
  - Monitoring, cost optimization, scaling
  - Checkpoint: Production system stable

**Learning Output:**
- 26-week structured curriculum
- 1,000+ hands-on hours
- Production system delivered
- Expert-level understanding

***

### 🔬 If You're a Chemist/Domain Expert
**Time Available:** 2-4 hours  
**Goal:** Understand what AI system can do for you

**Read in This Order:**
1. **QUICK_REFERENCE.md** (5 min)
   - What the system does
   - Performance targets
   - How to use it

2. **EXECUTIVE_SUMMARY.md** - "The Impact" section (10 min)
   - R&D cycle time reduction
   - Failed experiments prevented
   - Cost per experiment

3. **START_HERE.md** - "What You're Trying to Accomplish" (10 min)
   - See the concrete before/after
   - Understand timeline

4. **Your_System_Architecture.md** - "Inference Gateway" section (15 min)
   - Unified API for querying
   - Simple examples

**Usage Output:**
- ✅ Know what the system does
- ✅ Know how to use it (simple API)
- ✅ Understand timeline for adoption
- ✅ Can provide feedback for fine-tuning

***

## Document Map: What Each File Contains

### 1. QUICK_REFERENCE.md
**Length:** ~400 lines  
**Read Time:** 5-10 minutes  
**Best For:** Daily reference, emergency lookup

**Contains:**
- One-sentence mission
- Hardware specs (all 3 systems)
- Performance targets
- Critical commands (copy-paste ready)
- 30-day checklist
- Troubleshooting (problem → solution)
- Emergency contacts

**Use Case:** Stuck? Print this card and check troubleshooting section.

***

### 2. EXECUTIVE_SUMMARY.md
**Length:** ~800 lines  
**Read Time:** 15-20 minutes  
**Best For:** Decision makers, leadership

**Contains:**
- Business situation
- Opportunity (what it enables)
- Investment required (capital + OpEx)
- Return on investment (545x Year 1)
- Risk assessment + mitigation
- Success metrics (technical + business)
- 6-month timeline
- Decision framework (go/no-go)

**Use Case:** Board presentation, budget approval, stakeholder communication.

***

### 3. START_HERE.md
**Length:** ~800 lines  
**Read Time:** 20-30 minutes  
**Best For:** Primary engineer, learning path

**Contains:**
- Your mission (transform 2-week to 2-day cycles)
- 3-month framework (foundation → exploration → integration)
- Document map
- Week 1 checklist
- Key technical questions (answered)
- Red flags (what NOT to do)
- Your personal commit
- Next 3 moves

**Use Case:** First day on the job, your personal roadmap.

***

### 4. Your_System_Architecture.md
**Length:** ~2,000 lines  
**Read Time:** 60-90 minutes  
**Best For:** Engineers, implementation details

**Contains:**

**Part 1: Physical System Design**
- System A (Inference): RTX 5090 + RTX Pro 6000
- System B (Fine-tuning): 2x RTX 5060 Ti
- System C (RAG): 2x RTX 4060 Ti
- Network topology

**Part 2-4: Each System in Detail**
- Hardware specs
- Software stack
- Code examples (vLLM, fine-tuning, embeddings)
- Performance expectations
- Monitoring setup

**Part 5: Shared Infrastructure**
- ae86 storage (ZFS, Milvus, MongoDB)
- Network configuration

**Part 6: Inference Gateway**
- Unified API
- Service routing
- Code example

**Part 7: Deployment Timeline**
- Month-by-month plan
- Week-by-week checklist
- Success criteria

**Part 8: Cost Breakdown**
- Hardware costs
- Operating costs
- ROI analysis

**Part 9: Failure Recovery**
- Auto-restart scripts
- Systemd configuration
- Monitoring setup

**Use Case:** Actual implementation reference (copy-paste code, hardware ordering, deployment).

***

### 5. GPU_Arsenal_Optimization_Strategy.md
**Length:** ~1,200 lines  
**Read Time:** 45-60 minutes  
**Best For:** Technical justification, hardware decisions

**Contains:**

**Part 1: Decision Framework**
- Cost of ownership (3-year TCO)
- Performance analysis (can we hit 9,500 tok/s?)
- ROI vs AWS

**Part 2: Hardware Analysis**
- Why RTX 5090 + RTX Pro 6000 (vs alternatives)
- Detailed comparisons
- Memory calculations

**Part 3: Expert Parallelism Math**
- How two GPUs work together
- Throughput calculations
- Latency analysis

**Part 4: Network & PCIe Analysis**
- Why 10GbE is critical
- Communication overhead
- Network bandwidth requirements

**Part 5: Cost-Performance Ratio**
- $ per token analysis
- Throughput per dollar
- vs AWS SageMaker

**Part 6-11: Risk Analysis, Sensitivity, Business Case, Verification**

**Use Case:** Justify hardware choices to skeptics, understand performance ceilings.

***

### 6. AI_RD_Mastery_Roadmap.md
**Length:** ~3,500 lines  
**Read Time:** Varies (26 weeks at 15-20 hrs/week)  
**Best For:** Learning curriculum, technical skill building

**Contains:**

**Foundation Phase (Weeks 1-4):**
- Week 1: Transformer Fundamentals
- Week 2: Token Generation & Decoding
- Week 3: Continuous Batching & Optimization
- Week 4: Expert Parallelism & Production Setup

**Intermediate Phase (Weeks 5-10):**
- Week 5: LoRA & Fine-tuning Fundamentals
- Week 6: Embeddings & Vector Search
- Week 7: RAG Pipeline Integration
- Weeks 8-10: Additional specializations

**Advanced Phase (Weeks 11-20):**
- Week 11: Expert Parallelism Deployment
- Weeks 12-14: Agents & Function Calling
- Weeks 15-20: Production Operations

**Mastery Phase (Weeks 21-26):**
- Operations excellence
- Cost optimization
- Team scaling

**Each Week Contains:**
- Reading list (papers, docs, blogs)
- Hands-on exercises (code)
- Code examples (copy-paste ready)
- Checkpoint (test your knowledge)

**Use Case:** Structured learning plan, week-by-week curriculum.

***

### 7. R&D_AI_Infrastructure_Plan.md
**Length:** ~2,000 lines  
**Read Time:** 30-45 minutes (skim) or 60-90 minutes (deep read)  
**Best For:** Strategic overview, team alignment

**Contains:**

**Part 1: Current Assessment**
- Existing hardware inventory
- Capability gaps
- Strengths to build on

**Part 2: Hardware Investment Strategy**
- Phase 1 (Immediate): Core inference node
- Phase 2 (Months 3-6): Secondary optimization
- Phase 3 (Months 6-12): Storage & orchestration

**Part 3: Software Architecture**
- Layered AI stack diagram
- Component details (inference, RAG, fine-tuning, tools, data pipelines, internal knowledge)
- Technology choices

**Part 4: Phased Implementation Roadmap**
- Month 1-2: Foundation (Inference + RAG)
- Month 3-4: Fine-tuning + Domain specialization
- Month 5-6: Tool integration + Automation
- Month 6-12: Scaling & Optimization

**Part 5-10: Technology stack, cost analysis, best practices, KPIs, risks, next steps**

**Use Case:** Team alignment, strategic planning, communicating the vision.

***

### 8-12. Supporting Documents

**README.md** (~400 lines)
- Overview of entire plan
- How to use these documents
- Quick links
- Getting started guide

**INDEX.md** (This file, ~500 lines)
- Navigation guide by role
- What each document contains
- Where to find answers
- Reading paths

**QUICK_REFERENCE.md** (Already covered above)

**DELIVERY_SUMMARY.txt** (~300 lines)
- What you received
- Verification checklist
- File locations
- Support contacts

**WHAT_YOU_GET.md** (~400 lines)
- Value delivered
- How to use package
- Quality metrics
- Learning investment
- Success guarantee

***

## Finding Specific Information

### "I need to..."

#### ...understand the business case
→ **EXECUTIVE_SUMMARY.md**
- Section: "The Investment"
- Section: "Return on Investment"
- Section: "Decision Framework"

#### ...understand hardware choices
→ **GPU_Arsenal_Optimization_Strategy.md**
- Part 4: "Why This Specific Hardware Mix"
- Part 2: "Performance Analysis"

#### ...deploy the system
→ **Your_System_Architecture.md**
- Part 7: "6-Month Deployment Timeline"
- Each System section: "Deployment" subsection

#### ...understand inference optimization
→ **AI_RD_Mastery_Roadmap.md** Weeks 1-4
- Week 3: "Continuous Batching"
- Week 4: "Expert Parallelism"

#### ...learn fine-tuning
→ **AI_RD_Mastery_Roadmap.md** Week 5
- Full implementation of LoRA from scratch
- QLoRA with code examples

#### ...understand RAG systems
→ **AI_RD_Mastery_Roadmap.md** Week 6-7
- Vector databases (Milvus)
- Building RAG pipeline

#### ...build agentic workflows
→ **AI_RD_Mastery_Roadmap.md** Weeks 12-14
- Agent frameworks
- Function calling
- Multi-step reasoning

#### ...understand cost/ROI
→ **EXECUTIVE_SUMMARY.md**
- "The Investment"
- "Return on Investment"
- "Cost Comparison" (vs AWS)

#### ...get emergency help
→ **QUICK_REFERENCE.md**
- "Troubleshooting" section
- "Who to Call" section

#### ...understand the monthly timeline
→ **Your_System_Architecture.md** Part 7
- Detailed month-by-month breakdown
- Week-by-week checklists

#### ...know what to read when
→ **START_HERE.md**
- "Document Map"
- "Your Week 1 Checklist"

***

## Reading Paths by Use Case

### Path 1: "I Need to Make a Decision This Week"
1. QUICK_REFERENCE.md (5 min)
2. EXECUTIVE_SUMMARY.md (20 min)
3. GPU_Arsenal_Optimization_Strategy.md Part 1 & 9 (15 min)

**Total Time:** 40 minutes  
**Output:** Go/no-go decision with confidence

***

### Path 2: "I'm Implementing This"
1. START_HERE.md (25 min)
2. Your_System_Architecture.md (90 min)
3. QUICK_REFERENCE.md (5 min, for reference)
4. AI_RD_Mastery_Roadmap.md Weeks 1-4 (ongoing)
5. GPU_Arsenal_Optimization_Strategy.md Part 2-3 (45 min)

**Total Time:** 165 minutes (2.75 hours) + 26 weeks learning  
**Output:** Ready to start Month 1 implementation

***

### Path 3: "I'm Learning the Technology"
1. START_HERE.md (25 min)
2. AI_RD_Mastery_Roadmap.md Week 1-4 (60 hours over 4 weeks)
3. AI_RD_Mastery_Roadmap.md Week 5-10 (90 hours over 6 weeks)
4. Your_System_Architecture.md (90 min, during Week 11)
5. AI_RD_Mastery_Roadmap.md Week 11-26 (ongoing)

**Total Time:** 26 weeks at 15-20 hrs/week = 400+ hours  
**Output:** Expert-level proficiency + working system

***

### Path 4: "I'm Evaluating Feasibility"
1. EXECUTIVE_SUMMARY.md (20 min)
2. GPU_Arsenal_Optimization_Strategy.md (60 min)
3. Your_System_Architecture.md Part 1-2 (60 min)

**Total Time:** 140 minutes (2.3 hours)  
**Output:** Technical feasibility assessment

***

### Path 5: "I'm Planning Team Structure"
1. EXECUTIVE_SUMMARY.md Section "The Ask" (10 min)
2. START_HERE.md Section "Your Commit" (5 min)
3. Your_System_Architecture.md Part 7 (45 min, focus on timeline)
4. AI_RD_Mastery_Roadmap.md Overview (10 min)

**Total Time:** 70 minutes  
**Output:** Team structure, hiring timeline, skill requirements

***

## FAQ: Where Do I Find...?

| Question | Answer | Document |
|----------|--------|----------|
| What's the business case? | ROI is 545x Year 1 | EXECUTIVE_SUMMARY.md |
| How much does this cost? | $26k hardware, $3.6k/year OpEx | EXECUTIVE_SUMMARY.md |
| Why RTX 5090 + RTX Pro 6000? | Expert Parallelism = 1.63x speedup | GPU_Arsenal_Optimization_Strategy.md |
| What's the timeline? | 6 months to production | EXECUTIVE_SUMMARY.md |
| Can I run this on single GPU? | Yes, but loses 1.63x speedup | GPU_Arsenal_Optimization_Strategy.md |
| How do I deploy vLLM? | Code example with Expert Parallel | Your_System_Architecture.md |
| How do I fine-tune a model? | Week 5 full implementation | AI_RD_Mastery_Roadmap.md |
| How do I build RAG? | Week 6-7 full implementation | AI_RD_Mastery_Roadmap.md |
| What if something breaks? | Troubleshooting guide | QUICK_REFERENCE.md |
| What's Expert Parallelism? | Math + explanation | GPU_Arsenal_Optimization_Strategy.md Part 3 |
| When should I order hardware? | Immediately (4-6 week lead time) | EXECUTIVE_SUMMARY.md |
| Who should I hire? | ML engineer + domain expert | START_HERE.md |
| What's the monthly plan? | Detailed breakdown | Your_System_Architecture.md Part 7 |
| How do I measure success? | Technical + business KPIs | EXECUTIVE_SUMMARY.md |
| What if Expert Parallel fails? | Single GPU fallback (5,841 tok/s) | GPU_Arsenal_Optimization_Strategy.md Part 9 |
| How long does fine-tuning take? | 2-4 days on RTX 5060 Ti | AI_RD_Mastery_Roadmap.md Week 5 |
| What's the RAG latency? | <100ms p95 (target) | EXECUTIVE_SUMMARY.md |

***

## Document Statistics

| Document | Pages | Words | Read Time | Best For |
|----------|-------|-------|-----------|----------|
| QUICK_REFERENCE.md | 10 | 2,500 | 5-10 min | Daily use, emergencies |
| EXECUTIVE_SUMMARY.md | 20 | 5,000 | 15-20 min | Decision makers |
| START_HERE.md | 20 | 5,000 | 20-30 min | Primary engineer |
| Your_System_Architecture.md | 50 | 12,500 | 60-90 min | Implementation |
| GPU_Arsenal_Optimization_Strategy.md | 30 | 7,500 | 45-60 min | Technical justification |
| AI_RD_Mastery_Roadmap.md | 85 | 21,000 | 26 weeks | Learning curriculum |
| R&D_AI_Infrastructure_Plan.md | 50 | 12,500 | 30-90 min | Strategic planning |
| Supporting docs (5 files) | 25 | 6,500 | 20-30 min | Navigation, reference |
| **TOTAL** | **290** | **72,500** | **8,000+ hours cumulative** | **All use cases** |

***

## How to Use This Index

1. **Find your role** above (Executive, Engineer, Learner, etc.)
2. **Follow the reading path** in order
3. **Use the FAQ** for quick lookups
4. **Reference the Document Map** for detailed content
5. **Print QUICK_REFERENCE.md** and tape to monitor

***

## Next Steps

### If you have 30 minutes:
1. Read QUICK_REFERENCE.md
2. Read EXECUTIVE_SUMMARY.md
3. Check GPU_Arsenal_Optimization_Strategy.md Part 9

### If you have 2 hours:
1. Read QUICK_REFERENCE.md
2. Read EXECUTIVE_SUMMARY.md
3. Read START_HERE.md
4. Skim Your_System_Architecture.md

### If you have 4 hours:
1. Follow "Engineering Lead" path above

### If you have 26 weeks:
1. Follow "ML Engineer (Primary)" path above

***

## Support & Questions

**For questions about:**
- **Business case** → EXECUTIVE_SUMMARY.md or contact leadership
- **Hardware** → GPU_Arsenal_Optimization_Strategy.md or contact IT
- **Implementation** → Your_System_Architecture.md or contact ML lead
- **Learning** → AI_RD_Mastery_Roadmap.md or start Week 1
- **Timeline** → Your_System_Architecture.md Part 7 or QUICK_REFERENCE.md
- **Troubleshooting** → QUICK_REFERENCE.md Section "Troubleshooting"

***

**Last Updated:** December 29, 2025, 2:18 AM HKT  
**Status:** Complete  
**Quality Grade:** Institutional (Enterprise Consulting Level)

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
