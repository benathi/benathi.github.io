# What Actually Works for Fast Inference

Three things really move the needle for LLM inference speed. Everything else is incremental. Here's a peek into each, what we've learned, and where the frontier is.

---

## 1. Speculative Decoding

The single biggest lever for decode speed. A small, fast "drafter" model guesses what the big model will say, then the big model verifies those guesses in one parallel pass. When the drafter is right, you get multiple tokens for the cost of one forward pass. When it's wrong, you lose nothing.

The hard part isn't the algorithm. It's making it work *in practice*, *on real traffic*, *continuously*.

### The progression

```
Manual & Static                     Adaptive                        Autonomous
─────────────────────────────────────────────────────────────────────────────►

Bifurcated       Custom         DAS              ATLAS            Aurora
Attention        Speculative    (Beat the        (Adaptive        (When RL Meets
(ICML 2024)      Decoding       Long Tail)       Learning         Spec. Training)
                 (blog)                          Speculator)

Prerequisite:    Train on       Exploit RL       Pair static +    Frame as RL:
fast shared-     real customer  rollout          lightweight      accepted = reward
prefix batch     traffic for    distributions;   adaptive         rejected = penalty
decode via       domain-        suffix-tree      drafter +        Hot-swap in prod
split GEMMs      specific       drafter for      confidence       Zero downtime
                 drafters       long-tail        controller       Day-0 deployment
                                trajectories

2-6x batch       1.2-1.45x     50% rollout      501 TPS on       1.5x day-0
speedup          decode         time reduction   DeepSeek-V3.1    +1.25x over
                 speedup                         60% RL train     static baselines
                                                 time reduction
```

### What we learned

**Data matters more than algorithms for speculation.** No matter how clever the speculation algorithm, if the drafter is trained on the wrong data, acceptance rates tank. Custom SD showed this first — domain-specific data beats generic. DAS took it further: for RL rollouts, the rollout history itself is the best training signal. ATLAS made it online. Aurora made it continuous.

**The long tail kills you.** In RL training, a small fraction of very long trajectories dominate wall-clock time. DAS specifically targets these with more aggressive speculation budgets. This long-tail awareness turns speculative decoding from a "nice to have" into a "50% speedup."

**Training and serving shouldn't be separate.** ATLAS's insight was pairing a static and adaptive drafter. Aurora's bigger insight was that the boundary is artificial — just run RL on live traces and hot-swap. The speculator improves as it serves.

### Adjacent work

- **I-DLM** (COLM 2026) explores an alternative: diffusion-based parallel generation that matches AR quality (72.5 AIME-24, 3.1x throughput). Could complement or replace speculation for certain workloads.
- **CDLM** accelerates diffusion models via consistency modeling (3.6-14.5x faster sampling).

---

## 2. Quantization & Compression

Memory is the bottleneck for serving. The KV cache grows linearly with sequence length and batch size. For Llama 4 Scout at max context (10M tokens), the KV cache alone is 1.8 TiB — 8x the model weights. Compress the cache, and you serve more users, longer contexts, or both.

### The landscape

```
                    How aggressive?
                    
        Conservative (4-bit)                    Aggressive (2-bit / sparsity)
        ────────────────────────────────────────────────────────────────────

        System-Aware KV                         Kitty
        INT4 + Hadamard rotation                2-bit KV cache
        Zero serving overhead                   8x memory reduction
        "simplest method that works"            2-4x throughput

        ScaleSearch                             TEAL
        NVFP4 with searched scales              40-50% activation sparsity
        Tensor Core native FP4 attention        Training-free
        4.5-bit KV cache                        Stacks with quantization

                            CARE
                            GQA → MLA conversion
                            Same KV budget, way better quality
                            215x PPL improvement over naive SVD
```

### What we learned

**The serving system is the constraint, not the algorithm.** System-Aware KV-Cache (COLM 2026) tested every KV quantization method under real serving conditions — paged memory, fused attention, concurrent workloads. Most sophisticated methods break. The winner? Token-wise INT4 + block-diagonal Hadamard rotation. Simple, but actually works in production with zero overhead.

**The standard quantization scale is wrong.** ScaleSearch's key observation: GPU BFP formats use max-magnitude to pick block scales, but this minimizes range, not error. Searching over NVFP4's mantissa bits for the error-minimizing scale cuts quantization error by 26%. This improves weight PTQ by up to 7.5 points (GPQA), video generation by 14 points (VQA-a), and enables near-lossless FP4 attention.

**Don't just compress — restructure.** CARE doesn't make the KV cache smaller; it converts the attention mechanism from GQA to MLA, so the same budget goes further. Activation-aware factorization instead of naive SVD is the key — 215x better perplexity at matched budgets.

**Sparsity and quantization compound.** TEAL's 40-50% activation sparsity is orthogonal to weight quantization. Combined, the gains multiply: fewer bits per value *and* fewer values to process.

### Adjacent work

- **Opportunistic Expert Activation** — reduces MoE memory pressure by re-routing tokens to already-loaded experts. 39% MoE layer latency reduction, no retraining.
- **Ladder-Residual** (ICML 2025) — redesigns residual connections to overlap communication with computation in tensor parallelism. 29% end-to-end speedup for 70B models.

---

## 3. Smart Routing & Multi-Model

The most counterintuitive finding: you don't always need the biggest model. Sometimes an ensemble of cheap models beats the expensive one. Sometimes you just need to pick the right model for each sub-task. The trick is knowing when and how.

### The progression

```
Prove it              Distill it            Make it fast         Make it cheap
────────────────────────────────────────────────────────────────────────────►

MoA                   MoAA                  Staircase            Squeeze Evolve
(ICLR 2025)           (ICML 2025)           Streaming            (2026)

Layered multi-agent   Distill ensemble      Start response       Allocate strong
Open-source beats     into small models     from partial         vs. cheap models
GPT-4o at 65.1%       LLaMA-8B: 19→48      outputs. 93%         by marginal utility.
AlpacaEval            Arena-Hard            TTFT reduction       3x cost reduction.
                      Self-improving loop                        10x throughput.
```

### Beyond ensembles: the orchestration layer

- **Plan/Divide/Conquer** (ICLR 2026) — when model fidelity decays superlinearly with input length, chunking + cheap models beats GPT-4o in a single shot. The framework tells you *which strategy to use before you waste compute trying*.

- **Squeeze Evolve** — within an evolutionary pipeline, different stages have different marginal utility from a strong model. Allocate the expensive model only where it matters. The economic endgame: not "use many models" but "spend each dollar where it has the highest return."

- **ThunderAgent** — the operating system that makes all of this run. "LLM Programs" co-schedule inference, KV caches, and tool execution across heterogeneous resources. Without this, you can't compose speculation + quantized caches + multi-model routing in a single system. 1.5-3.6x serving throughput, 1.8-3.9x RL rollouts.

### What we learned

**Collaborativeness is real.** Models produce better outputs when they see other models' responses, even from weaker models. This isn't averaging — there's a genuine collaborative effect. MoAA showed you can distill this into a single model that then surpasses any individual in the original ensemble.

**Latency was the blocker, not quality.** MoA's quality was never the problem. Staircase Streaming's 93% TTFT reduction is what made multi-agent inference deployable. One paper turned a research result into a product.

**Verification feeds routing.** Weaver ensembles weak verifiers to match o3-mini. V1 shows pairwise comparison beats pointwise scoring. CREST steers reasoning without training. These give you better routing signals — knowing which output is best — without expensive reward models.

---

## How They Compound

In a real serving system, these three multiply:

- **Spec decoding + quantization:** Aurora works better when Kitty compresses the KV cache — larger batches mean more parallelism for draft verification.
- **Quantization + routing:** Squeeze Evolve routes cheap models for easy stages. With CARE + Kitty, those cheap models serve at a fraction of the memory cost.
- **Routing + spec decoding:** ThunderAgent co-schedules everything. Multi-model pipelines use speculation within each model call, and the program-aware scheduler manages KV caches across the full agentic workflow.

The full stack: Aurora speculation + Kitty/ScaleSearch quantized KV + CARE MLA attention + TEAL sparsity + Staircase multi-agent streaming + ThunderAgent orchestration.

### Paper → Dimension Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DECODE TPS                                     │
│                                                                         │
│  Speculative Decoding Stack:                                            │
│    Bifurcated Attn ──▶ Custom SD ──▶ DAS ──▶ ATLAS ──▶ Aurora          │
│    (2-6x batch)   (1.2-1.45x) (50% RL) (501 TPS)  (self-improving)    │
│                                                                         │
│  Activation & Compute:                                                  │
│    TEAL (1.5-1.8x via sparsity)                                        │
│    Opp. Expert Activation (39% MoE layer speedup)                       │
│    Ladder-Residual (29% via comm overlap)                               │
│                                                                         │
│  New Generation Paradigms:                                              │
│    CDLM (3.6-14.5x diffusion speedup)                                  │
│    I-DLM (3.1x throughput, first DLM matching AR quality)               │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                        PREFILL LATENCY                                  │
│                                                                         │
│    Staircase Streaming (93% TTFT reduction for multi-agent)             │
│    Bifurcated Attn (shared-prefix batch decoding)                       │
│    ScaleSearch Attention (FP4 QK^T and PV on Tensor Cores)              │
│    Ladder-Residual (comm overlap helps prefill too)                     │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                       MEMORY REDUCTION                                  │
│                                                                         │
│  KV Cache Compression:                                                  │
│    Kitty (2-bit KV, 8x reduction, 2-4x throughput)                     │
│    System-Aware KV (INT4 + Hadamard, zero overhead in real serving)     │
│    ScaleSearch (4.5-bit NVFP4 KV cache)                                │
│    CARE (GQA→MLA conversion, same cache budget, 215x better PPL)       │
│                                                                         │
│  Activation Sparsity:                                                   │
│    TEAL (40-50% sparsity, compounds with quantization)                  │
│                                                                         │
│  Efficient Attention:                                                   │
│    Bifurcated Attn (no redundant prefix loading)                        │
│    Opp. Expert (fewer unique experts loaded per batch)                  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                      QUALITY PER FLOP                                   │
│                                                                         │
│  Multi-Model Routing & Orchestration:                                   │
│    MoA (open-source ensemble beats GPT-4o)                              │
│    MoAA (distill ensemble into small model)                             │
│    Squeeze Evolve (3x cost reduction via marginal-utility allocation)   │
│    Plan/Divide/Conquer (weak models + chunking beats GPT-4o)            │
│                                                                         │
│  Smarter Reasoning:                                                     │
│    Token Economies (simple baselines match complex strategies)           │
│    Think Deep/Fast (majority voting is competitive)                     │
│    CREST (+17.5% accuracy, -37.6% tokens, no training)                  │
│    V1 (pairwise verification: +10% Pass@1)                              │
│    Weaver (weak verifier ensemble matches o3-mini)                      │
│                                                                         │
│  Domain-Specific Quality:                                               │
│    Dragonfly / Dragonfly-Med (vision-language, medical)                 │
│    BioMed-R1 (reasoning-focused medical training)                       │
│    OpenBiomedVid (VLMs learning from educational videos)                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Key Insight: These Dimensions Multiply

Most papers improve one dimension. But the magic is that they **compose**:

- **Decode TPS x Memory** — Aurora's speculative decoding runs faster when the KV cache is compressed by Kitty (larger batches = more parallelism for speculation). TEAL's activation sparsity reduces per-token compute *and* memory.

- **Memory x Quality/FLOP** — CARE's GQA→MLA conversion doesn't just save memory; it improves attention expressivity at the same cache budget. Squeeze Evolve routes cheap models for easy stages, meaning memory is freed for the hard stages that need the big model.

- **Decode TPS x Quality/FLOP** — CREST steers reasoning to be both faster (fewer tokens) and better (higher accuracy). V1's pairwise verification is more compute-efficient than pointwise *and* more accurate. I-DLM generates tokens in parallel *without* sacrificing quality.

- **Prefill x Memory** — ScaleSearch enables FP4 attention computation directly on Tensor Cores (faster prefill) *and* stores KV in 4.5-bit format (less memory). System-Aware KV quantization achieves this with zero serving overhead.

### Papers That Span Multiple Dimensions

Some papers are hard to place in just one box because they attack several at once:

| Paper | Decode TPS | Prefill | Memory | Quality/FLOP |
|-------|:---:|:---:|:---:|:---:|
| **Bifurcated Attention** | x | x | x | |
| **TEAL** | x | | x | |
| **ScaleSearch** | | x | x | |
| **Kitty** | x | | x | |
| **CARE** | | | x | x |
| **Ladder-Residual** | x | x | | |
| **I-DLM** | x | | | x |
| **CREST** | x | | | x |
| **Aurora** | x | | | |
| **Staircase Streaming** | | x | | x |
| **Squeeze Evolve** | | | | x |
| **ThunderAgent** | x | x | x | |

### How This Perspective Helps Tell the Story

Instead of organizing by technique (speculative decoding, quantization, etc.), you can walk someone through the user-facing experience:

> **"Your generation is slow?"** → Here's the speculative decoding stack (Bifurcated Attn → Aurora), activation sparsity (TEAL), expert routing (Opp. Expert), architecture redesign (Ladder-Residual), and a whole new generation paradigm (I-DLM).
>
> **"Your prompt takes too long to process?"** → Staircase Streaming for multi-agent, ScaleSearch for FP4 attention, Ladder-Residual for communication overlap.
>
> **"You're running out of GPU memory?"** → Kitty for 2-bit KV, System-Aware for practical INT4, CARE for MLA conversion, TEAL for activation sparsity, ScaleSearch for FP4 KV.
>
> **"You want better answers without paying more?"** → MoA/MoAA for collective intelligence, Squeeze Evolve for cost-optimal routing, CREST/V1/Weaver for efficient reasoning, Plan/D&C for long context.

This is the "menu" framing — the user picks their bottleneck, and the research provides the solution stack.

---

## Perspective 2: Low-Level to High-Level (TODO)

*Coming back to this — the second slice organizes the same papers from bit-level operations up to system-level orchestration.*

---

## Appendix: Earlier Narrative Draft (kept for reference)

### The One-Sentence Version

Every paper is an answer to the same question: **"How do we get more intelligence per dollar?"** — and the answers compound.

### The Story

Imagine you're building the fastest race car in the world. You wouldn't just upgrade the engine — you'd also reduce drag, lighten the frame, add turbo, plan smarter pit stops, and redesign the track itself. That's what this body of work does for LLM inference. It's not one optimization; it's a **full-stack assault on the cost of intelligence**, where each layer of improvement multiplies with the others.

The story begins in **mid-2024** with two observations that define everything that follows:

**Observation 1: You're wasting most of your compute.** "Reasoning in Token Economies" (EMNLP 2024) showed something uncomfortable: most fancy reasoning strategies (multi-agent debate, Reflexion, etc.) don't actually beat simple chain-of-thought when you control for compute budget. The gains come from spending more tokens, not from better algorithms. This insight — *don't spend more compute, spend it smarter* — becomes the philosophical backbone of the entire research program.

**Observation 2: The hardware is underutilized.** "Bifurcated Attention" (ICML 2024) showed that standard attention wastes massive amounts of memory bandwidth by redundantly loading shared context for every sequence in a batch. A simple structural change (splitting attention into two GEMMs) yields 2-6x speedups. Meanwhile, "TEAL" (ICLR 2025) showed that 40-50% of activations in modern LLMs are near-zero and can be skipped — for free, no retraining needed. The hardware is there; we're just not using it well.

These two observations fork into **four parallel research tracks** that develop over the next two years and then reconverge:

---

### Track 1: Make Every Bit Count (Low-Level Optimization)

The question: *How much can you compress without losing anything?*

This track pushes the precision frontier downward. TEAL (2024) proved activation sparsity works training-free. Then the quantization papers arrived in force:

- **Kitty** pushes KV cache to 2-bit with dynamic channel precision — 8x memory reduction, 2-4x throughput gain.
- **ScaleSearch** discovers that the standard way GPUs pick quantization scales (max-magnitude) is suboptimal — searching for better scales in NVFP4's mantissa bits cuts quantization error by 26% and enables end-to-end FP4 attention.
- **System-Aware KV-Cache** (COLM 2026) asks the practical question: *which quantization methods actually work in a real serving system with paged memory?* Answer: the simplest one — INT4 + Hadamard rotation. Everything fancier breaks serving constraints.
- **CARE** (ICLR 2026) converts existing GQA models to the more efficient MLA attention format, using activation-aware factorization instead of naive SVD — 215x better perplexity at matched KV budgets.

But compression isn't the only lever. **Ladder-Residual** (ICML 2025) shows you can redesign the model architecture itself to overlap communication and computation in tensor parallelism — 29% inference speedup for a 70B model just by changing the residual connections. **Opportunistic Expert Activation** reduces MoE decode latency by 39% by dynamically re-routing tokens to piggyback on already-loaded experts. And **I-DLM** (COLM 2026) attempts to break out of autoregressive decoding entirely — building the first diffusion language model that matches AR quality while generating tokens in parallel.

The cross-cutting insight: **every layer of the model is leaving performance on the table**, and the gains stack multiplicatively.

---

### Track 2: Speculative Decoding — The Flagship Arc

The question: *What if a small model could predict what the big model will say?*

This is the most complete narrative arc — five papers that build directly on each other, each solving the limitation of the last:

**Chapter 1: The Foundation.** Bifurcated Attention (ICML 2024) isn't a speculative decoding paper per se, but it solves a prerequisite: making shared-prefix parallel decoding fast. When you're verifying multiple draft sequences against the same context, you need this.

**Chapter 2: Domain Customization.** The Customized Speculative Decoding blog (May 2025) shows that training small speculators on *actual customer traffic* yields 1.2-1.45x speedups and 49-61% GPU cost reduction. But there's a problem: you need offline data collection and manual retraining whenever the traffic distribution shifts.

**Chapter 3: Speculation Meets RL Training.** DAS (Nov 2025) realizes that RL post-training — where models generate thousands of rollouts for reinforcement learning — is a *perfect* use case for speculation. It builds adaptive drafters from historical rollouts using suffix trees and allocates speculation budgets based on trajectory length distributions. 50% rollout time reduction. But the speculator is still static.

**Chapter 4: Learning at Runtime.** ATLAS (Oct 2025) pairs a static speculator with a lightweight adaptive one that learns from live traffic, plus a confidence controller that picks between them. No manual tuning — it just gets faster as it runs. 501 TPS on DeepSeek-V3.1. But the system still treats training and serving as separate pipelines.

**Chapter 5: The Full Vision.** Aurora (2026) closes the loop entirely. It frames speculator training as an RL problem — accepted tokens are reward, rejected tokens are feedback — and runs training and serving in a single unified system. Speculators are hot-swapped without service interruption. It supports day-0 deployment (start serving immediately, improve continuously) and automatically adapts to domain drift. This is speculative decoding that *never stops getting faster*.

The arc: manual → domain-specific → distribution-aware → runtime-adaptive → fully autonomous.

---

### Track 3: Many Models Are Better Than One

The question: *Can weak models combine to beat strong ones?*

**MoA** (ICLR 2025) proved the concept: a layered architecture where multiple LLMs refine each other's outputs scored 65.1% on AlpacaEval 2.0, beating GPT-4o's 57.5% using *only open-source models*. The key insight was "collaborativeness" — models produce better outputs just from seeing other models' responses.

But MoA has two problems: it's slow (multiple serial inference rounds) and the intelligence stays in the pipeline rather than in any single model.

**MoAA** (ICML 2025) solves the second problem: it distills the ensemble's collective intelligence into small models via synthetic data generation + preference optimization. LLaMA-3.1-8B goes from 19.5 to 48.3 on Arena-Hard. Even the strongest models in the ensemble improve — a true self-improvement loop with no proprietary dependencies.

**Staircase Streaming** (2025) solves the first problem: instead of waiting for all intermediate agents to finish, begin the final response from partial outputs. TTFT drops by 93%. Multi-agent inference becomes practical for interactive use.

**Plan/Divide/Conquer** (ICLR 2026) extends the multi-model idea to long contexts. Rather than feeding 128K tokens to one model, split the document into chunks processed by cheaper models in parallel, with a manager aggregating. The theoretical framework explains *exactly when* this helps (model noise growing superlinearly with length) and when it doesn't (tasks needing subtle cross-document connections).

**Squeeze Evolve** (2026) brings economic optimization: allocate strong vs. cheap models based on marginal utility at each stage of evolutionary inference. 3x cost reduction, 10x throughput increase. The first verifier-free evolutionary method to match verifier-based methods.

The arc: prove it works → distill it → make it fast → understand when it works → make it cheap.

---

### Track 4: Spend Reasoning Compute Wisely

The question: *How do you make models think better without thinking longer?*

This track directly descends from the Token Economies insight (most compute is wasted on reasoning). The research develops both the *understanding* and the *solutions*:

**Understanding:** Think Deep/Think Fast (2025) does a comprehensive study and confirms: for reasoning models, simple majority voting matches or beats sophisticated methods. Non-reasoning models can't close the gap no matter how much compute you throw at them. The implication: *model quality matters more than inference strategy*.

**Solutions from the inside:** CREST (2025) discovers that specific attention heads govern reasoning behaviors like verification and backtracking. By steering these heads at inference time (no training!), you get +17.5% accuracy while using 37.6% fewer tokens. The model literally thinks better *and* faster.

**Solutions from the outside:** V1 (2026) shows that models are much better at pairwise comparison than absolute scoring. Its tournament-based self-verification improves Pass@1 by up to 10% on code and math. V1-PairRL jointly trains one model as both generator and verifier — no separate reward model needed.

**Verification at scale:** Weaver (2025) ensembles multiple imperfect verifiers via weak supervision, approaching oracle-level quality without any labeled data. Llama 3.3 70B + Weaver matches o3-mini (87.7% vs 86.7%) purely at inference time.

**Domain-specific:** Disentangling Reasoning and Knowledge in Medical LLMs (2025) reveals that biomedical models are much weaker at reasoning than at recall, and BioMed-R1 trained on reasoning-heavy examples achieves the best performance among its size class. The "spend wisely" philosophy applied to healthcare.

---

### Where the Tracks Converge: ThunderAgent and the Full Stack

All four tracks are optimizing different parts of the same system. But who orchestrates them?

**ThunderAgent** (2026) is the systems answer. It introduces "LLM Programs" — a unified abstraction that tracks the full lifecycle of agentic workflows, co-scheduling LLM inference, KV cache management, and tool execution across heterogeneous resources. It's the operating system layer where all the individual optimizations (speculative decoding, quantized KV caches, multi-model routing) actually compose.

And in a meta twist, the team uses **AI agents to build AI inference systems** (Aug 2025 blog) — using LLM agents to automate speculative decoding pipeline optimization, achieving 1.22-1.37x speedups with minimal human intervention. The tools are building themselves.

---

### The Compound Effect

Here's why this body of work is more than the sum of its parts. Consider a single inference request in 2026 vs. 2024:

1. The KV cache is **8x smaller** (Kitty 2-bit quantization) with **near-zero overhead** (System-Aware fused kernels)
2. Attention uses **MLA format** converted from GQA (CARE), with **FP4 Tensor Core operations** (ScaleSearch)
3. **40-50% of activations are skipped** (TEAL sparsity)
4. Communication is **hidden behind computation** (Ladder-Residual architecture)
5. The speculator **learns continuously from live traffic** (Aurora), with **50% faster RL rollouts** (DAS)
6. Multiple models collaborate but the **user sees a single low-latency stream** (Staircase Streaming)
7. Reasoning compute is **steered to productive paths** (CREST) and **verified by weak verifier ensembles** (Weaver)
8. The whole stack is **co-scheduled by a program-aware system** (ThunderAgent)

Each of these is a published, benchmarked result. Together, they represent a multiplicative speedup that no single paper could achieve — and a vision of inference infrastructure where the system is continuously, autonomously getting faster.

---

### The Diagram (Gamified Version)

Think of this as an RPG skill tree. You start at the center and unlock branches. Each node is a paper. Edges show dependencies and synergies.

```
                              ┌─────────────────────┐
                              │   THE QUEST:         │
                              │   Maximum Intelligence│
                              │   Per Dollar          │
                              └──────────┬────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
              ┌─────▼─────┐       ┌──────▼──────┐     ┌──────▼──────┐
              │ COMPRESS   │       │  PARALLELIZE │     │  THINK      │
              │ EVERYTHING │       │  EVERYTHING  │     │  SMARTER    │
              └─────┬──────┘       └──────┬──────┘     └──────┬──────┘
                    │                     │                    │
        ┌───────┬──┴──┬───────┐    ┌─────┴─────┐      ┌──────┴──────┐
        │       │     │       │    │           │      │             │
     ┌──▼──┐ ┌─▼──┐ ┌▼────┐ ┌▼─┐  │  ┌────┐   │   ┌──▼──┐    ┌────▼──┐
     │TEAL │ │Kit-│ │Scale│ │CA│  │  │Bif. │   │   │Token│    │Think  │
     │     │ │ty  │ │Srch │ │RE│  │  │Attn │   │   │Econ │    │Deep/  │
     │ICLR │ │    │ │     │ │  │  │  │ICML │   │   │EMNLP│    │Fast   │
     │2025 │ │    │ │MLSys│ │IC│  │  │2024 │   │   │2024 │    │       │
     └──┬──┘ └─┬──┘ └┬────┘ │LR│  │  └──┬──┘   │   └──┬──┘    └───┬───┘
        │      │     │      │26│  │     │      │      │            │
        ▼      ▼     ▼      └┬─┘  │     ▼      │      ▼            ▼
    ┌───────────────────┐    │    │  ┌──────┐  │   ┌──────┐   ┌────────┐
    │  Opp.Expert       │    │    │  │Custom│  │   │CREST │   │Weaver  │
    │  Ladder-Residual  │    │    │  │Spec. │  │   │      │   │        │
    │  Sys-Aware KV     │    │    │  │Decode│  │   │no    │   │weak    │
    │  ICML'25, COLM'26 │    │    │  └──┬───┘  │   │train │   │verify  │
    └───────┬───────────┘    │    │     │      │   └──┬───┘   └───┬────┘
            │                │    │     ▼      │      │           │
            │                │    │  ┌──────┐  │      ▼           ▼
            │                │    │  │DAS   │  │   ┌──────┐  ┌────────┐
            │                │    │  │RL    │  │   │V1    │  │BioMed  │
            │                │    │  │train │  │   │pair  │  │R1      │
            │                │    │  └──┬───┘  │   │verify│  └────────┘
            │                │    │     │      │   └──────┘
            │                │    │     ▼      │
            ▼                │    │  ┌──────┐  │
    ┌───────────────┐        │    │  │ATLAS │  │
    │  GENERATION   │        │    │  │learn │  │
    │  PARADIGMS    │        │    │  │at    │  │
    │               │        │    │  │run   │  │
    │  CDLM         │        │    │  └──┬───┘  │
    │  I-DLM        │        │    │     │      │
    │  (COLM'26)    │        │    │     ▼      │
    └───────┬───────┘        │    │  ┌──────┐  │
            │                │    │  │AURORA│  │         ┌───────────────┐
            │                │    │  │self- │  │         │ MULTI-MODEL   │
            │                │    │  │drive │  │         │               │
            │                │    │  └──┬───┘  │    ┌────▼────┐         │
            │                │    │     │      │    │MoA      │         │
            │                │    │     │      │    │ICLR'25  │         │
            │                │    │     │      │    └────┬────┘         │
            │                │    │     │      │         │              │
            │                │    │     │      │    ┌────▼────┐   ┌────▼────┐
            │                │    │     │      │    │MoAA     │   │Staircase│
            │                │    │     │      │    │ICML'25  │   │Stream   │
            │                │    │     │      │    └────┬────┘   └────┬────┘
            │                │    │     │      │         │             │
            │                │    │     │      │    ┌────▼────┐  ┌────▼─────┐
            │                │    │     │      │    │Squeeze  │  │Plan/D&C  │
            │                │    │     │      │    │Evolve   │  │ICLR'26   │
            │                │    │     │      │    └────┬────┘  └────┬─────┘
            │                │    │     │      │         │            │
            └────────────────┴────┴─────┴──────┴─────────┴────────────┘
                                         │
                              ┌──────────▼────────────┐
                              │    THUNDERAGENT        │
                              │    The Operating       │
                              │    System for All      │
                              │    of the Above        │
                              │                        │
                              │  1.5-3.9x throughput   │
                              │  co-schedules LLM +    │
                              │  tools + KV cache +    │
                              │  RL rollouts           │
                              └────────────────────────┘
```

**How to read this:** Start from the top (the quest). Each branch is a skill tree you level up. Papers lower in a branch build on papers above. The branches converge at ThunderAgent — the systems layer where everything composes.

**Cross-branch synergies (the combo multipliers):**
- TEAL sparsity + Kitty quantization + ScaleSearch FP4 = compound compression
- Bifurcated Attention + Aurora speculation = fast parallel verified decoding
- MoA collective intelligence + Staircase latency fix + Squeeze Evolve cost optimization = practical multi-model serving
- CREST reasoning steering + V1 pairwise verification + Weaver weak verifiers = efficient reasoning at every level
- Aurora continuous learning + DAS RL rollouts + ThunderAgent scheduling = self-improving RL training infrastructure

---

## Detailed Summaries

---

### 2024

---

#### Mixture-of-Agents (MoA)
**Paper:** [2406.04692](https://arxiv.org/abs/2406.04692) — ICLR 2025 | **Blog:** [together.ai/blog/together-moa](https://www.together.ai/blog/together-moa)

**Pillar:** Multi-Model Intelligence

With the growing number of available LLMs, this work explores how to harness the collective expertise of multiple models. MoA proposes a layered architecture where each layer contains multiple LLM agents, and each agent receives all outputs from the previous layer as context when generating its response. The key finding is "collaborativeness" — models produce better outputs when shown other models' responses, even from less capable systems.

MoA achieved state-of-the-art on AlpacaEval 2.0 with 65.1%, surpassing GPT-4o's 57.5% — using only open-source LLMs. This was the foundational result that proved many weak models can beat one strong model, launching an entire research direction for the team.

---

#### Reasoning in Token Economies
**Paper:** [2406.06461](https://arxiv.org/abs/2406.06461) — EMNLP 2024

**Pillar:** Scaling Reasoning Efficiently

This paper introduced a budget-aware evaluation framework for LLM reasoning strategies, arguing that traditional evaluations are misleading because they ignore compute cost. When chain-of-thought self-consistency is given comparable token budgets, it frequently matches or outperforms more elaborate methods like multi-agent debate and Reflexion.

The key insight: many complex reasoning strategies don't outperform simpler baselines due to algorithmic ingenuity — their gains come largely from consuming more compute. Some strategies even degrade with additional budget. This set the intellectual foundation for the team's later work on efficient test-time scaling.

---

#### Bifurcated Attention
**Paper:** [2403.08845](https://arxiv.org/abs/2403.08845) — ICML 2024

**Pillar:** Speculative Decoding / Low-Level Optimization

In shared-context batch decoding scenarios (e.g., generating multiple candidate answers from the same prompt), standard attention mechanisms incur redundant memory IO by repeatedly loading the shared prefix KV cache for each sequence. Bifurcated Attention splits the attention computation into two separate GEMMs: one over the shared prefill KV cache and one over the per-sequence decoding KV cache.

While maintaining identical computation (no approximation), this dramatically reduces memory IO — over 2.1x speedup when sampling 16 output sequences and more than 6.2x speedup when sampling 32 sequences at context lengths exceeding 8K tokens. This is a key building block for any parallel decoding or speculative decoding system that uses shared prefixes.

---

#### Token Alignment via Character Matching
**Paper:** [2403.08688](https://arxiv.org/abs/2403.08688) — ACL 2024 (Findings)

**Pillar:** Low-Level Optimization

Generative language models produce incorrect outputs when prompted with partial tokens (e.g., mid-word text or partial indentation in code). This tokenization artifact occurs because partial tokens fall outside the training distribution. Token Alignment backtracks to the last complete token boundary and constrains generation to match the original prompt prefix character-by-character.

A practical, targeted fix for code completion and text autocompletion systems with minimal latency overhead.

---

#### Dragonfly
**Paper:** [2406.00977](https://arxiv.org/abs/2406.00977)

**Pillar:** Multi-Model Intelligence (Vision)

Current VLMs struggle with fine-grained details from small objects, charts, and embedded text. Dragonfly extends multi-crop techniques by zooming in beyond native resolution and extracting features from many image sub-crops, using mean-pooling aggregation to manage compute cost.

Among 7-8B parameter models, Dragonfly ranks at the top across ten general-domain benchmarks. The biomedical variant, Dragonfly-Med, sets new benchmarks on medical tasks — 91.6% on SLAKE vs. 84.8% for Med-Gemini. This established the team's foothold in vision-language and biomedical AI.

---

#### TEAL: Training-Free Activation Sparsity
**Paper:** [2408.14690](https://arxiv.org/abs/2408.14690) — ICLR 2025 | **Blog:** [together.ai/blog/teal](https://www.together.ai/blog/teal-training-free-activation-sparsity-in-large-language-models)

**Pillar:** Low-Level Optimization

LLM inference is memory-bound, making activation sparsity attractive. Previous methods only worked with ReLU-based architectures, but modern models use SwiGLU. TEAL exploits the observation that hidden states follow Gaussian/Laplacian distributions, enabling effective magnitude pruning without any retraining.

TEAL achieves 40-50% model-wide sparsity with up to 1.8x decoding speedup across Llama-2/3 and Mistral families. Compatible with weight quantization for compounding gains. This was one of the team's first "low-level optimization" papers — proving you can get significant speedups without touching the model's weights.

---

#### RedPajama
**Paper:** [2411.12372](https://arxiv.org/abs/2411.12372) | NeurIPS 2024 Datasets & Benchmarks

**Pillar:** Systems / Data Infrastructure

RedPajama released two massive open datasets: V1 (a LLaMA training data reproduction) and V2 (100T+ raw web tokens with quality signals). The datasets have been adopted for training production models including Snowflake Arctic, Salesforce XGen, and AI2's OLMo.

This foundational data infrastructure work supports the broader ecosystem and Together AI's mission of open, transparent AI development.

---

### 2025

---

#### Ladder-Residual
**Paper:** [2501.06589](https://arxiv.org/abs/2501.06589) — ICML 2025

**Pillar:** Low-Level Optimization

Rather than just optimizing systems, this paper redesigns the model architecture itself. Ladder-Residual modifies residual connections to enable overlapping of communication and computation in tensor parallelism, effectively hiding communication latency.

For a 70B Transformer with TP across 8 GPUs, this achieves a 29% end-to-end wall-clock speedup at inference time. Parts of Llama-3.1 8B can be converted with minimal accuracy loss using only 3B tokens of retraining. A rare example of co-designing architecture and distributed systems for inference.

---

#### Think Deep, Think Fast
**Paper:** [2504.14047](https://arxiv.org/abs/2504.14047)

**Pillar:** Scaling Reasoning Efficiently

A comprehensive analysis of verifier-free inference-time scaling methods. The key findings: non-reasoning models cannot close the gap with reasoning models even with extreme budgets. For reasoning models, simple majority voting is as good or better than sophisticated methods like best-of-N or sequential revisions.

This is the spiritual successor to "Reasoning in Token Economies" — confirming and deepening the insight that smart compute allocation beats brute-force scaling.

---

#### Scaling Instruction-Tuned LLMs to Million-Token Contexts
**Paper:** [2504.12637](https://arxiv.org/abs/2504.12637) — ICLR 2025

**Pillar:** Systems / Long Context

Addresses the near-total absence of open-source instruction tuning data with contexts exceeding 100K tokens. A hierarchical synthetic data generation strategy combined with step-by-step RoPE scaling enables instruction-tuned LLMs to handle up to 1M token contexts while maintaining general task performance.

---

#### How Well Can General VLMs Learn Medicine by Watching Videos?
**Paper:** [2504.14391](https://arxiv.org/abs/2504.14391)

**Pillar:** Multi-Model Intelligence (Vision / Medical)

Fine-tuning general VLMs on 1031 hours of curated YouTube biomedical videos yields massive gains — up to 98.7% improvement on video tasks for a 2B model. Introduces OpenBiomedVid dataset and expert-curated evaluation benchmarks.

Extends the Dragonfly line into video, showing that educational content designed for humans is surprisingly effective for training models.

---

#### Boosting DeepSeek-R1 with Customized Speculative Decoding
**Blog:** [together.ai/blog/customized-speculative-decoding](https://www.together.ai/blog/customized-speculative-decoding)

**Pillar:** Speculative Decoding

Training small "speculator" models on actual customer inference traffic so they can draft tokens that the target model verifies in parallel. Performance scales with data volume: ~20M tokens already exceeds 1.10x speedup, reaching 1.20x at 50M tokens. GPU costs drop 23-26% vs. standard speculative decoding.

The key insight: domain-specific speculator customization is high-ROI for production deployments. This set the stage for the automated, continuous learning approach in ATLAS and Aurora.

---

#### Improving Model Alignment Through Collective Intelligence (MoAA)
**Paper:** [2505.03059](https://arxiv.org/abs/2505.03059) — ICML 2025 | **Blog:** [together.ai/blog/moaa](https://www.together.ai/blog/moaa)

**Pillar:** Multi-Model Intelligence

MoAA extends MoA from inference-time to training-time: it distills collective intelligence from multiple open-source LLMs into smaller models. Two-stage process: MoAA-SFT synthesizes diverse training data, then MoAA-DPO uses the ensemble as a reward model.

LLaMA-3.1-8B goes from 19.5 to 48.3 on Arena-Hard. Gemma-2-9B reaches performance comparable to Llama-3.1-70B. Even the strongest models in the ensemble benefit, creating a self-improving loop without proprietary model dependencies.

---

#### Disentangling Reasoning and Knowledge in Medical LLMs
**Paper:** [2505.11462](https://arxiv.org/abs/2505.11462)

**Pillar:** Scaling Reasoning Efficiently (Medical)

Builds a classifier to separate medical QA into reasoning vs. knowledge subsets. Finds only 32.8% of questions require complex reasoning, and biomedical LLMs are much weaker at reasoning than recall. BioMed-R1 trained with RL on reasoning-heavy examples achieves the best performance among similarly sized models.

Extends the "spend compute where it matters" philosophy to the medical domain.

---

#### When Does Divide and Conquer Work for Long Context?
**Paper:** [2506.16411](https://arxiv.org/abs/2506.16411) — ICLR 2026 | **Blog:** [together.ai/blog/plan-divide-conquer](https://www.together.ai/blog/plan-divide-conquer)

**Pillar:** Multi-Model Intelligence / Long Context

A theoretical framework that decomposes long-context LLM failures into three noise types: task noise (cross-chunk dependencies), model noise (confusion growing with length), and aggregator noise (imperfect combination). Explains why weaker models with chunking can outperform GPT-4o in a single shot.

Presented at ICLR 2026. The framework's Planner-Worker-Manager architecture lets Llama-3-70B match GPT-4o quality at lower cost. Provides principled guidance for choosing between single-pass and chunked strategies.

---

#### Shrinking the Generation-Verification Gap (Weaver)
**Paper:** [2506.18203](https://arxiv.org/abs/2506.18203)

**Pillar:** Scaling Reasoning Efficiently

Weaver ensembles multiple imperfect LM verifiers via weak supervision to approach oracle-level verification quality — without any labeled data. Llama 3.3 70B with Weaver reaches 87.7% on reasoning tasks, matching o3-mini (86.7%) and far exceeding GPT-4o (69.0%), purely at inference time.

The practical answer to "how do you verify reasoning without expensive finetuning?"

---

#### Data Diversification Methods in Alignment
**Paper:** [2507.02173](https://arxiv.org/abs/2507.02173)

**Pillar:** Scaling Reasoning Efficiently

Diversified-ThinkSolve (DTS) generates diverse reasoning paths for preference optimization, improving math performance by up to 7.1% on GSM8K at only 1.03x compute overhead — far cheaper than MCTS-based alternatives (5x cost).

The recurring theme: structured diversity beats brute-force compute.

---

#### Imitate Optimal Policy (Action Collapse)
**Paper:** [2509.02737](https://arxiv.org/abs/2509.02737)

**Pillar:** Scaling Reasoning Efficiently (RL Theory)

Discovers "Action Collapse" in policy gradient networks — a structured convergence analogous to neural collapse in classification. Fixing the action layer to a simplex ETF forces maximally separated action representations, yielding faster and more robust RL training.

A theoretical contribution that deepens understanding of how RL training dynamics work.

---

#### Staircase Streaming
**Paper:** [2510.05059](https://arxiv.org/abs/2510.05059)

**Pillar:** Multi-Model Intelligence

Multi-agent methods like MoA are powerful but slow — you wait for all intermediate agents before generating the final response. Staircase streaming begins the final response as soon as partial intermediate outputs arrive, reducing time-to-first-token by up to 93%.

The critical UX fix that makes multi-agent inference practical for interactive applications. Directly addresses MoA's main weakness.

---

#### Opportunistic Expert Activation
**Paper:** [2511.02237](https://arxiv.org/abs/2511.02237)

**Pillar:** Low-Level Optimization

MoE models are memory-bound during decode because many distinct experts must be loaded. This work dynamically re-routes tokens to piggyback on experts already loaded for other tokens in the same batch, reducing unique activations. 39% MoE layer latency reduction on Qwen3-30B with no accuracy loss and no retraining.

A clever systems-aware optimization that exploits batch structure.

---

#### Intelligence per Watt
**Paper:** [2511.07885](https://arxiv.org/abs/2511.07885)

**Pillar:** Systems / Efficiency Measurement

Proposes "intelligence per watt" (IPW) as a metric for local LLM inference efficiency. Key findings: local models can handle 88.7% of real-world queries, IPW improved 5.3x from 2023-2025, and local accelerators achieve at least 1.4x lower IPW than cloud. Makes the case that local inference is viable for most workloads.

---

#### Beat the Long Tail (DAS)
**Paper:** [2511.13841](https://arxiv.org/abs/2511.13841)

**Pillar:** Speculative Decoding

RL post-training is bottlenecked by rollout generation, where a long-tail of trajectory lengths dominates wall-clock time. DAS builds adaptive drafters from historical rollouts using suffix trees and allocates more aggressive speculation to long-tail trajectories. Reduces rollout time by up to 50% while preserving identical training curves.

The bridge between speculative decoding and RL training efficiency.

---

#### Kitty: 2-bit KV Cache Quantization
**Paper:** [2511.18643](https://arxiv.org/abs/2511.18643)

**Pillar:** Low-Level Optimization

Achieves near-lossless 2-bit KV cache quantization via algorithm-system co-design. Dynamic Channel-wise Precision Boost ranks Key-cache channels by sensitivity and keeps only a fraction at higher precision. Custom page layouts and Triton kernels handle the mixed precision efficiently.

Cuts KV memory by ~8x with negligible accuracy loss, enabling up to 8x larger batches and 2.1-4.1x higher throughput. The most aggressive KV quantization result to date.

---

#### CDLM: Consistency Diffusion Language Models
**Paper:** [2511.19269](https://arxiv.org/abs/2511.19269)

**Pillar:** Low-Level Optimization

Accelerates diffusion language models by 3.6-14.5x through consistency modeling (fewer sampling steps) and block-wise causal attention (enabling KV caching). An alternative generation paradigm to autoregressive decoding — parallel token generation with dramatically reduced step count.

---

#### ScaleSearch: Search Your NVFP4 Scales!
**Paper:** Under review at MLSys

**Pillar:** Low-Level Optimization

Standard Block Floating Point (BFP) quantization uses a fixed scale based on the maximum magnitude of each block, but this can be suboptimal. ScaleSearch exploits the mantissa bits in NVIDIA's NVFP4 microscaling format to search for the optimal block scale that minimizes quantization error for a given distribution. The method is architecture-agnostic and integrates into existing PTQ and low-precision attention pipelines.

ScaleSearch reduces quantization error by 26% on synthetic Gaussian data and delivers substantial real-world gains: up to 7.5 points improvement on GPQA for Qwen3-8B weight PTQ, 14 points improvement in VQA-a for video generation (Mochi) over SageAttention3, and 0.9 PPL improvement on Wikitext-2 for Llama 3.1 70B. The paper also introduces ScaleSearchAttention, an end-to-end NVFP4 attention mechanism that performs QK^T and PV matrix multiplications directly on Tensor Cores without dequantization, stores the KV cache in compact 4.5-bit format, and achieves near-zero accuracy degradation through incoherence processing, matrix decomposition, and attention-sink-aware mixed precision.

---

#### Understanding and Steering Reasoning (CREST)
**Paper:** [2512.24574](https://arxiv.org/abs/2512.24574)

**Pillar:** Scaling Reasoning Efficiently

CREST identifies specialized attention heads governing reasoning behaviors (verification, backtracking) and steers them at inference time. Improves accuracy by up to 17.5% while cutting token usage by 37.6% — no training required.

A training-free method to make reasoning models both better and faster by suppressing unproductive reasoning patterns.

---

#### ATLAS
**Blog:** [together.ai/blog/adaptive-learning-speculator-system-atlas](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas)

**Pillar:** Speculative Decoding

ATLAS pairs a heavyweight static speculator with a lightweight adaptive one that learns from real-time traffic, plus a confidence-aware controller. Eliminates manual tuning: performance automatically improves as the system specializes to workload characteristics.

Reached 501 TPS on DeepSeek-V3.1 and reduced RL-MATH training time by over 60%. The key stepping stone from static to fully dynamic speculative decoding.

---

#### How Together AI Uses AI Agents for Engineering
**Blog:** [together.ai/blog/ai-agents-to-automate-complex-engineering-tasks](https://www.together.ai/blog/ai-agents-to-automate-complex-engineering-tasks)

**Pillar:** Systems / Meta

Practical lessons on using LLM agents to automate multi-day engineering workflows like speculative decoding optimization. Six patterns for effective automation. Agents achieved 1.22-1.37x speedups with minimal human intervention.

A meta-level contribution: using AI to accelerate the development of AI inference systems.

---

### 2026

---

#### Aurora
**Paper:** [2602.06932](https://arxiv.org/abs/2602.06932) | **Blog:** [together.ai/blog/aurora](https://www.together.ai/blog/aurora)

**Pillar:** Speculative Decoding

The culmination of the speculative decoding arc. Aurora frames online speculator learning as an asynchronous RL problem: accepted tokens are positive reward, rejected tokens are negative feedback. The system integrates inference and training servers with hot-swapped speculator updates and zero service interruption.

Aurora supports day-0 deployment (serve immediately, improve on the fly), achieves 1.5x day-0 speedup on frontier models, and an additional 1.25x over well-trained static speculators when adapting to distribution shifts. This is the "self-driving" version of speculative decoding — the system that never stops getting faster.

---

#### ThunderAgent
**Paper:** [2602.13692](https://arxiv.org/abs/2602.13692) | **Website:** [thunderagent.ai](https://thunderagent.ai/)

**Pillar:** Systems (The Glue)

ThunderAgent introduces "LLM Programs" — a unified abstraction for co-scheduling LLM inference and tool execution across heterogeneous resources (KV caches, system states, disk, network ports). Its program-aware scheduler maximizes KV cache hit rates and manages memory, while a tool resource manager enables async environment preparation.

1.5-3.6x throughput improvements in serving, 1.8-3.9x in RL rollouts, 4.2x disk memory savings. This is the systems layer that ties together all the individual optimizations — the infrastructure for agentic AI at scale.

---

#### CARE: Converting GQA to MLA
**Paper:** [2603.17946](https://arxiv.org/abs/2603.17946) — ICLR 2026

**Pillar:** Low-Level Optimization

An activation-aware method for converting pretrained GQA models to multi-head latent attention (MLA) format. Three innovations: activation-preserving factorization, adjusted-rank allocation, and KV-parity mapping. Reduces perplexity by up to 215x over naive SVD baselines at matched KV-cache budgets.

With brief fine-tuning, fully recovers original accuracy — a practical pipeline for deploying MLA-based efficient inference on existing models.

---

#### V1: Unifying Generation and Self-Verification
**Paper:** [2603.04304](https://arxiv.org/abs/2603.04304)

**Pillar:** Scaling Reasoning Efficiently

Replaces pointwise scoring with pairwise self-verification for test-time scaling. V1-Infer uses uncertainty-guided tournament ranking; V1-PairRL jointly trains a single model as both generator and pairwise verifier.

Improves Pass@1 by up to 10% on code and math benchmarks. The insight that models are substantially better at pairwise comparisons than absolute scoring opens a new paradigm for parallel reasoners.

---

#### Squeeze Evolve
**Paper:** [2604.07725](https://arxiv.org/abs/2604.07725)

**Pillar:** Multi-Model Intelligence

Multi-model orchestration for verifier-free evolution. Allocates strong vs. cheap models based on marginal utility at each evolutionary stage. Cuts API costs by ~3x and boosts throughput by ~10x while matching or exceeding verifier-based methods.

The first verifier-free evolutionary method to match verifier-based methods on discovery tasks. Represents the maturation of the multi-model intelligence pillar — from showing it works (MoA) to making it economically practical.

---

#### Introspective Diffusion Language Models (I-DLM)
**Paper:** Under review at COLM 2026

**Pillar:** Low-Level Optimization (Generation Paradigm)

Diffusion language models (DLMs) promise parallel token generation but consistently lag behind autoregressive (AR) models in quality. This work argues the gap stems from a failure of *introspective consistency*: AR models agree with what they generate, whereas DLMs often do not. The paper formalizes this via the introspective acceptance rate and shows that causal masking plus logit shifting in AR models implicitly enforces this consistency. I-DLM preserves this property while retaining diffusion-style parallelism through a novel introspective strided decoding (ISD) algorithm, which verifies previously generated tokens while advancing new ones in the same forward pass.

I-DLM-8B is the first DLM to match the quality of its same-scale AR counterpart while surpassing all prior DLMs in both quality and serving efficiency across 15 benchmarks. It scores 72.5 on AIME-24 and 45.1 on LiveCodeBench-v6, outperforming LLaDA-2.1-mini (16B) by 29+ and 14+ points respectively. With single-pass self-speculative decoding and gated LoRA, ISD enables lossless acceleration and 3.1x higher throughput than AR on MATH-500. This represents a potential paradigm shift — diffusion models that are finally competitive with autoregressive for real deployment.

---

#### System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving
**Paper:** Under review at COLM 2026

**Pillar:** Low-Level Optimization (Quantization)

Many KV-cache compression methods improve offline accuracy or compression ratio but violate practical serving constraints — paged memory layouts, regular memory access patterns, and fused attention execution. This work identifies the minimal set of 4-bit KV-cache quantization methods that remain viable under real serving constraints, finding that a simple design — token-wise INT4 quantization with block-diagonal Hadamard rotation — consistently achieves the best accuracy-efficiency trade-off.

More complex methods like vector quantization and Hessian-aware quantization provide only marginal additional gains once serving compatibility is enforced. The paper implements a fused rotation-quantization kernel that integrates directly into paged KV-cache layouts with zero measurable end-to-end overhead, matching plain INT4 throughput across concurrency levels. The key message: effective KV-cache compression is fundamentally a systems co-design problem, and under real constraints, the simplest viable method wins.

---

## The Speculative Decoding Arc (A Story in 5 Chapters)

This is perhaps the most compelling narrative thread:

1. **Bifurcated Attention** (2024) — The foundation: make shared-prefix parallel decoding fast via split GEMMs.
2. **Customized Speculative Decoding** (May 2025) — Train domain-specific speculators on real customer traffic for 1.2-1.45x speedups.
3. **DAS** (Nov 2025) — Apply speculation to RL training rollouts, exploiting long-tail distributions. 50% rollout time reduction.
4. **ATLAS** (Oct 2025) — Runtime-learning speculators with confidence-aware control. 501 TPS on DeepSeek-V3.1.
5. **Aurora** (2026) — The full vision: RL-based continuous speculator learning from live traces. Day-0 deployment, zero-downtime updates, automatic domain adaptation.

## The Multi-Model Arc (A Story in 4 Chapters)

1. **MoA** (Jun 2024) — Layered multi-agent architecture beats GPT-4o with open-source models.
2. **MoAA** (May 2025) — Distill collective intelligence into small models. Self-improving loop.
3. **Staircase Streaming** (Oct 2025) — Make multi-agent pipelines 93% lower latency.
4. **Squeeze Evolve** (2026) — Economically optimal model allocation across evolutionary stages.

## The "Spend Compute Wisely" Arc

1. **Token Economies** (Jun 2024) — Complex strategies waste compute vs. simple baselines.
2. **Think Deep/Fast** (Apr 2025) — Majority voting is competitive; non-reasoning models can't catch up.
3. **CREST** (Dec 2025) — Steer reasoning heads to be efficient, no training needed.
4. **V1** (Mar 2026) — Pairwise verification beats pointwise; jointly train generator + verifier.
5. **Weaver** (Jun 2025) — Ensemble weak verifiers to approach oracle quality.
