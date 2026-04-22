## Highlighted Work

My research focuses on inference efficiency, ranging from lower-level optimizatin such as bifurcated attention for parallel sampling, quantization, KV compression, to higher level optimization such as routing and multi-model agent orchestration.

Below are some of the work that are highly impactful and widely adopted for real-world LLM efficiency.


### Bifurcated Attention: Accelerating Massively Parallel Decoding with Shared Prefixes in LLMs
Crucial for parallel sampling (reduce KV cache memory as well as improving loading time). Highly applicable for RL rollout where we want to massively parallelizing sampling to obtain thousands of possible answers, in order to use these answers for verification and improve reward signaling.



### System Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving 
Investigations into quality degradation and proposing approaches to reduce KV quantization errors for frontier language models. Highly applicable for real-world LLM serving with improved throughput via higher batch size.


### Search Your NVFP4 Scales!
A proposal of ways to reduce quality gap between NVFP4 vs FP8/FP16 model weight quantization by smart searching of quantization scales. Another high-impact work applicable for frontier model efficiency.



### Introspective Diffusion Language Models
Adapting autoregressive models to be more diffusion-like, including a lossless version where we use speculative decoding framework to perform rejection sampling, while using a novel algorithm to perform both verification and generation at the same time to obtain higher inference efficiency.


### ATLAS and Aurora
Keep speculative decoding models up to date with the latest data distributions. Very important for LLM workload with shifting traffic patterns, including RL rollout where the policy gets updated quite often which requires co-evolving draft models.



### Squeeze Evolve: Unified Multi-Model Orchestration for Verifier-Free Evolution
A major progress in evolution algorithms to (1) improve performances and reduce stagnation via multi-model usage and (2) cost optimization due to smarter choices of model selection.




### Beat the Long Tail: Distribution-Aware Speculative Decoding for RL Training
Improving efficiency of RL rollout in the scenario of synchronous RL by allocating for speculative decoding budget during low-batch scenario.



### Mixture of Agents
One of the pioneering work that shows the potential of a multi-agent system as a way to further augment capabilities.



### ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System
A smart routing mechinism to reduce overhead for agentic scaffolding via program-aware scheduling. Helps increse cache hit rate and overall throughput.


### Ladder Residual
An alternative architecture that modifies residual streams to be non sequential, allowing overlaping of communciation and computation and reduce inference latency due to tensor parallel.


## Publications
