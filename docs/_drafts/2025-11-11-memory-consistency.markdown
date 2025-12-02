---
layout: post
title:  "CUDA Memory Consistency Model"
# date:   2025-11-11 11:18:26 -0800
categories: CUDA
typora-root-url: ..
---



We explain these points in order. First, a `wgmma.fence` instruction ensures that `wgmma.mma_async` only accesses certain RMEM addresses after all prior accesses to such addresses have finished. Without the `wgmma.fence`, the behavior is undefined. An exception to this rule is that Hopper allows *multiple* `wgmma.mma_async` instructions to be in flight simultaneously. As long as these `wgmma.mma_async` instructions have the same accumulator shape, they can share the same accumulator tensor, i.e., write to the same register memory addresses. In that case, a fence is not required. For example, we don’t need to insert a `wgmma.fence` within the loop over `MMA_K` done as part of the `cute::gemm` call.



```cpp
cute::warpgroup_arrive();
cute::gemm(tiled_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
cute::warpgroup_commit_batch();
cute::warpgroup_wait<0>();
```





Below is a clear, accurate, up-to-date summary of the Blackwell (SM100) and Hopper (SM90) async-proxy memory model rules, based on the PTX ISA 8.x/9.x updates and NVIDIA’s memory model papers.

This is the definitive explanation of async-proxy fences, TMA, and how async engines interact with CUDA’s scoped memory model.

⸻

⚡️ Background: What is an async proxy?

Starting with Hopper (SM90) and continuing in Blackwell (SM100), some memory operations are executed by async engines that do not participate in normal GPU memory ordering rules.

This includes:

Hopper (SM90)
	•	TMA (Tensor Memory Accelerator): async global→shared copies
	•	GMMA: tensor core loads
	•	cp.async.bulk.tensor / tma.load
	•	cp.async.cg.shared.global.phys → global→shared async pipeline
	•	Async barriers (mbarrier) interacting with async copy engines

Blackwell (SM100)
	•	UMMA (Unified Multi-Modal Accelerator): async local/remote memory engines
	•	Async data movement within tensor memory space
	•	More async hardware paths that do not obey PTX standard .acq/.rel scopes unless fenced

These engines move data outside the normal memory consistency pipeline → hence “proxy.”

⸻

🧠 Why async-proxy rules exist

GPU cores have a memory model defined by:
	•	Scopes: .cta, .gpu, .sys
	•	Orders: .acq, .rel, .sc, .weak

But:
	•	Async engines (TMA/UMMA) do not obey these orders.
	•	They can read/write memory without ordering guarantees relative to normal ld/st, atom, etc.

Thus, NVIDIA introduced proxy-memory operations and fences.

⸻

🚧 Hopper & Blackwell: Async Proxy Memory Model Rules

1. Async operations are not ordered w.r.t. normal PTX loads/stores

Example:

tma.load ...
st.global ...

There is no guarantee that the TMA completes before or after the st.global.

To order them, you must use a proxy fence.

Equivalent for:
	•	cp.async.bulk.tensor
	•	mma.load (tensor core async)
	•	Blackwell UMMA load/store ops

⸻

2. The floor of ordering: fence.proxy and mbarrier.proxy

PTX introduces:

Hopper (SM90):

fence.proxy.async
fence.proxy.tensormem
mbarrier.proxy

Blackwell (SM100) adds new proxy types:

fence.proxy.umba       // UMMA engines
fence.proxy.tma        // Tensor Memory Accelerator
fence.proxy.cluster
... and combinations

These fences establish happens-before edges between async engines and core memory model.

Meaning:

st.global.relaxed
fence.proxy.async
... async engine sees the writes ...

Without this fence: async engines may observe stale data.

⸻

3. All async storage targets have separate proxy domains

Hopper domains:
	•	async
	•	tensormem
	•	generic

Blackwell extends this:
	•	umma (new async memory engines)
	•	tma (old TMA becomes its own domain)
	•	cluster-proxy (for cluster shared memory)
	•	tensorcore-proxy (GMMA pathways)

Comparing domains is key:
	•	Operations in different proxy domains are not ordered unless fenced.

⸻

🔄 Example: Hopper TMA sync with shared memory

tma.load.tensor.shared::cta.desc ...
mbarrier.arrive.expect_tx ...
mbarrier.wait ...
fence.proxy.tensormem
ld.shared ...

Explanation:
	1.	tma.load kicks off async copy into shared memory.
	2.	mbarrier.wait ensures copy completed, but does not order memory wrt. SM core.
	3.	fence.proxy.tensormem is required to make SM core see newly-written data.

Without fencing: SM may read stale or partially-written shared memory.

⸻

🧱 Hopper/Blackwell Async Memory Model Rules (Textbook summary)

Rule A — Async engines do NOT participate in CUDA’s memory model.

They are “outside” the .acq/.rel/.sc ordering rules.

Rule B — Async-copy writes become visible only after an appropriate proxy fence.

Rule C — Barriers (mbarrier) handle completion but NOT memory visibility ordering.

Completion is distinct from visibility.

Rule D — Proxy domain fences must match the async operation type.

Wrong domain → no synchronization.

Rule E — Cross-proxy ordering requires two fences:
	•	One releases from producer domain
	•	One acquires in consumer domain

⸻

📦 Blackwell-specific updates (beyond Hopper)

Blackwell (PTX ISA 9.x) introduces:

1. UMMA async instructions
	•	They operate in their own proxy domain (proxy.umma)
	•	Require fence.proxy.umma for visibility ordering

2. Cluster Tensor Memory (CTM) domain

For distributed shared memory across SMs in a cluster.

3. More explicit proxy categories

E.g., Blackwell distinguishes:
	•	proxy.tma
	•	proxy.umba
	•	proxy.cluster
	•	proxy.tensormem

Hopper treated TMA/TensorCore more implicitly.

4. Proxy+scope combined fences

New forms like:

fence.sc.proxy.tma.gpu
fence.acq_rel.proxy.umma.cta

These create explicit HB edges between async pathways and normal memory across scopes.

⸻

🧩 Simplified mental model

Normal CUDA memory model = CPU-like acquire/release + scopes.
Async engines = DMA engines / tensor loaders outside the model.
Proxy domains = isolated worlds that must be connected with fences.

normal core ops
   |
   |  (barrier / mbarrier)
   |
async engine ops

Fences wire them together.

⸻

✨ Want diagrams or examples?

I can provide:

✅ diagrams showing proxy domains
✅ litmus tests demonstrating incorrect behavior without proxy fences
✅ a Blackwell-to-Hopper comparison table
✅ examples from TMA, UMMA, and GMMA pipelines
✅ CUTLASS pipelines explained with proxy rules

Tell me what you’d like!    `Sadfsda`



## References

- https://github.com/NVlabs/mixedproxy
- [GPU Concurrency: Weak Behaviours and Programming Assumptions](https://dl.acm.org/doi/10.1145/2775054.2694391)
- A Formal Analysis of the NVIDIA PTX Memory Consistency Model
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-synchronization-domains
- [PTX Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model)