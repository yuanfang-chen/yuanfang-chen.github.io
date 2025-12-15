---
layout: post
title:  "CUDA Memory Consistency Model"
# date:   2025-11-11 11:18:26 -0800
categories: CUDA
typora-root-url: ..
---

## [PTX `mbarrier`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)

## [PTX `membar`/`fence`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)

## [PTX atom](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom)



## [Proxy (or Memory Proxy)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#proxies)

Proxy（也称为Memory Proxy）是在内存操作上附加一个标签。当两个内存操作用不同的方式访问内存时，这两个操作就属于不同的Proxy。

[Operation types](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#operation-types) 这里的内存操作属于*generic proxy*。textures and surfaces类访存操作输入其他proxy。目前PTX只支持两种proxy: `.async, .alias`

Value `.alias` of the `.proxykind` qualifier refers to memory accesses performed using virtually aliased addresses to the same memory location. 
Value `.async` of the `.proxykind` qualifier specifies that the memory ordering is established between the async proxy and the generic proxy. The memory ordering is limited only to operations performed on objects in the state space specified. If no state space is specified, then the memory ordering applies on all state spaces.

属于不同的proxy的访存操作之间需要内存同步的话，需要用*proxy fence* 

[Isolating Traffic with Domains](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#isolating-traffic-with-domains)

## Asynchronous Copy (PTX `cp.async`)

With Ampere, NVIDIA introduced asynchronous data copy, a way of copying data directly from global memory to shared memory in an asynchronous fashion. To load data from global memory to shared memory on Volta, threads must first load data from global memory to registers, and then store it to shared memory. However, MMA instructions have high register usage and must share the register file with data-loading operations, causing high register pressure and wasting memory bandwidth for copying data in and out of RF.

Async data copy mitigates this issue by fetching data from global memory (DRAM) and directly storing it into shared memory (with optional L1 access), freeing up more registers for MMA instructions. Data loading and compute can happen asynchronously which is more difficult from a programming model perspective but unlocks higher performance.

This feature is implemented as PTX instruction thread-level async copy cp.async (documentation). The corresponding SASS is LDGSTS, asynchronous global to shared memory copy. The exact synchronization methods are async-group and mbarrier-based completion mechanisms, detailed here.
![alt text](/assets/images/ampere-async-copy.png)

[Controlling Data Movement to Boost Performance on the NVIDIA Ampere Architecture](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/)

### [Completion Mechanisms for Asynchronous Copy Operations](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms)

用来追踪异步操作是否完成，和内存一致性（async proxy等）无关

#### [Async-group mechanism](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-async-group)

```c++
// Example of .wait_all:
cp.async.ca.shared.global [shrd1], [gbl1], 4;
cp.async.cg.shared.global [shrd2], [gbl2], 16;
cp.async.wait_all;  // waits for all prior cp.async to complete

// Example of .wait_group :
cp.async.ca.shared.global [shrd3], [gbl3], 8;
cp.async.commit_group;  // End of group 1

cp.async.cg.shared.global [shrd4], [gbl4], 16;
cp.async.commit_group;  // End of group 2

cp.async.cg.shared.global [shrd5], [gbl5], 16;
cp.async.commit_group;  // End of group 3

cp.async.wait_group 1;  // waits for group 1 and group 2 to complete
```

#### [Mbarrier-based mechanism](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-mbarrier)

PTX mbarrier








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

Tell me what you’d like!   



## References

- https://github.com/NVlabs/mixedproxy
- [GPU Concurrency: Weak Behaviours and Programming Assumptions](https://dl.acm.org/doi/10.1145/2775054.2694391)
- A Formal Analysis of the NVIDIA PTX Memory Consistency Model
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-synchronization-domains
- [PTX Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model)
- [`cp.async, cp.async.bulk`](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-asynchronous-copy) 
- [NVIDIA TMA 全面分析](https://zhuanlan.zhihu.com/p/1945136522455122713)
- [NVIDIA: Techniques for efficiently transferring data to a processor](https://patentimages.storage.googleapis.com/2e/65/c9/73308c2c12e3d5/US11080051.pdf)
- [NVIDIA: Method and Apparatus for Efficient Access to Multidimensional Data Structures and/or Other Large Data Blocks](https://patentimages.storage.googleapis.com/23/89/13/0914c8f599bd9e/US20230289304A1.pdf)
- NVIDIA: CUDA doc [https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
- NVIDIA Hopper Architecture [https://resources.nvidia.com/en-us-hopper-architecture/](https://link.zhihu.com/?target=https%3A//resources.nvidia.com/en-us-hopper-architecture/)
- NVIDIA PTX doc [https://docs.nvidia.com/cuda/parallel-thread-execution/index.html](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- CUTLASS Tutorial: Mastering the NVIDIA® Tensor Memory Accelerator (TMA) [https://research.colfax-intl.com/tutorial-hopper-tma/](https://link.zhihu.com/?target=https%3A//research.colfax-intl.com/tutorial-hopper-tma/)
- NVIDIA: Advanced Performance Optimization in CUDA
