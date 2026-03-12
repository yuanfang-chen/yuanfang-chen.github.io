---
layout: post
title:  "PyTorch2总结"
# date:   2025-12-08 11:18:26 -0800
categories: ML
typora-root-url: ..
mathjax: true
---

## PyTorch QA 体系全景

### 1. CI/CD Infrastructure

PyTorch runs one of the largest open-source CI systems in existence, built on GitHub Actions with self-hosted runners. The `pytorch/test-infra` repository hosts the supporting infrastructure, including logic for tracking disabled and slow tests, as well as the CI jobs HUD/dashboard.

The CI runs on every pull request and includes a huge matrix of configurations: multiple Python versions, multiple CUDA versions (and ROCm for AMD), CPU-only builds, different operating systems (Linux, macOS, Windows), and different compilers. In 2025, the PyTorch Foundation committed to fully open-sourcing the CI infrastructure and extending it to support multiple compute providers. There's even a dedicated multi-cloud CI working group led by IBM's Andrea Frittoli.

The HUD (Heads-Up Display) at `hud.pytorch.org` is the central dashboard for monitoring CI health. It provides views into benchmarks, metrics, flaky test tracking, disabled tests, cost analysis, queue time analysis, nightly dashboards, and failure classification.

**Dr. CI** is a bot that runs on the test-infra side. It periodically updates comments on PRs with CI status summaries, and the workflow backfills job data into DynamoDB and ClickHouse for analytics. Dr. CI helps developers quickly understand whether CI failures on their PR are their fault or pre-existing flakes.

------

### 2. The Test Suite Itself

The PyTorch test suite is enormous — thousands of test files under `test/` in the main repo. Tests are organized by module: `test_autograd.py`, `test_cuda.py`, `test_nn.py`, `test_torch.py`, `test_jit.py`, `test_inductor.py`, distributed tests, export tests, and many more. The tests cover:

**Correctness testing** — verifying that operators produce numerically correct results across dtypes (float32, float16, bfloat16, int variants), devices (CPU, CUDA, MPS), and configurations (eager, scripted, compiled). This includes gradient checking via `torch.autograd.gradcheck`.

**Operator-level testing** — PyTorch has an `OpInfo` system where each operator has metadata describing its valid inputs, expected behaviors, supported dtypes, and known issues. This allows writing generic test templates that run across *all* registered operators, rather than writing per-operator tests. This is critical for catching regressions when ~2000+ operators are involved.

**BC (backward compatibility) lint** — you'll see this as a CI workflow called "BC Lint" on PRs. PyTorch has a formal policy where deprecated behavior must throw a warning for at least two releases and 180 days of nightlies before removal. The BC lint checks enforce that public API signatures haven't changed in backward-incompatible ways.

**Linting** — PyTorch uses `lintrunner` for code quality checks, including clang-tidy for C++, flake8/ruff for Python, and custom linters for things like ensuring proper type annotations, checking for banned patterns, etc.

------

### 3. Performance Regression Testing (TorchBench)

This is the piece you've explored before. TorchBench is integrated into PyTorch's CI to perform daily sanity checks on performance regressions for every nightly release. It has identified multiple commits that caused unexpected slowdowns, with five reverted and two merged with optimization fixes.

The regression detection mechanism works as follows: A threshold of 7% in execution time or memory usage triggers an alert. If at least one TorchBench benchmark exceeds this, CI automatically files a GitHub issue with a detailed performance report and the problematic commit. To reduce overhead from the 70+ daily commits, CI only checks the nightly build, then uses binary search to bisect the offending commit if a regression is found.

TorchBench measures execution time and GPU memory, and can run in both CPU-only and CPU+GPU mode for training and inference scenarios. It also supports PR-level benchmarking — developers can add a magic keyword "RUN_TORCHBENCH" in their PR to trigger benchmarks and visualize results on the HUD.

------

### 4. The Dynamo/Inductor Dashboard (the 163+ Model Suite)

This is the system under `benchmarks/dynamo/` that you explored. For validating `torch.compile`, PyTorch used a set of 163 open-source models across HuggingFace, TIMM, and TorchBench, covering image classification, object detection, image generation, NLP, recommendation systems, and reinforcement learning.

Three benchmark runner scripts (`torchbench.py`, `huggingface.py`, `timm_models.py`) accept flags for `--accuracy` or `--performance`, `--training` or `--inference`, and various device/precision configurations. The dashboard measures:

Three key metrics: geometric mean speedup (vs eager), mean compilation time, and peak memory footprint compression ratio. Multiple settings of TorchInductor are tested, including default, with cudagraphs, and with dynamic shapes.

Individual dashboard runs can be triggered manually via a "Run workflow" button, selecting a PR's branch — though this is expensive and the team asks that it be used wisely. Daily results are published to the TorchInductor Performance Dashboard, which now covers both GPU (A100) and CPU.

------

### 5. Flaky Test Management

At the scale of PyTorch's CI, flaky tests are a major QA concern. There's an elaborate automated system for managing them:

The flaky test detection system automatically files GitHub issues for tests detected as flaky, with a dedicated "Disable Flaky Tests" workflow that runs on the test-infra repo. When a test is found to be flaky (e.g., failing some percentage of runs within a 6-hour window), the bot:

1. Files a GitHub issue with the tag `module: flaky-tests` and `skipped`
2. Automatically disables the test in CI so it doesn't block developers
3. CI shields developers from flaky tests by showing them as green, though this makes log parsing harder — the system warns "DO NOT ASSUME THINGS ARE OKAY IF THE CI IS GREEN."
4. All disabled tests are tracked at `hud.pytorch.org/disabled`, and relevant oncall teams are auto-CC'd on the issue.

The system also periodically re-runs disabled tests to check if they've been fixed, and auto-closes issues for tests that no longer exist in the codebase.

------

### 6. Release Process

The release process has two phases: Phase 1 for feature/fix cutoff and branch creation, and Phase 2 for extended integration, stability, and performance testing based on Release Candidate builds.

Cherry-pick criteria for the release branch are strict: only fixes for regressions against the most recent minor release, critical fixes for silent correctness / backward compatibility / crashes / deadlocks / memory leaks, and compilation fixes for third-party library compatibility. Anything else requires special dispensation from the release managers.

The RC process normally takes about 2 weeks, with an announcement on the official channel about RC availability. Patch releases have even stricter criteria and require all cherry-picks to link to high-priority GitHub issues or CI failures.

Nightly builds are a critical part of QA. Every night, the latest commit on `main` is built into wheels and conda packages for all supported platform combinations. Users of `pytorch-nightly` serve as an additional early-warning system for regressions.

------

### 7. Backward Compatibility Policy

PyTorch categorizes components as "prototype," "beta," or "stable." BC-breaking changes to prototype and beta components can be made without warning, but stable APIs follow a strict deprecation cycle. The policy requires:

Deprecated behavior must throw `TORCH_WARN_ONCE` warnings describing the deprecation, when it takes effect, and a stable alternative. Warnings must appear for two releases and 180 days of nightlies. If the new behavior would cause a "silent breakage," the old behavior must throw a runtime error for an additional two releases and 180 days.

------

### 8. Additional QA Mechanisms

A few other pieces worth mentioning:

**Oncall rotation** — PyTorch modules have designated oncall teams (visible in issue tags like `oncall: pt2`, `oncall: distributed`, `oncall: cpu inductor`). When CI detects issues in a module, the appropriate oncall gets auto-CC'd.

**`pytorchbot`** — a bot that handles merge queue management, cherry-picks, auto-labeling, and reviewer assignment. It enforces that PRs pass required CI checks before merging.

**Test sharding and parallelization** — the test suite is sharded across many machines. The `--total-partitions` and `--partition-id` flags in the benchmark scripts reflect this. Tests are also prioritized so that historically-relevant tests for a given code change run first.

**CUDA memory leak detection** — specific tests check for GPU memory leaks, and these are themselves tested for correctness (as you can see from the disabled test issues about `test_cuda_memory_leak_detection`).

**Multi-accelerator testing** — beyond NVIDIA GPUs, PyTorch CI tests against AMD ROCm, Intel XPU, and Apple MPS. Third-party accelerator teams like IBM Spyre are building layered test pyramids covering op-level correctness, inductor compilation, module-level tests, model quality, and end-to-end inference, targeting 95%+ pass rates on nightly runs.

The overall picture is one of defense in depth: unit tests at the operator level, integration tests at the model level, performance regression testing at the benchmark level, flaky test automation for reliability, and a strict release process with RC testing cycles. It's an enormous investment — PyTorch is one of the few open-source projects where the CI infrastructure itself has a dedicated engineering team and its own conference keynote.



## **CI资源提供商**

AWS、GCP、Azure、IBM，以及加速器供应商的外部 CI

---------

The "compute providers" here refers specifically to cloud infrastructure providers that host the CI runners — not end-user cloud platforms for training models. Here's what I could piece together:

**Historically**, PyTorch CI ran almost entirely on AWS. Meta funded and managed the infrastructure, using AWS EC2 instances (including GPU instances like A100s for CUDA testing). Recent TAC meeting minutes mention that several build workflows have been migrated from C5 instances to C7i, and that the Cloudflare CDN migration has started. NVIDIA and AWS specifically joined a recent infrastructure call and are looking to add x86-Windows-CUDA tests.

**The migration and multi-cloud push**: The CI Working Group completed the migration of the PyTorch CI pipeline from Meta to the Linux Foundation in September 2024, and has since shifted focus to enhancing and managing it. The Multi-cloud Working Group, led by IBM's Andrea Frittoli, is developing solutions to expand CI into multiple cloud technologies.

**The founding members** who sit on the governing board and drive this are: AMD, AWS, Google Cloud, Meta, Microsoft Azure, and NVIDIA. IBM joined as a premier member in 2023, and Frittoli (IBM) leads the multi-cloud WG directly.

So the specific cloud providers being targeted are primarily **AWS** (the incumbent), **Google Cloud**, and **Microsoft Azure**, with **IBM** actively driving the multi-cloud architecture. The TAC minutes don't enumerate every provider explicitly, but the Multi-cloud WG is evaluating options for fleet management and has defined runner requirements — suggesting they're still in the process of onboarding additional providers beyond AWS.

The "multiple compute providers" language is also about **accelerator vendors** contributing their own CI — for example, IBM is building out-of-tree CI for their Spyre accelerator as a contribution to the PyTorch ecosystem, targeting 95%+ nightly pass rates. Similarly, AMD runs ROCm CI, Intel runs XPU CI, and Qualcomm contributes edge-device testing for ExecuTorch. The TAC's vision is to create a standardized way for these external CI systems to integrate with the main PyTorch CI, rather than everyone running completely siloed infrastructure.



## In-tree vs out-of-tree 加速器的 CI 流程

in-tree： CUDA/ROCm/XPU 
out-of-tree：PrivateUse1/OpenReg

----------

### 1. In-Tree Accelerators (CUDA, ROCm, XPU, MPS)

PyTorch currently has these in-tree devices: CPU, CUDA, META, MPS, and XPU. ROCm (AMD) reuses the CUDA dispatch key via HIP translation, so it's effectively in-tree too.

**How their CI works:**

These backends run directly in the main `pytorch/pytorch` CI on GitHub Actions. Their tests are part of the main test matrix — when you open a PR, the CI matrix includes jobs like `linux-focal-cuda12.1-py3.11`, `linux-focal-rocm6.2`, `linux-focal-xpu`, etc. Each job builds PyTorch with that backend enabled, then runs the relevant test suite against it.

The key characteristics:

**CUDA (NVIDIA)** is the most deeply integrated. NVIDIA provides GPU runners (A100s) to the PyTorch CI. CUDA tests run on every PR as blocking checks. The full Dynamo/Inductor performance dashboard runs nightly on A100s. NVIDIA also joined recent infrastructure calls and is looking to add x86-Windows-CUDA tests.

**ROCm (AMD)** has progressively expanded its in-tree CI presence. AMD has invested significantly in expanding CI coverage across multiple GPU generations — MI200 series (MI210X, MI250X), MI300 series (MI300X, MI325X), and MI350 series (MI350X, MI355X). ROCm tests show up in the main CI as labeled workflows. AMD also maintains an internal dashboard that tests pre-release ROCm builds and updated Triton compilers as part of their broader AISWHUD initiative. So there's a two-tier approach: some ROCm jobs run in the main PyTorch CI, and AMD runs additional internal CI on pre-release hardware/software that isn't yet public.

**XPU (Intel)** was added as an in-tree device starting in PyTorch 2.4. It supports both eager mode and `torch.compile`, with FP32, BF16, FP16, and AMP all supported. Intel contributes XPU CI runners to the main test matrix. Intel also runs CPU Inductor benchmarks and files regression issues (as you can see from the `oncall: cpu inductor` tag on issues).

**MPS (Apple)** runs on macOS runners in CI, covering Apple Silicon (M-series) GPU testing.

For all in-tree backends, the procedure is: the vendor provides the hardware runners (or cloud credits for them), the CI workflows are defined in `.github/workflows/` in the main PyTorch repo, and the tests run against the standard `test/` suite with device-specific skips/decorators. The HUD at `hud.pytorch.org` shows all of these jobs in a unified view.

------

### 2. Out-of-Tree Accelerators (PrivateUse1 / OpenReg path)

This is the more interesting and rapidly evolving story. Since PyTorch 2.1, the community has made significant progress in streamlining out-of-tree accelerator integration through refinements to the PrivateUse1 dispatch key, core subsystem extension mechanisms, and device-agnostic refactoring of key modules like `torch.accelerator`.

**The integration mechanism: PrivateUse1**

PrivateUse1 is a customizable device dispatch key, similar to CUDA/CPU/XPU, reserved for out-of-tree devices. It lets developers extend PyTorch's functionality without modifying the core framework. Vendors implement operator dispatch, memory management, stream/event handling, RNG, profiler hooks, and AMP support — all registering against the PrivateUse1 key.

**OpenReg: the reference implementation and CI safety net**

OpenReg is a self-contained CPU-based accelerator simulator designed to facilitate integration via PrivateUse1. It's not meant to be a real backend — it's a minimalist reference implementation for mechanism verification.

OpenReg serves two critical QA roles:

First, it's an in-tree test backend for PrivateUse1, ensuring quality stability through CI/CD. It also acts as a reference example for new backend integration, following a "minimality principle" (just enough to verify all integration paths) and an "authenticity principle" (integrating the same way a real accelerator would).

Second, it's been added to the PyTorch test suite so that whenever code is updated that touches PrivateUse1 paths, device-generic test cases execute automatically, catching issues that might break out-of-tree backends — even if the out-of-tree vendors don't have runners in the main CI.

**The Accelerator Integration Working Group's CI infrastructure**

The `pytorch-fdn/accelerator-integration-wg` repository on GitHub contains workflows and scripts that automate testing for integrating different hardware devices with PyTorch. It provides reusable workflows that vendors can adopt.

The WG's roadmap includes: building quality validations and standards for out-of-tree accelerators, CI infra refactoring to improve extensibility, and establishing quality standards that will eventually become "Compute Platform Quality Criteria." The TAC minutes you saw earlier reference vendors being asked to review these criteria.

**How an out-of-tree vendor actually runs CI**

The typical pattern is a **layered test pyramid** run on the vendor's own infrastructure:

For example, IBM's Spyre builds out-of-tree CI that validates a full stack: op-level correctness, inductor compilation and lowering, module-level tests (attention, normalization, activations), model quality/performance, and end-to-end vLLM inference. Their target is 95%+ pass rate on nightly runs, with the full pipeline completing in under three hours.

IBM explicitly builds this as an out-of-tree CI contribution to the PyTorch ecosystem, establishing patterns that other accelerator teams can adopt.

The practical flow for an out-of-tree vendor is:

1. Implement the PrivateUse1 hooks following OpenReg as a template (runtime, operators, fallback, AMP, profiler, etc.)
2. Register a CPU fallback mechanism so unsupported operators automatically fall back to CPU for execution, with results passed back to the new backend afterward — this is critical for early-stage development where not all ops are implemented.
3. Leverage the `pytorch-integration-tests` repository, which automates testing of PyTorch's device-specific functionalities across various hardware platforms including NPUs and other specialized devices, using GitHub Actions.
4. Run the Dynamo benchmark suites (`torchbench.py`, `huggingface.py`, `timm_models.py`) against your device to measure compilation success rate and performance.
5. Optionally integrate with the WG's reusable CI workflows from `pytorch-fdn/accelerator-integration-wg`.

**Key difference from in-tree**: out-of-tree vendors run their own CI on their own infrastructure. They don't get jobs in the main `pytorch/pytorch` GitHub Actions. Instead, they monitor the main repo's nightly builds and run their test pyramid against those nightlies. Regression failures are flagged within hours of the nightly build. The TAC's 2025 vision is to provide standardized ways for these external CI systems to report results back into the main HUD and to trigger alerts when upstream changes break out-of-tree backends — but that infrastructure is still being built.

**Notable out-of-tree examples**: Huawei Ascend NPU (which was one of the earliest PrivateUse1 adopters and heavily drove the mechanism's improvements), Intel Gaudi, IBM Spyre, and various other custom silicon vendors participating in the Accelerator Integration WG.

The overall trajectory is toward a world where vendors maintain full control over their accelerator integration timelines, enabling fast iteration cycles and reducing reliance on upstream coordination, while OpenReg protects them from upstream breakage by acting as the canary in the coal mine within the main CI.



## Out-of-tree CI/CD 的实际效果与局限性

检测延迟、OpenReg 的覆盖盲区、真实破坏案例、结构性差距、改进方向

-------
Great question — and the honest answer is **no, it's not the same experience, and there are significant gaps**. Let me search for the concrete details on where things stand.This is the right question to ask, and the honest answer is: **no, the out-of-tree CI/CD experience is fundamentally worse than in-tree, and there are structural reasons why it probably can't be fully equivalent.** Let me break down exactly where the gaps are.

---

### The core asymmetry: when breakage is detected

For **in-tree** accelerators (CUDA, ROCm, XPU), if someone's PR breaks something, the CI catches it *before the code merges*. The developer sees red on their PR, Dr. CI comments with the failure, and the merge is blocked. The feedback loop is measured in hours.

For **out-of-tree** accelerators, the best-case scenario is that the vendor discovers breakage when they run their own nightly CI against the next PyTorch nightly build — meaning the offending code has *already been merged into main*. The feedback loop is measured in **days**, and the fix now requires filing an issue, convincing a maintainer to revert or patch, and waiting for another nightly. In the worst case, vendors discover breakage weeks later when they try to upgrade to a new PyTorch release.

This is the fundamental gap, and no amount of tooling fully closes it.

---

### What OpenReg actually catches vs. what it doesn't

OpenReg is a real improvement, but it's important to be precise about its scope. The PyTorch team explicitly acknowledges that unit tests alone are not sufficient to prevent new pull requests from inadvertently affecting out-of-tree device compatibility. OpenReg was added to mitigate this, but here's what it actually covers:

**What OpenReg catches:** breakage to the PrivateUse1 *mechanism itself* — the dispatch key plumbing, the hooks interface, the device guard system, the generator registration, the autoload entrypoint machinery. If someone refactors `c10::DeviceType` or changes how `PrivateUse1HooksInterface` works, OpenReg's tests in the main CI will turn red. This is genuinely valuable.

**What OpenReg does NOT catch:**

First, **operator-level breakage**. OpenReg is a CPU-based simulator following a "minimality principle." It implements the bare minimum operators to verify integration paths. If someone changes the semantics of `aten::conv2d` or modifies how autograd handles certain backward paths on real hardware, OpenReg won't detect it because it's exercising a CPU fallback, not real accelerator kernels.

Second, **performance regressions**. OpenReg has no real hardware, so there's no way to detect that a change to TorchInductor generated slower Triton code or that a memory allocator change caused fragmentation on a real device. In-tree accelerators get TorchBench and the Dynamo dashboard for this. Out-of-tree gets nothing from upstream.

Third, **subtle device-specific interactions**. Things like stream synchronization ordering, async memory copy timing, multi-device peer-to-peer access, and distributed collectives over vendor-specific communication fabrics — these are impossible to test without real hardware. OpenReg simulates device isolation via subprocesses with request/response queues, which is architecturally completely different from how actual accelerator runtimes work.

Fourth, **cross-module integration breakage**. The early PrivateUse1 mechanism lacked support for several modules like Storage, AMP, Distributed, and others. Even though the mechanism has been enhanced since PyTorch 2.1, the coverage for these higher-level features in OpenReg is still evolving. Most PyTorch unit tests focus on CPU and CUDA devices, which limits participation from other hardware. The effort to generalize test decorators and remove CUDA-hardcoded assumptions is ongoing but far from complete.

---

### Real-world breakage: what actually happens to out-of-tree vendors

The GitHub issues tell the story concretely:

AWS's Neuron team (for Inferentia/Trainium) reported that as of PyTorch 2.4, their PrivateUse1 backend registration that had worked from torch 1.13 through 2.3 suddenly broke with `ModuleNotFoundError: No module named 'torch.privateuseone'`. This is a backward-incompatible change to the PrivateUse1 mechanism itself — the kind of thing OpenReg *should* catch, but apparently the test coverage wasn't sufficient at that point to prevent it from shipping.

Another user found that saving a model trained on an out-of-tree backend and loading it on CPU/GPU fails because `torch.load` tries to parse the custom device type, requiring the device plugin to be installed even on platforms where it's irrelevant. The suggested workarounds — calling `torch.utils.rename_privateuse1_backend()` before loading, or never saving `torch.Device` objects — are awkward hacks that reflect incomplete integration.

A developer trying to run an OpenCL backend on PyTorch 2.4/nightly found that upstream changes (like making an Allocator method non-const) broke compilation with no documentation of what changed or why. This is the classic out-of-tree problem: the C++ ABI surface that PrivateUse1 backends depend on is not formally stabilized, so any internal refactor can break downstream builds.

Even seemingly innocent changes cause problems — a module-level `torch.accelerator.is_available()` check at import time broke out-of-tree backends because the device plugin hadn't been registered yet at that point in the import sequence.

---

### The structural problems that can't be solved by tooling alone

There are several fundamental reasons why out-of-tree CI can never fully match in-tree:

**No pre-merge signal.** The main PyTorch CI cannot run tests on hardware it doesn't have. Unless every accelerator vendor donates runners to the main CI (which doesn't scale — there are dozens of vendors), there's no way to get a "does this PR break Huawei Ascend NPU?" signal before merge.

**The BC policy doesn't cover internal C++ APIs.** PyTorch's BC policy guarantees deprecation warnings for two releases and 180 days of nightlies for *public stable* APIs. But PrivateUse1 backends depend heavily on *internal* C++ interfaces — `at::PrivateUse1HooksInterface`, `c10::DispatchKey`, allocator interfaces, profiler stubs, etc. These internal interfaces can and do change without the formal deprecation process. The Accelerator Integration WG identified "Version Compatibility: BC-breaks not aware for accelerators in ecosystem" as one of the key challenges.

**Only one PrivateUse1 slot.** Only one out-of-tree device can occupy PrivateUse1 at a time — PrivateUse2 and PrivateUse3 haven't received the same treatment and are much less functional. This means you can't even test two out-of-tree backends in the same PyTorch environment simultaneously, which limits integration testing.

**Test infrastructure is CUDA-centric.** Most PyTorch unit tests focus on CPU and CUDA, with hard-coded device restrictions and CUDA-specific decorators. The plan to modify the testing framework for better non-CUDA support is underway but not complete. An out-of-tree vendor who wants to run PyTorch's standard test suite against their device has to wade through thousands of `@skipIfNoCUDA` and `@onlyCUDA` decorators, figure out which tests are relevant, and deal with tests that implicitly assume CUDA semantics (like specific floating-point behavior or memory ordering).

---

### What the community is actively building to close the gap

To be fair, the situation is *improving rapidly*. The Accelerator Integration WG is tackling this head-on:

Their roadmap includes building quality validations and standards for out-of-tree accelerators, CI infra refactoring, and an upstream repo (`accelerator-integration-wg`) with reusable CI workflows.

As of late 2025, testing guidance was evolving to include two categories: OpInfo-based tests for verifying operator implementation, and integration-level tests for device-specific paths.

The `pytorch-integration-tests` repository aims to be a fully automated system that continuously validates PyTorch's compatibility with different hardware backends, running automated tests across different devices using GitHub Actions.

And the newer OpenReg documentation is genuinely excellent — the four-axis guide (Runtime, Operators, Python Frontend, Higher-Level Features) with paired code examples is far better than what existed even a year ago.

---

### Bottom line

If I'm being frank: PrivateUse1/OpenReg gives out-of-tree accelerators a **development pathway** that's dramatically better than what existed before PyTorch 2.1 (where vendors had to fork PyTorch entirely). But it does *not* give them CI/CD parity with in-tree accelerators. The gaps are:

- Detection latency: days/weeks vs. hours
- Coverage: mechanism plumbing vs. full functional + performance
- BC guarantees: informal vs. formal
- Test infrastructure: partially adapted vs. native

For a vendor like Huawei (Ascend NPU) or IBM (Spyre) with dedicated engineering teams, the out-of-tree path is workable — they build their own layered CI, accept the latency cost, and maintain close relationships with PyTorch maintainers to get breakage fixed quickly. For a smaller vendor or a research lab building a custom accelerator, the friction is significantly higher. The experience today is more like maintaining a downstream Linux kernel fork than it is like developing a kernel module against a stable ABI.



## Real-World PyTorch Model Compatibility CI Systems

### 1. **TorchBench** (Meta / PyTorch team) — `pytorch/benchmark`

The official PyTorch benchmark and CI suite.

- A collection of open-source benchmarks used to evaluate PyTorch performance. `torchbenchmark/models` contains copies of popular or exemplary workloads modified to expose a standardized API for benchmark drivers and optionally enable backends such as TorchInductor/TorchScript.

- TorchBench consists of 84 models from various domains. Based on GitHub Actions, they create a series of GitHub workflows to continuously test performance regression with TorchBench — this was the first such CI effort for the PyTorch repository.

- Currently, models run on nightly PyTorch builds and push data to Meta's internal database. The Nightly CI publishes both V1 and V0 performance scores.

- Integrating it into your own CI is straightforward:

  ```python
  from torchbenchmark.models import stable_diffusion_text_encoderbenchmark = Model(test="eval", device="cuda")model, example = benchmark.get_module()model(*example)
  ```

------

### 2. **torch.compile Compatibility Suite** (PyTorch 2.0 launch)

Used to validate `torch.compile` across a massive model corpus before release.

- PyTorch ran `torch.compile` on 163 open-source models from HuggingFace, TIMM, and TorchBench to evaluate performance gains across tasks like Image Classification, Object Detection, Image Generation, NLP (Language Modeling, Q&A, Sequence Classification), Recommender Systems, and Reinforcement Learning. `torch.compile` worked on 93% of the models, achieving 43% faster execution on NVIDIA A100.

------

### 3. **Hugging Face Transformers CI** — model zoo testing

- Any changes to the modeling or PyTorch examples code requires running the model zoo tests. Tests requiring GPUs run on a different CI nightly (since PR CI doesn't have GPUs). Slow tests run on a scheduled basis rather than in PR CI checks.
- They use a `@slow` decorator pattern to skip heavy model tests in PRs and run them nightly instead — a common pattern for large model suites.

------

### 4. **BackendBench** (GPUMODE)

- Provides operators and inputs derived from 155 model traces found in TIMM (67), HuggingFace Transformers (45), and TorchBench (43). These are also the models PyTorch developers use to validate performance.

------

### 5. **Common Infrastructure Push** (torchbench + TIMM + HuggingFace)

- There has been active work on a common benchmarking infrastructure that allows getting comparable results for different backend technologies. TorchBench already installs HF and TIMM repos as part of its setup, but models and configs not part of TorchBench are not easy to access.
- Running each model in a subprocess is recommended — it adds ~10s per model initialization but is a worthy price for reliable, isolated results. The `--isolate` flag in torchbench.py enables subprocess isolation.

------

### Key Design Patterns from Production Systems

| Pattern                       | How it's done                                                |
| ----------------------------- | ------------------------------------------------------------ |
| **Tiered testing**            | Fast tests on every PR; slow/GPU tests nightly               |
| **Subprocess isolation**      | Each model runs in its own process to prevent state contamination |
| **Standardized model API**    | All models expose `get_module()` → `(model, example_inputs)` |
| **Multi-source model corpus** | Pull from TorchBench + TIMM + HuggingFace for broad coverage |
| **Nightly vs. PR gates**      | Only ~10–20 representative models block PRs; full suite runs overnight |

The PyTorch team's approach of testing across 163+ models before a major release (2.0) is essentially the gold standard for this kind of compatibility CI at scale.



## CI/CD repos

Based on the PyTorch organization's repositories, here are the projects specifically related to testing, infrastructure, and CI/CD:

* **[test-infra](https://github.com/pytorch/test-infra)**: Hosts code supporting the testing infrastructure for the PyTorch organization, including logic to track disabled tests.
* **[ci-infra](https://github.com/pytorch/ci-infra)**: Infrastructure as code (HCL/Terraform) for the organization's continuous integration setup.
* **[pytorch-integration-testing](https://github.com/pytorch/pytorch-integration-testing)**: Used for testing downstream libraries using PyTorch release candidates.
* **[builder](https://github.com/pytorch/builder)**: Contains the continuous builder and binary build scripts for PyTorch.
* **[ci-hud](https://github.com/pytorch/ci-hud)**: A dashboard (HUD) for visualizing CI activity and job status across the organization.
* **[pytorch-ci-dockerfiles](https://github.com/pytorch/pytorch-ci-dockerfiles)**: Scripts used to generate the Docker images required for PyTorch CI environments.
* **[dr-ci](https://github.com/pytorch/dr-ci)**: A tool designed to help diagnose and remediate CI job failures.
* **[expecttest](https://github.com/pytorch/expecttest)**: A specialized testing utility used within the PyTorch ecosystem.
* **[add-annotations-github-action](https://github.com/pytorch/add-annotations-github-action)**: A custom GitHub Action to run tools like clang-tidy and annotate failures directly on PRs.
* **[labeler-github-action](https://github.com/pytorch/labeler-github-action)**: A GitHub Action for automatically labeling issues and pull requests.
* **[ossci-job-dsl](https://github.com/pytorch/ossci-job-dsl)**: Jenkins job definitions for the OSSCI (Open Source Science Initiative) infrastructure.
* **[torchhub_testing](https://github.com/pytorch/torchhub_testing)**: A dedicated repository for testing TorchHub functionality.



## References

- https://deepwiki.com/pytorch/pytorch
- [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/pdf/1912.01703) 
- [TORCH.FX: PRACTICAL PROGRAM CAPTURE AND TRANSFORMATION FOR DEEP LEARNING IN PYTHON](https://arxiv.org/pdf/2112.08429) 
- [FX Technical Overview](https://github.com/pytorch/pytorch/blob/main/torch/fx/README.md)
- [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation](https://docs.pytorch.org/assets/pytorch2-2.pdf)
- [PyTorch internals -- 讲 pytorch 原理的文章，可以好好看看，这个是给 pytorch 社区贡献代码的基础知识](https://blog.ezyang.com/2019/05/pytorch-internals/)
- [PyTorch 2.0 & XLA—The Latest Cutting Edge Features](https://pytorch.org/blog/pytorch-2-0-xla/)
- [TorchDynamo(torch.compile) integration in PyTorch XLA](https://docs.pytorch.org/xla/master/torch_compile.html)
- [OpenReg](https://pytorch.org/blog/openreg-a-self-contained-pytorch-accelerator-simulator/)
- 

