Here's a detailed walkthrough of TorchDynamo's implementation:

------

## Architecture Overview

TorchDynamo is a Python-level JIT compiler that intercepts CPython's frame evaluation to capture PyTorch operations into an FX graph, optimize it via a backend (e.g., Inductor), and cache the result with guards for validity.

------

## 1. Entry Point: Frame Evaluation Hook

**[eval_frame.py](vscode-webview://05ae4l0dinki967cd4fcokgkc205cpm98srb0hr8i7binord5l30/torch/_dynamo/eval_frame.py)**

When you call `torch.compile(fn)`, Dynamo hooks into CPython's frame evaluation via `torch._C._dynamo.eval_frame`. Every time the compiled function is called, this C extension intercepts the frame *before* Python executes it.

Key mechanisms:

- **DynamoStance** controls compilation mode (`"default"`, `"force_eager"`, `"eager_then_compile"`, etc.)
- **OptimizedModule** wraps `nn.Module`, patching `forward` to route through Dynamo while preserving the original module's interface
- **`_callback_from_stance()`** selects the compilation backend based on current stance

------

## 2. Frame Conversion: [convert_frame.py](vscode-webview://05ae4l0dinki967cd4fcokgkc205cpm98srb0hr8i7binord5l30/torch/_dynamo/convert_frame.py)

The actual compilation pipeline. `ConvertFrame` receives a Python frame and:

1. Preserves global state (grad mode, RNG, dispatch keys)
2. Creates an `InstructionTranslator` to symbolically execute the bytecode
3. Builds guards for cache invalidation
4. Generates an FX graph
5. Caches the compiled code

Uses `input_codes`/`output_codes` to track which code objects have been compiled and what was generated.

------

## 3. Symbolic Execution: [symbolic_convert.py](vscode-webview://05ae4l0dinki967cd4fcokgkc205cpm98srb0hr8i7binord5l30/torch/_dynamo/symbolic_convert.py)

The largest file (~255K bytes) — the core tracing engine.

### InstructionTranslatorBase

Symbolically executes Python bytecode using a dispatch table (one handler per opcode). Key state:

- **`symbolic_locals`** — local variables as `VariableTracker` objects
- **`stack`** — the Python value stack, but symbolic
- **`instruction_pointer`** — current bytecode offset
- **`block_stack`** — tracks context managers / with blocks

**`step()`** processes one bytecode instruction: updates line info, dispatches to the opcode handler (e.g., `_opcode_LOAD_FAST`, `_opcode_BINARY_ADD`), and catches `Unsupported` exceptions for graph breaks.

### Graph Breaks

When Dynamo hits an unsupported operation:

1. Catches the `Unsupported` exception
2. Compiles the graph accumulated so far
3. Creates a **resume function** for the remaining code
4. Restarts analysis using the `SpeculationLog`, skipping past the break point

This allows partial compilation — the function is split into compiled subgraphs connected by eager Python.

### Function Inlining

`inline_user_function_return()` uses `InliningInstructionTranslator` to trace into user-defined functions, expanding the graph without a function call boundary.

------

## 4. Variable Tracking: [variables/](vscode-webview://05ae4l0dinki967cd4fcokgkc205cpm98srb0hr8i7binord5l30/torch/_dynamo/variables/)

Every Python value during tracing is represented as a `VariableTracker` subclass:

| Type                           | Represents                                                   |
| ------------------------------ | ------------------------------------------------------------ |
| `TensorVariable`               | `torch.Tensor` — stores dtype, device, shape, stride, requires_grad; operations create FX proxy nodes |
| `ConstantVariable`             | Immutable Python values (int, float, str, bool, None)        |
| `SymNodeVariable`              | Symbolic scalars (SymInt, SymFloat) for dynamic shapes       |
| `NNModuleVariable`             | `nn.Module` instances — handles parameter/buffer access      |
| `UserFunctionVariable`         | User-defined functions — can be inlined                      |
| `ListVariable`, `DictVariable` | Container types with mutation tracking                       |

**`VariableBuilder`** ([variables/builder.py](vscode-webview://05ae4l0dinki967cd4fcokgkc205cpm98srb0hr8i7binord5l30/torch/_dynamo/variables/builder.py)) is the factory — given a real Python value, it determines the right `VariableTracker` class, creates guards, and wraps it with source tracking.

### Mutation Types

- **`ValueMutationNew`** — mutations on objects created during tracing (can be skipped if object doesn't escape)
- **`ValueMutationExisting`** — mutations on pre-existing objects (must be replayed after graph execution)
- **`AttributeMutation\*`** — same distinction for attribute mutations

------

## 5. Guard System: [guards.py](vscode-webview://05ae4l0dinki967cd4fcokgkc205cpm98srb0hr8i7binord5l30/torch/_dynamo/guards.py)

Guards are conditions that must hold for cached compiled code to remain valid. If any guard fails, recompilation is triggered.

**Guard categories:**

- **Type guards** — `type(x) is Tensor`
- **Identity guards** — `x is same_object`
- **Tensor guards** — dtype, device, layout, shape, requires_grad
- **Container guards** — dict size/keys, list/tuple length
- **Relational guards** — symbolic shape constraints

Guards form a **tree** (`RootGuardManager` → child `GuardManager` nodes) matching the object hierarchy. At runtime, `check_nopybind()` (C++) walks this tree for fast validation.

------

## 6. Output Graph: [output_graph.py](vscode-webview://05ae4l0dinki967cd4fcokgkc205cpm98srb0hr8i7binord5l30/torch/_dynamo/output_graph.py)

`OutputGraph` coordinates FX graph construction:

- Manages `SubgraphTracer` (extends `fx.Tracer`) — records operations as FX nodes, creates proxies
- Collects graph inputs/outputs, guards, `nn_modules`
- Handles symbolic shapes and shape specialization
- Supports nested tracers for higher-order ops (vmap, map, etc.)

**`compile_subgraph()`** finalizes the graph: extracts outputs, removes dead code, installs guards, and passes to the backend compiler.

------

## 7. Code Generation: [codegen.py](vscode-webview://05ae4l0dinki967cd4fcokgkc205cpm98srb0hr8i7binord5l30/torch/_dynamo/codegen.py)

`PyCodegen` generates Python bytecode for the output code:

1. Load inputs onto the stack
2. Call the compiled function (Inductor kernel, etc.)
3. Replay side effects (mutations)
4. Reconstruct local variables for any code after the graph

Uses a **tempvar** system to avoid redundant reconstruction, and prefers **source-based reconstruction** to preserve aliasing relationships.

------

## 8. Side Effects: [side_effects.py](vscode-webview://05ae4l0dinki967cd4fcokgkc205cpm98srb0hr8i7binord5l30/torch/_dynamo/side_effects.py)

`SideEffects` tracks mutations during tracing:

- **`store_attr_mutations`** — attribute mutations per variable
- **`id_to_variable`** — maps object ids to tracked variables
- **`check_allowed_side_effect()`** — validates mutation safety (blocks unsafe mutations in higher-order ops)

During codegen, mutations are replayed after graph execution: `obj.attr = new_value`, `dict[key] = value`, etc.

------

## 9. End-to-End Execution Flow



```
torch.compile(fn)(x)
  │
  ▼
CPython frame hook (C extension)
  │
  ▼
Check cache: guards valid? ──yes──► Execute cached compiled code
  │ no
  ▼
ConvertFrame: preserve global state
  │
  ▼
InstructionTranslator: symbolic bytecode execution
  │  LOAD_FAST 'x'  → TensorVariable (proxy)
  │  LOAD_CONST 1   → ConstantVariable(1)
  │  BINARY_ADD     → FX node: add(x, 1)
  │  RETURN_VALUE   → mark output
  │
  ▼
OutputGraph.compile_subgraph()
  │  finalize graph, install guards
  │
  ▼
Backend compiler (Inductor → C++/Triton kernels)
  │
  ▼
PyCodegen: generate output bytecode
  │
  ▼
Cache with RootGuardManager, execute
```

------

## Key Design Patterns

- **Lazy evaluation** — symbolic execution records operations, doesn't compute values
- **Checkpoint & rollback** — graph breaks restart analysis with speculation log to skip past failure points
- **Guard tree** — hierarchical guard structure matching Python object hierarchy for fast validity checking
- **Source tracking** — every variable knows where it came from, enabling correct reconstruction and aliasing