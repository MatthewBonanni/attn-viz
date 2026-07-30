# Attention Mechanism Visualizer

**Try it:** https://matthewbonanni.github.io/attn-viz/

An interactive, tensor-level visualization of transformer attention. It renders
tensors and operations as isometric 3D blocks, updates shapes and costs as model
and runtime parameters change, and provides detailed diagrams for individual
operations.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![Flask](https://img.shields.io/badge/flask-3.x-green)
![D3.js](https://img.shields.io/badge/d3.js-v7-orange)

## Supported attention variants

| Variant | What the visualizer shows |
| --- | --- |
| **MHA** | Independent query, key, and value projections for every head |
| **GQA** | Query-head groups sharing a smaller number of KV heads |
| **MQA** | All query heads sharing one KV head |
| **MLA** | Compressed KV latents and side-by-side up-projected and absorbed paths |
| **DSA** | MLA with a lightning indexer, causal top-k selection, and sparse core attention |

Presets cover representative GPT-2, Llama, Mistral, Qwen, Gemma, Phi, Command,
StarCoder, and DeepSeek configurations.

## Features

- Interactive tensor shapes, operation graphs, tooltips, and click-through detail views
- Independent per-request context (`S`) and query (`S_q`) lengths for mixed batches
- Prefill, extend, speculative-decode, and decode workload visualization
- FlashAttention, PagedAttention, and sliding-window attention overlays
- Tensor-parallel and data-parallel layouts, including TP collective traffic
- KV-cache size, paged-block occupancy, and variant-specific derived values
- FLOPs, HBM traffic, arithmetic intensity, and ideal roofline lower bounds for
  A100, H100, B200, and B300
- MLA path crossover analysis and DSA dense-fallback visibility
- Searchable glossary and shareable, URL-encoded view state

## Run locally

The project requires Python 3.11 or newer and
[uv](https://docs.astral.sh/uv/getting-started/installation/). D3 is loaded from
the public D3 CDN, so the page needs network access when it first loads.

```bash
uv sync
uv run python app.py
```

Open http://localhost:5001. The Flask server only serves the static application;
all graph construction, rendering, and cost calculations run in the browser.

## Use the visualizer

1. Choose an attention variant or model preset.
2. Set the architecture dimensions and the batch's per-request `S` and `S_q` values.
3. Adjust TP/DP ranks or enable FlashAttention, PagedAttention, or sliding-window attention.
4. Click a tensor, operation, or outlined group for its detailed explanation.
5. Scroll to zoom, drag to pan, and double-click to fit the graph to the viewport.
6. Select **Copy link** to share the exact configuration.

DSA uses its own fused sparse-attention path, so the FlashAttention and
sliding-window controls do not apply to that view. PagedAttention changes cache
layout and block accounting; it does not change the mathematical attention result.

## Performance model

The cost panel is an explanatory model, not a benchmark or latency predictor. It
reports single-GPU work when TP=DP=1. With parallelism enabled, it reports local TP
work and the busiest data-parallel rank. Variable-length batches are priced from
request-local attention volumes rather than allowing attention across requests.

For each GPU, the displayed time is this ideal lower bound:

```text
sum(max(op FLOPs / dtype peak, op HBM bytes / peak bandwidth))
    + serialized TP ring all-reduce at peak interconnect bandwidth
```

The estimator:

- counts the exact live causal band for sliding-window attention;
- limits DSA's sparse core to the causally available top-k tokens;
- uses BF16 peaks for ordinary attention and FP8 peaks for FP8 indexer operations;
- models the HBM reads and writes of the displayed operations, including annotated
  FlashAttention fusion and compact GQA/MQA storage; and
- links each GPU row to the hardware specification used by the model.

It excludes kernel-launch overhead, occupancy and utilization loss, cache effects,
contention, synchronization, software overhead, network topology, and overlap.
Actual end-to-end latency will therefore be higher and may have a different
bottleneck.

## Development and tests

The cost-model tests use Node's built-in test runner and have no npm dependencies:

```bash
npm test
```

The tests cover mixed-request batching, exact sliding-window and DSA pair counts,
DP critical-rank selection, TP communication, compact GQA storage, and per-operation
roofline aggregation.

## Project structure

```text
├── app.py                    # Minimal Flask development server
├── index.html                # Single-page UI and styles
├── package.json              # JavaScript test command and ES module config
├── pyproject.toml            # Python metadata and Flask dependency
├── preview_card.html         # Social preview artwork source
├── static/
│   ├── preview.png           # Social preview image
│   └── js/
│       ├── main.js           # Controls, presets, state, and update loop
│       ├── graphs.js         # Attention graph definitions
│       ├── render.js         # Isometric SVG layout and rendering
│       ├── costs.js          # Operation costs and roofline estimates
│       ├── url-state.js      # Shareable URL serialization
│       └── details/          # Tensor and operation detail visualizations
└── tests/
    └── costs.test.js         # Cost-model regression tests
```
