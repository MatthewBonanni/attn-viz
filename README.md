# Attention Mechanism Visualizer

**Try it:** https://matthewbonanni.github.io/attn-viz/

Interactive visualization tool for exploring how self-attention works in transformer models. Renders tensor shapes and operations as isometric 3D blocks, letting you adjust architecture parameters in real time and compare different attention variants.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![Flask](https://img.shields.io/badge/flask-3.1-green)
![D3.js](https://img.shields.io/badge/d3.js-v7-orange)

## Attention Variants

- **MHA** — Multi-Head Attention (standard transformer)
- **GQA** — Grouped-Query Attention (Llama, Mistral, Qwen)
- **MQA** — Multi-Query Attention (StarCoder, Falcon)
- **MLA** — Multi-Head Latent Attention (DeepSeek)

## Features

- **Isometric 3D tensor blocks** colored by role (Q/K/V/output/etc.)
- **Adjustable parameters** — batch size, sequence length, heads, head dimension, KV heads, latent dimension, RoPE dimension, tensor parallelism ranks, block size
- **Model presets** — GPT-2, Llama 3.1, Mistral 7B, Qwen 2.5/3, Gemma 3, Phi-4, Command A, StarCoder, DeepSeek R1
- **FlashAttention** toggle to show fused kernel tiling
- **Paged Attention** toggle for block-based KV cache visualization
- **Tensor Parallelism** toggle to show sharding across ranks
- **RoPE** (Rotary Position Embedding) applied to all variants
- **Cost analysis** — FLOPs, memory transfer, arithmetic intensity, and ideal roofline lower bounds for A100/H100/B200/B300
- **Click any tensor or operation** to see detailed breakdowns in the detail panel
- **MLA dual-path view** showing both prefill and decode paths

## Getting Started

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Run the server
python app.py
```

Open [http://localhost:5001](http://localhost:5001) in your browser.

## Project Structure

```
├── app.py                  # Flask server
├── pyproject.toml          # Python project config
├── templates/
│   └── index.html          # Single-page app
└── static/
    └── js/
        ├── main.js         # Entry point, sliders, presets, toggles, update loop
        ├── render.js       # Isometric projection and SVG rendering
        ├── graphs.js       # Attention variant graph definitions
        ├── costs.js        # FLOPs, memory, arithmetic intensity, roofline analysis
        └── details/        # Detail panel visualizations (matmul, softmax,
            ├── index.js    #   FlashAttention tiling, RoPE, KV cache, mask, etc.)
            └── ...
```

## How It Works

1. Select an attention variant (MHA/GQA/MQA/MLA) or load a model preset
2. Adjust dimension sliders — the visualization updates immediately
3. Click any tensor block or operation node to inspect it in the detail panel
4. Toggle Paged Attention or Tensor Parallelism to see how they change the computation graph

## Performance Model

The cost overlay reports single-GPU work or the critical-rank work when tensor/data
parallelism is enabled. Variable-length batches use request-local attention volumes
(`sum(S_q[i] * S[i])`) rather than multiplying packed-batch totals. Sliding-window
attention counts the exact live causal band, and DSA counts the causal top-k volume.

GPU time is an **ideal lower bound**, not a latency prediction:

```text
sum(max(op FLOPs / dtype peak, op HBM bytes / peak bandwidth))
    + serialized TP ring all-reduce at peak interconnect bandwidth
```

It uses dense BF16/FP8 hardware peaks and excludes kernel launches, occupancy loss,
contention, synchronization, and other implementation overhead. GPU specification
sources are linked from the table in the app.
