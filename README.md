# Attention Mechanism Visualizer

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
- **Model presets** — GPT-2, Llama 3.1, Mistral 7B, Qwen 2.5, StarCoder, DeepSeek R1
- **Paged Attention** toggle for block-based KV cache visualization
- **Tensor Parallelism** toggle to show sharding across ranks
- **RoPE** (Rotary Position Embedding) applied to all variants
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
        ├── main.js         # Entry point, sliders, presets, update loop
        ├── render.js       # Isometric projection and SVG rendering
        ├── graphs.js       # Attention mechanism definitions
        └── details.js      # Detail panel visualizations
```

## How It Works

1. Select an attention variant (MHA/GQA/MQA/MLA) or load a model preset
2. Adjust dimension sliders — the visualization updates immediately
3. Click any tensor block or operation node to inspect it in the detail panel
4. Toggle Paged Attention or Tensor Parallelism to see how they change the computation graph
