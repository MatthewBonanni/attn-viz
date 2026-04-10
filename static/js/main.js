// main.js — Entry point: D3 setup, zoom/pan, sliders, presets, toggles, update loop

import { renderGraph, computeSharedStagePositions } from './render.js';
import { mhaGraph, gqaGraph, mqaGraph, mlaUpprojGraph, mlaAbsorbedGraph, VARIANT_DESCS } from './graphs.js';
import { showDetail, showTensorDetail, hideDetail, refreshDetail } from './details/index.js';

const GRAPH_FNS = {
    mha: mhaGraph, gqa: gqaGraph, mqa: mqaGraph,
};

const VARIANTS = [
    { id: 'mha', label: 'MHA' },
    { id: 'gqa', label: 'GQA' },
    { id: 'mqa', label: 'MQA' },
    { id: 'mla', label: 'MLA' },
];

const SLIDER_DEFS = {
    B:      { label: 'B (batch)',         min: 1, max: 16,   step: 1,  default: 2 },
    S:      { label: 'S (seq length)',    min: 1, max: 2048, step: 1,  default: 8 },
    n_h:    { label: 'n_h (heads)',       min: 1, max: 128,  step: 1,  default: 8 },
    d_h:    { label: 'd_h (head dim)',    min: 1, max: 256,  step: 1,  default: 64 },
    n_kv:   { label: 'n_kv (KV heads)',   min: 1, max: 128,  step: 1,  default: 2 },
    d_c:    { label: 'd_c (latent dim)',  min: 1, max: 4096, step: 1,  default: 512 },
    d_r:    { label: 'd_r (RoPE dim)',   min: 1, max: 256,  step: 1,  default: 64 },
    tp_size:{ label: 'TP ranks',          min: 1, max: 8,    step: 1,  default: 1 },
    block_size: { label: 'Block size',    min: 1, max: 128,  step: 1,  default: 16 },
};

const RUNTIME_SLIDERS = ['B', 'S'];

const VARIANT_SLIDERS = {
    mha: ['n_h', 'd_h', 'tp_size'],
    gqa: ['n_h', 'd_h', 'n_kv', 'tp_size'],
    mqa: ['n_h', 'd_h', 'tp_size'],
    mla: ['n_h', 'd_h', 'd_c', 'd_r', 'tp_size'],
};

// Model presets
const PRESETS = [
    { name: 'Custom', variant: null },
    { name: 'GPT-2 (124M)', variant: 'mha', B: 1, S: 12, n_h: 12, d_h: 64 },
    { name: 'GPT-2 XL (1.5B)', variant: 'mha', B: 1, S: 12, n_h: 25, d_h: 64 },
    { name: 'Llama 3.1 8B', variant: 'gqa', B: 1, S: 12, n_h: 32, d_h: 128, n_kv: 8 },
    { name: 'Llama 3.1 70B', variant: 'gqa', B: 1, S: 12, n_h: 64, d_h: 128, n_kv: 8 },
    { name: 'Llama 3.1 405B', variant: 'gqa', B: 1, S: 12, n_h: 128, d_h: 128, n_kv: 8 },
    { name: 'Mistral 7B', variant: 'gqa', B: 1, S: 12, n_h: 32, d_h: 128, n_kv: 8 },
    { name: 'Qwen 2.5 72B', variant: 'gqa', B: 1, S: 12, n_h: 64, d_h: 128, n_kv: 8 },
    { name: 'StarCoder (15B)', variant: 'mqa', B: 1, S: 12, n_h: 48, d_h: 128 },
    { name: 'DeepSeek R1', variant: 'mla', B: 1, S: 12, n_h: 128, d_h: 128, d_c: 512, d_r: 64 },
];

// Default preset index per variant
const VARIANT_DEFAULT_PRESETS = {
    mha: 1,   // GPT-2 (124M)
    gqa: 3,   // Llama 3.1 8B
    mqa: 8,   // StarCoder (15B)
    mla: 9,   // DeepSeek R1
};

// --- State ---

let params = {};
for (const [k, v] of Object.entries(SLIDER_DEFS)) {
    params[k] = v.default;
}
params.pagedAttn = false;
params.seqLens = [4, 8];     // per-request context lengths (tokens in KV cache)
params.queryLens = [4, 1];   // per-request new tokens (prefill=many, decode=1)

let currentVariant = 'mha';

// --- SVG + zoom setup ---

const svg = d3.select('#main-svg');
const scene = svg.append('g').attr('id', 'scene');

const zoom = d3.zoom()
    .scaleExtent([0.05, 8])
    .on('zoom', (event) => scene.attr('transform', event.transform));

svg.call(zoom);
svg.on('dblclick.zoom', fitToView);

function fitToView() {
    const bbox = scene.node().getBBox();
    if (bbox.width === 0 || bbox.height === 0) return;
    const { width, height } = svg.node().getBoundingClientRect();
    const pad = 60;
    const scale = Math.min(
        (width - pad * 2) / bbox.width,
        (height - pad * 2) / bbox.height
    ) * 0.9;
    const tx = (width - bbox.width * scale) / 2 - bbox.x * scale;
    const ty = (height - bbox.height * scale) / 2 - bbox.y * scale;
    svg.transition().duration(400)
        .call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}

// --- Variant tabs ---

function buildVariantTabs() {
    const container = d3.select('#variant-tabs');
    container.selectAll('*').remove();

    for (const v of VARIANTS) {
        container.append('button')
            .attr('data-variant', v.id)
            .classed('active', v.id === currentVariant)
            .text(v.label)
            .on('click', () => {
                currentVariant = v.id;
                container.selectAll('button').classed('active', false);
                container.select(`[data-variant="${v.id}"]`).classed('active', true);
                // Apply default preset for this variant
                const presetIdx = VARIANT_DEFAULT_PRESETS[v.id] || 0;
                const preset = PRESETS[presetIdx];
                if (preset && preset.variant) {
                    for (const key of ['B', 'S', 'n_h', 'd_h', 'n_kv', 'd_c', 'd_r']) {
                        if (preset[key] != null) params[key] = preset[key];
                    }
                }
                d3.select('#preset-select').property('value', String(presetIdx));
                updateVariantDesc();
                buildSliders();
                update();
                setTimeout(fitToView, 50);
            });
    }
}

// --- Variant description ---

function updateVariantDesc() {
    d3.select('#variant-desc').html(VARIANT_DESCS[currentVariant] || '');
}

// --- Presets ---

function buildPresets() {
    const select = d3.select('#preset-select');
    select.selectAll('*').remove();

    PRESETS.forEach((p, i) => {
        select.append('option')
            .attr('value', i)
            .property('disabled', p.disabled || false)
            .text(p.name);
    });

    select.on('change', function() {
        const preset = PRESETS[+this.value];
        if (!preset || !preset.variant) return;

        currentVariant = preset.variant;
        for (const key of ['B', 'S', 'n_h', 'd_h', 'n_kv', 'd_c', 'd_r']) {
            if (preset[key] != null) params[key] = preset[key];
        }

        d3.select('#variant-tabs').selectAll('button').classed('active', false);
        d3.select('#variant-tabs').select(`[data-variant="${currentVariant}"]`).classed('active', true);
        updateVariantDesc();
        buildSliders();
        update();
        setTimeout(fitToView, 50);
    });
}

// --- Toggles ---

function setupToggles() {
    // Paged Attention toggle
    d3.select('#toggle-paged').on('click', function() {
        params.pagedAttn = !params.pagedAttn;
        d3.select(this).classed('active', params.pagedAttn);
        d3.select('#seq-lengths').classed('visible', params.pagedAttn);
        buildSliders();
        update();
    });

}

// --- Sliders ---

function buildSlider(container, key) {
    const def = SLIDER_DEFS[key];
    const group = container.append('div').attr('class', 'slider-group');
    const header = group.append('div').attr('class', 'slider-header');
    header.append('span').attr('class', 'dim-name').text(def.label);

    const numInput = header.append('input')
        .attr('class', 'dim-input')
        .attr('type', 'number')
        .attr('min', def.min)
        .attr('max', key === 'n_kv' ? params.n_h : key === 'tp_size' ? Math.min(8, params.n_h) : def.max)
        .attr('step', def.step)
        .attr('value', params[key]);

    const rangeInput = group.append('input')
        .attr('type', 'range')
        .attr('min', def.min)
        .attr('max', key === 'n_kv' ? params.n_h : key === 'tp_size' ? Math.min(8, params.n_h) : def.max)
        .attr('step', def.step)
        .attr('value', params[key]);

    function onSliderChange(newVal) {
        let v = +newVal;
        // Snap B and tp_size to nearest power of 2
        if (key === 'B' || key === 'tp_size') {
            v = Math.pow(2, Math.round(Math.log2(Math.max(1, v))));
            v = Math.max(def.min, Math.min(key === 'tp_size' ? Math.min(8, params.n_h) : def.max, v));
        }
        params[key] = v;

        // Switch preset to "Custom" when a model architecture param changes
        if (['n_h', 'd_h', 'n_kv', 'd_c', 'd_r'].includes(key)) {
            d3.select('#preset-select').property('value', '0');
        }

        if (key === 'n_h' && currentVariant === 'gqa') {
            if (params.n_kv > params.n_h) params.n_kv = params.n_h;
        }
        if (key === 'n_kv') {
            params.n_kv = Math.min(params.n_kv, params.n_h);
        }
        if (key === 'n_h' && params.tp_size > 1) {
            if (params.tp_size > Math.min(8, params.n_h)) params.tp_size = Math.min(8, params.n_h);
        }

        numInput.property('value', params[key]);
        rangeInput.property('value', params[key]);

        // Extend seqLens/queryLens arrays BEFORE update() so addPagedAnnotations sees the right length
        if ((key === 'B' || key === 'S') && params.pagedAttn) buildSeqLengthInputs();

        updateDerived();
        update();

        // Update n_kv and tp_size slider max when n_h changes (don't rebuild — that kills the drag)
        if (key === 'n_h') {
            d3.selectAll('#sliders input[type="range"]').each(function() {
                const group = this.parentNode;
                const label = d3.select(group).select('.dim-name').text();
                if (label.includes('n_kv')) {
                    d3.select(this).attr('max', params.n_h);
                    d3.select(group).select('input[type="number"]').attr('max', params.n_h);
                } else if (label.includes('TP')) {
                    const tpMax = Math.min(8, params.n_h);
                    d3.select(this).attr('max', tpMax);
                    d3.select(group).select('input[type="number"]').attr('max', tpMax);
                }
            });
        }
    }

    rangeInput.on('input', function() { onSliderChange(this.value); });
    numInput.on('change', function() {
        let v = Math.max(def.min, Math.min(key === 'n_kv' ? params.n_h : key === 'tp_size' ? Math.min(8, params.n_h) : def.max, +this.value || def.min));
        // Snap to power of 2 for B and tp_size
        if (key === 'B' || key === 'tp_size') {
            v = Math.pow(2, Math.round(Math.log2(Math.max(1, v))));
            v = Math.max(def.min, Math.min(key === 'tp_size' ? Math.min(8, params.n_h) : def.max, v));
        }
        this.value = v;
        onSliderChange(v);
    });
}

function buildSliders() {
    // Runtime sliders (B, S) — always visible, variant-independent
    const rtContainer = d3.select('#runtime-sliders');
    rtContainer.selectAll('*').remove();
    for (const key of RUNTIME_SLIDERS) {
        buildSlider(rtContainer, key);
    }

    // Model architecture sliders (variant-specific)
    const dimContainer = d3.select('#sliders');
    dimContainer.selectAll('*').remove();
    for (const key of VARIANT_SLIDERS[currentVariant]) {
        buildSlider(dimContainer, key);
    }

    // Paged attention sliders (separate section)
    const pagedContainer = d3.select('#paged-sliders');
    pagedContainer.selectAll('*').remove();
    pagedContainer.classed('visible', params.pagedAttn);
    if (params.pagedAttn) {
        pagedContainer.append('div').attr('class', 'slider-section-label').text('Paged Attention');
        buildSlider(pagedContainer, 'block_size');
        buildSeqLengthInputs();
    }

    updateDerived();
}

function buildSeqLengthInputs() {
    const container = d3.select('#seq-lengths');
    container.selectAll('*').remove();
    container.classed('visible', true);

    container.append('div').style('font-size', '10px').style('color', '#666')
        .style('margin-bottom', '4px').text('Per-request lengths (cached context + new tokens):');

    // Ensure arrays match B
    while (params.seqLens.length < params.B) params.seqLens.push(params.S);
    while (params.seqLens.length > params.B) params.seqLens.pop();
    while (params.queryLens.length < params.B) params.queryLens.push(1);
    while (params.queryLens.length > params.B) params.queryLens.pop();

    // Header row
    const header = container.append('div').attr('class', 'seq-row')
        .style('color', '#666').style('font-size', '9px');
    header.append('span').style('width', '42px').text('');
    header.append('span').style('width', '50px').style('text-align', 'center').text('Cached');
    header.append('span').style('width', '50px').style('text-align', 'center').text('New');
    header.append('span').style('font-style', 'italic').text('Type');

    for (let i = 0; i < params.B; i++) {
        const row = container.append('div').attr('class', 'seq-row');
        row.append('span').style('width', '42px').text(`Req ${i}:`);

        // Context length (cached KV tokens)
        const ctxInp = row.append('input')
            .attr('type', 'number')
            .attr('min', 0)
            .attr('step', 1)
            .property('value', params.seqLens[i]);

        ctxInp.on('input', function() {
            let v = parseInt(this.value, 10);
            if (isNaN(v)) return;
            v = Math.max(0, v);
            params.seqLens[i] = v;
            updateDerived();
        });
        ctxInp.on('change', function() {
            let v = parseInt(this.value, 10);
            if (isNaN(v)) v = 0;
            v = Math.max(0, v);
            params.seqLens[i] = v;
            this.value = v;
            updateDerived();
            update();
        });

        // Query length (new tokens)
        const qInp = row.append('input')
            .attr('type', 'number')
            .attr('min', 1)
            .attr('step', 1)
            .property('value', params.queryLens[i]);

        qInp.on('input', function() {
            let v = parseInt(this.value, 10);
            if (isNaN(v)) return;
            v = Math.max(1, v);
            params.queryLens[i] = v;
            updateDerived();
        });
        qInp.on('change', function() {
            let v = parseInt(this.value, 10);
            if (isNaN(v)) v = 1;
            v = Math.max(1, v);
            params.queryLens[i] = v;
            this.value = v;
            updateDerived();
            update();
        });

        // Type label (auto-detect prefill vs decode)
        row.append('span')
            .style('font-size', '9px')
            .style('color', params.queryLens[i] > 1 ? '#f39c12' : '#3498db')
            .text(params.queryLens[i] > 1 ? 'prefill' : 'decode');
    }
}

function updateDerived() {
    const D = params.n_h * params.d_h;
    let html = `<div class="derived-dim">D = n_h × d_h = <span>${D}</span></div>`;

    if (currentVariant === 'mla') {
        const dr = params.d_r || 64;
        const totalCache = params.d_c + dr;
        const ratio = (2 * params.n_h * params.d_h / totalCache).toFixed(1);
        html += `<div class="derived-dim">Cache per token: <span>d_c + d_r = ${totalCache}</span></div>`;
        html += `<div class="derived-dim">KV cache reduction: <span>${ratio}×</span></div>`;
    }
    if (currentVariant === 'gqa') {
        const gpc = Math.floor(params.n_h / params.n_kv);
        const ratio = (params.n_h / params.n_kv).toFixed(1);
        html += `<div class="derived-dim">Heads per group: <span>${gpc}</span> (${ratio}× reduction)</div>`;
    }
    if (params.tp_size > 1) {
        const headsPerRank = Math.floor(params.n_h / params.tp_size);
        html += `<div class="derived-dim">Heads per rank: <span>${headsPerRank}</span></div>`;
    }
    if (params.pagedAttn) {
        const ctxLens = params.seqLens.slice(0, params.B);
        const queryLens = params.queryLens.slice(0, params.B);
        const totalLens = ctxLens.map((c, i) => c + queryLens[i]);
        const blocksPerSeq = totalLens.map(s => Math.ceil(s / params.block_size));
        const totalBlocks = blocksPerSeq.reduce((a, b) => a + b, 0);
        html += `<div class="derived-dim">Total per req: <span>[${totalLens.join(', ')}]</span></div>`;
        html += `<div class="derived-dim">Blocks per req: <span>[${blocksPerSeq.join(', ')}]</span></div>`;
        html += `<div class="derived-dim">Total KV blocks: <span>${totalBlocks}</span></div>`;
    }

    d3.select('#derived').html(html);
}

// --- Update ---

function annotateGraph(graph) {
    if (params.tp_size > 1) addTpAnnotations(graph, params);
    if (params.pagedAttn) addPagedAnnotations(graph, params);
}

function update() {
    scene.selectAll('*').remove();

    if (currentVariant === 'mla') {
        renderMlaStacked();
        return;
    }

    const graphFn = GRAPH_FNS[currentVariant];
    if (!graphFn) return;
    const graph = graphFn(params);
    annotateGraph(graph);

    renderGraph(scene, graph, params,
        (op) => showDetail(op, graph, params),
        (tensor) => showTensorDetail(tensor, params)
    );

    refreshDetail([graph], params);
}

function renderMlaStacked() {
    const upprojGraph = mlaUpprojGraph(params);
    const absorbedGraph = mlaAbsorbedGraph(params);
    annotateGraph(upprojGraph);
    annotateGraph(absorbedGraph);

    // Compute shared stage positions so both paths align horizontally
    const sharedStageX = computeSharedStagePositions(upprojGraph, absorbedGraph);

    // Render prefill path
    const prefillGroup = scene.append('g');
    const onTensorClick = (tensor) => showTensorDetail(tensor, params);

    renderGraph(prefillGroup, upprojGraph, params,
        (op) => showDetail(op, upprojGraph, params),
        onTensorClick,
        scene,
        sharedStageX
    );

    const bbox1 = prefillGroup.node().getBBox();

    // Section label for prefill
    scene.append('text')
        .attr('x', bbox1.x + bbox1.width / 2)
        .attr('y', bbox1.y - 14)
        .attr('text-anchor', 'middle')
        .attr('fill', '#aaa')
        .attr('font-size', '13px')
        .attr('font-weight', '600')
        .attr('font-family', 'Inter, system-ui, sans-serif')
        .text('Prefill Path (Up-projected) \u2014 compute-bound');

    // Render decode path below
    const gap = 80;
    const offsetY = bbox1.y + bbox1.height + gap;

    const decodeGroup = scene.append('g')
        .attr('transform', `translate(0, ${offsetY})`);
    renderGraph(decodeGroup, absorbedGraph, params,
        (op) => showDetail(op, absorbedGraph, params),
        onTensorClick,
        scene,
        sharedStageX
    );

    const bbox2 = decodeGroup.node().getBBox();

    // Section label for decode
    scene.append('text')
        .attr('x', bbox1.x + bbox1.width / 2)
        .attr('y', offsetY + bbox2.y - 14)
        .attr('text-anchor', 'middle')
        .attr('fill', '#aaa')
        .attr('font-size', '13px')
        .attr('font-weight', '600')
        .attr('font-family', 'Inter, system-ui, sans-serif')
        .text('Decode Path (Absorbed) \u2014 memory-bandwidth-bound');

    refreshDetail([upprojGraph, absorbedGraph], params);
}

// --- TP annotations ---

function addTpAnnotations(graph, params) {
    const tp = params.tp_size;
    for (const t of graph.tensors) {
        if (t.dimNames && t.dimNames.includes('n_h')) {
            t.tpSharded = 'n_h';
            t.tpSize = tp;
        }
        if (t.dimNames && t.dimNames.includes('n_kv')) {
            t.tpSharded = 'n_kv';
            t.tpSize = tp;
        }
    }

    // Add all-reduce before output
    const outProjOp = graph.ops.find(o => o.id === 'out_proj');
    if (outProjOp) {
        outProjOp.desc += ` With TP=${tp}, each rank computes a partial output. An all-reduce (sum) combines results across ${tp} ranks.`;
        outProjOp.tpAllReduce = true;
        outProjOp.tpSize = tp;
    }
}

// --- Paged attention annotations ---

function addPagedAnnotations(graph, params) {
    const bs = params.block_size;
    const ctxLens = [...params.seqLens].slice(0, params.B);
    const queryLens = [...params.queryLens].slice(0, params.B);
    // Total KV length per request = cached context + new tokens
    const totalLens = ctxLens.map((c, i) => c + queryLens[i]);
    const blocksPerSeq = totalLens.map(s => Math.ceil(s / bs));
    const totalBlocks = blocksPerSeq.reduce((a, b) => a + b, 0);

    for (const t of graph.tensors) {
        if (t.type === 'mask') {
            t.pagedMask = true;
            t.seqLens = totalLens;
            t.queryLens = queryLens;
            t.ctxLens = ctxLens;
            const totalTokens = totalLens.reduce((a, b) => a + b, 0);
            t.shape = [totalTokens, totalTokens];
            t.dimNames = ['\u03a3_S', '\u03a3_S'];
            const reqDescs = ctxLens.map((c, i) => `req${i}: ${queryLens[i]} new + ${c} cached`).join(', ');
            t.desc = `Variable-length causal mask for paged attention. ${reqDescs}. Total ${totalTokens} tokens. Each request attends only within its own sequence (block-diagonal) and causally.`;
        }
        // Mark KV cache tensors with paged layout
        // MLA caches c_kv + k_r (not decompressed K, V)
        const hasCKV = graph.tensors.some(t2 => t2.id === 'c_KV');
        let isCache = false;
        if (hasCKV) {
            isCache = t.id === 'c_KV' || t.id === 'k_r';
        } else {
            isCache = t.id === 'K' || t.id === 'V' || t.id === 'K_1' || t.id === 'V_1' ||
                      t.id === 'K_g' || t.id === 'V_g';
        }
        if (isCache) {
            t.badge = 'PAGED';
            // Derive per-token dims from shape (remove B and S)
            const perTokenDims = t.shape.length === 4
                ? t.dimNames.slice(1, 2).concat(t.dimNames.slice(3))  // [n_h, d_h]
                : t.dimNames.slice(2);  // [d_c] or [d_r]
            const perTokenShape = t.shape.length === 4
                ? [t.shape[1], t.shape[3]]
                : t.shape.slice(2);
            t.pagedBlockDims = perTokenDims;
            t.pagedBlockShape = perTokenShape;
            const layoutStr = `[num_blocks, block_size, ${perTokenDims.join(', ')}]`;
            let cacheName;
            if (t.id === 'c_KV') cacheName = 'Compressed KV latent';
            else if (t.id === 'k_r') cacheName = 'Decoupled RoPE key';
            else cacheName = t.id.startsWith('K') ? 'Key' : 'Value';
            t.desc = `${cacheName} cache stored in paged blocks. ` +
                `Physical layout: ${layoutStr}. ` +
                `Each sequence maps to non-contiguous blocks via a block table. ` +
                `Block size=${bs}, blocks per seq: [${blocksPerSeq.join(', ')}], total blocks: ${totalBlocks}. ` +
                `New tokens are appended to the last block; a new block is allocated when the current one fills.`;
        }
    }
}

// --- Detail panel close ---

d3.select('#close-detail').on('click', hideDetail);

// Click empty area to deselect and close detail panel
svg.on('click', () => {
    scene.selectAll('.tensor-block').classed('selected', false).attr('filter', null);
    scene.selectAll('.op-node').classed('selected', false);
    hideDetail();
});

// --- Init ---

// Apply default preset for initial variant
const initPresetIdx = VARIANT_DEFAULT_PRESETS[currentVariant] || 0;
const initPreset = PRESETS[initPresetIdx];
if (initPreset && initPreset.variant) {
    for (const key of ['B', 'S', 'n_h', 'd_h', 'n_kv', 'd_c', 'd_r']) {
        if (initPreset[key] != null) params[key] = initPreset[key];
    }
}

buildVariantTabs();
updateVariantDesc();
buildPresets();
d3.select('#preset-select').property('value', String(initPresetIdx));
setupToggles();
buildSliders();
update();
setTimeout(fitToView, 100);

window.addEventListener('resize', () => fitToView());
