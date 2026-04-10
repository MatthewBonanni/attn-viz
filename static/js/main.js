// main.js — Entry point: D3 setup, zoom/pan, sliders, presets, toggles, update loop

import { renderGraph } from './render.js';
import { mhaGraph, gqaGraph, mqaGraph, mlaUpprojGraph, mlaAbsorbedGraph, VARIANT_DESCS, MLA_MODE_DESCS } from './graphs.js';
import { showDetail, hideDetail } from './details.js';

const GRAPH_FNS = {
    mha: mhaGraph, gqa: gqaGraph, mqa: mqaGraph,
    mla_upproj: mlaUpprojGraph, mla_absorbed: mlaAbsorbedGraph,
};

const VARIANTS = [
    { id: 'mha', label: 'MHA' },
    { id: 'gqa', label: 'GQA' },
    { id: 'mqa', label: 'MQA' },
    { id: 'mla', label: 'MLA' },
];

const MLA_MODES = [
    { id: 'upproj', label: 'Up-projected' },
    { id: 'absorbed', label: 'Absorbed' },
];

const SLIDER_DEFS = {
    B:      { label: 'B (batch)',         min: 1, max: 64,   step: 1,  default: 2 },
    S:      { label: 'S (seq length)',    min: 1, max: 2048, step: 1,  default: 8 },
    n_h:    { label: 'n_h (heads)',       min: 1, max: 128,  step: 1,  default: 8 },
    d_h:    { label: 'd_h (head dim)',    min: 1, max: 256,  step: 1,  default: 64 },
    n_kv:   { label: 'n_kv (KV heads)',   min: 1, max: 128,  step: 1,  default: 2 },
    d_c:    { label: 'd_c (latent dim)',  min: 1, max: 4096, step: 1,  default: 512 },
    tp_size:{ label: 'TP ranks',          min: 1, max: 8,    step: 1,  default: 1 },
    block_size: { label: 'Block size',    min: 1, max: 128,  step: 1,  default: 16 },
};

const VARIANT_SLIDERS = {
    mha: ['B', 'S', 'n_h', 'd_h'],
    gqa: ['B', 'S', 'n_h', 'd_h', 'n_kv'],
    mqa: ['B', 'S', 'n_h', 'd_h'],
    mla: ['B', 'S', 'n_h', 'd_h', 'd_c'],
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
    { name: 'DeepSeek R1', variant: 'mla', B: 1, S: 12, n_h: 128, d_h: 128, d_c: 512 },
];

// --- State ---

let params = {};
for (const [k, v] of Object.entries(SLIDER_DEFS)) {
    params[k] = v.default;
}
params.pagedAttn = false;
params.tp = false;
params.seqLens = [4, 8];

let currentVariant = 'mha';
let mlaMode = 'upproj';

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
                d3.select('#preset-select').property('value', '0');
                buildMlaToggle();
                updateVariantDesc();
                buildSliders();
                update();
                setTimeout(fitToView, 50);
            });
    }
}

// --- MLA sub-variant toggle ---

function buildMlaToggle() {
    const container = d3.select('#mla-mode');
    container.selectAll('*').remove();
    container.classed('visible', currentVariant === 'mla');

    if (currentVariant !== 'mla') return;

    for (const m of MLA_MODES) {
        container.append('button')
            .classed('active', m.id === mlaMode)
            .text(m.label)
            .on('click', () => {
                mlaMode = m.id;
                container.selectAll('button').classed('active', false);
                d3.select(d3.event ? d3.event.target : null);
                container.selectAll('button').each(function(_, i) {
                    d3.select(this).classed('active', MLA_MODES[i].id === mlaMode);
                });
                updateVariantDesc();
                update();
                setTimeout(fitToView, 50);
            });
    }
}

// --- Variant description ---

function updateVariantDesc() {
    let desc = VARIANT_DESCS[currentVariant] || '';
    if (currentVariant === 'mla') {
        desc += '<br><br>' + (MLA_MODE_DESCS[mlaMode] || '');
    }
    d3.select('#variant-desc').html(desc);
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
        for (const key of ['B', 'S', 'n_h', 'd_h', 'n_kv', 'd_c']) {
            if (preset[key] != null) params[key] = preset[key];
        }

        d3.select('#variant-tabs').selectAll('button').classed('active', false);
        d3.select('#variant-tabs').select(`[data-variant="${currentVariant}"]`).classed('active', true);
        buildMlaToggle();
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

    // Tensor Parallelism toggle
    d3.select('#toggle-tp').on('click', function() {
        params.tp = !params.tp;
        d3.select(this).classed('active', params.tp);
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
        .attr('max', key === 'n_kv' ? params.n_h : def.max)
        .attr('step', def.step)
        .attr('value', params[key]);

    const rangeInput = group.append('input')
        .attr('type', 'range')
        .attr('min', def.min)
        .attr('max', key === 'n_kv' ? params.n_h : def.max)
        .attr('step', def.step)
        .attr('value', params[key]);

    function onSliderChange(newVal) {
        params[key] = +newVal;

        if (key === 'n_h' && currentVariant === 'gqa') {
            if (params.n_kv > params.n_h) params.n_kv = params.n_h;
        }
        if (key === 'n_kv') {
            params.n_kv = Math.min(params.n_kv, params.n_h);
        }
        if (key === 'n_h' && params.tp) {
            if (params.tp_size > params.n_h) params.tp_size = params.n_h;
        }

        numInput.property('value', params[key]);
        rangeInput.property('value', params[key]);
        updateDerived();
        update();

        if (key === 'n_h') buildSliders();
        if ((key === 'B' || key === 'S') && params.pagedAttn) buildSeqLengthInputs();
    }

    rangeInput.on('input', function() { onSliderChange(this.value); });
    numInput.on('change', function() {
        let v = Math.max(def.min, Math.min(key === 'n_kv' ? params.n_h : def.max, +this.value || def.min));
        this.value = v;
        onSliderChange(v);
    });
}

function buildSliders() {
    // Model dimension sliders
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

    // TP sliders (separate section)
    const tpContainer = d3.select('#tp-sliders');
    tpContainer.selectAll('*').remove();
    tpContainer.classed('visible', params.tp);
    if (params.tp) {
        tpContainer.append('div').attr('class', 'slider-section-label').text('Tensor Parallelism');
        buildSlider(tpContainer, 'tp_size');
    }

    updateDerived();
}

function buildSeqLengthInputs() {
    const container = d3.select('#seq-lengths');
    container.selectAll('*').remove();
    container.classed('visible', true);

    container.append('div').style('font-size', '10px').style('color', '#666')
        .style('margin-bottom', '4px').text('Per-sequence context lengths:');

    // Ensure seqLens matches B
    while (params.seqLens.length < params.B) params.seqLens.push(params.S);
    while (params.seqLens.length > params.B) params.seqLens.pop();

    for (let i = 0; i < params.B; i++) {
        const row = container.append('div').attr('class', 'seq-row');
        row.append('span').text(`Seq ${i}:`);
        const inp = row.append('input')
            .attr('type', 'number')
            .attr('min', 1)
            .attr('max', params.S)
            .attr('step', 1)
            .property('value', Math.min(params.seqLens[i], params.S));

        inp.on('input', function() {
            const v = Math.max(1, Math.min(params.S, +this.value || 1));
            params.seqLens[i] = v;
            update();
        });
        inp.on('change', function() {
            // Clamp and sync on blur/Enter
            const v = Math.max(1, Math.min(params.S, +this.value || 1));
            params.seqLens[i] = v;
            this.value = v;
            update();
        });
    }
}

function updateDerived() {
    const D = params.n_h * params.d_h;
    let html = `<div class="derived-dim">D = n_h × d_h = <span>${D}</span></div>`;

    if (currentVariant === 'mla') {
        const ratio = (2 * params.n_h * params.d_h / params.d_c).toFixed(1);
        html += `<div class="derived-dim">KV cache reduction: <span>${ratio}×</span></div>`;
    }
    if (currentVariant === 'gqa') {
        const gpc = Math.floor(params.n_h / params.n_kv);
        const ratio = (params.n_h / params.n_kv).toFixed(1);
        html += `<div class="derived-dim">Heads per group: <span>${gpc}</span> (${ratio}× reduction)</div>`;
    }
    if (params.tp && params.tp_size > 1) {
        const headsPerRank = Math.floor(params.n_h / params.tp_size);
        html += `<div class="derived-dim">Heads per rank: <span>${headsPerRank}</span></div>`;
    }
    if (params.pagedAttn) {
        const seqLens = params.seqLens.slice(0, params.B);
        const blocksPerSeq = seqLens.map(s => Math.ceil(s / params.block_size));
        const totalBlocks = blocksPerSeq.reduce((a, b) => a + b, 0);
        html += `<div class="derived-dim">Blocks per seq: <span>[${blocksPerSeq.join(', ')}]</span></div>`;
        html += `<div class="derived-dim">Total KV blocks: <span>${totalBlocks}</span></div>`;
    }

    d3.select('#derived').html(html);
}

// --- Update ---

function update() {
    let graphKey;
    if (currentVariant === 'mla') {
        graphKey = `mla_${mlaMode}`;
    } else {
        graphKey = currentVariant;
    }

    const graphFn = GRAPH_FNS[graphKey];
    if (!graphFn) return;
    const graph = graphFn(params);

    // Add TP annotations
    if (params.tp && params.tp_size > 1) {
        addTpAnnotations(graph, params);
    }

    // Add paged attention annotations
    if (params.pagedAttn) {
        addPagedAnnotations(graph, params);
    }

    renderGraph(scene, graph, params,
        (op) => showDetail(op, graph, params),
        (tensor) => { /* highlight handled by render */ }
    );
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
    }
}

// --- Paged attention annotations ---

function addPagedAnnotations(graph, params) {
    const bs = params.block_size;
    const seqLens = [...params.seqLens].slice(0, params.B);
    const totalBlocks = seqLens.reduce((sum, s) => sum + Math.ceil(s / bs), 0);

    for (const t of graph.tensors) {
        if (t.type === 'mask') {
            t.pagedMask = true;
            t.seqLens = seqLens;
            const totalS = seqLens.reduce((a, b) => a + b, 0);
            t.shape = [totalS, totalS];
            t.dimNames = ['Σ_S', 'Σ_S'];
            t.desc = `Variable-length causal mask for paged attention. Each sequence has a different context length: [${seqLens.join(', ')}], total ${totalS} tokens. Tokens can only attend within their own sequence (block-diagonal) and causally within that sequence.`;
        }
        // Mark KV cache tensors with paged layout
        const isKV = t.id === 'K' || t.id === 'V' || t.id === 'K_1' || t.id === 'V_1' ||
                     t.id === 'K_g' || t.id === 'V_g' || t.id === 'c_KV';
        if (isKV && !t.badge) {
            t.badge = 'PAGED';
            const isK = t.id.startsWith('K');
            const blocksPerSeq = seqLens.map(s => Math.ceil(s / bs));
            t.desc = `${isK ? 'Key' : 'Value'} cache stored in paged blocks. ` +
                `Physical layout: [num_blocks, block_size, n_heads, d_h]. ` +
                `Each sequence maps to non-contiguous blocks via a block table. ` +
                `Block size=${bs}, blocks per seq: [${blocksPerSeq.join(', ')}], total blocks: ${totalBlocks}. ` +
                `New tokens are appended to the last block; a new block is allocated when the current one fills.`;
        }
    }
}

// --- Detail panel close ---

d3.select('#close-detail').on('click', hideDetail);

// Click empty area to deselect
svg.on('click', () => {
    scene.selectAll('.tensor-block').classed('selected', false);
});

// --- Init ---

buildVariantTabs();
buildMlaToggle();
updateVariantDesc();
buildPresets();
setupToggles();
buildSliders();
update();
setTimeout(fitToView, 100);

window.addEventListener('resize', () => fitToView());
