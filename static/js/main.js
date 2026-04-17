// main.js — Entry point: D3 setup, zoom/pan, sliders, presets, toggles, update loop

import { renderGraph, computeSharedStagePositions, setDimScaleContext } from './render.js';
import { mhaGraph, gqaGraph, mqaGraph, mlaUpprojGraph, mlaAbsorbedGraph, VARIANT_DESCS } from './graphs.js';
import { showDetail, showTensorDetail, showGroupDetail, hideDetail, refreshDetail } from './details/index.js';
import { computePipelineTotals, fmtNum, fmtBytes, tensorBytes, computeRooflineThreshold, GPU_SPECS } from './costs.js';

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
    B:      { label: 'B (batch)',         min: 1, max: 16,   step: 1,  default: 4, logScale: true },
    S:      { label: 'S (seq length)',     min: 1, max: 8192, step: 1,  default: 1024, logScale: true },
    S_q:    { label: 'S_q (query len)',   min: 1, max: 8192, step: 1,  default: 1024, logScale: true },
    n_h:    { label: 'n_h (heads)',       min: 1, max: 128,  step: 1,  default: 8 },
    d_h:    { label: 'd_h (head dim)',    min: 1, max: 256,  step: 1,  default: 64 },
    n_kv:   { label: 'n_kv (KV heads)',   min: 1, max: 128,  step: 1,  default: 2 },
    d_c:    { label: 'd_c (KV latent)',   min: 1, max: 4096, step: 1,  default: 512 },
    d_q:    { label: 'd_q (Q latent)',   min: 1, max: 4096, step: 1,  default: 1536 },
    d_r:    { label: 'd_r (RoPE dim)',   min: 1, max: 256,  step: 1,  default: 64 },
    tp_size:{ label: 'TP ranks',          min: 1, max: 8,    step: 1,  default: 1 },
    dp_size:{ label: 'DP ranks',          min: 1, max: 8,    step: 1,  default: 1 },
    block_size: { label: 'Block size',    min: 1, max: 128,  step: 1,  default: 16 },
};

// Find the divisor of n that is closest to target
function nearestDivisor(n, target) {
    target = Math.max(1, Math.min(n, target));
    let best = 1;
    for (let d = 1; d * d <= n; d++) {
        if (n % d === 0) {
            if (Math.abs(d - target) < Math.abs(best - target)) best = d;
            const comp = n / d;
            if (Math.abs(comp - target) < Math.abs(best - target)) best = comp;
        }
    }
    return best;
}

// Find the largest power-of-2 TP size that evenly divides n_h (max 8)
function maxTpForHeads(n_h) {
    for (let tp = 8; tp >= 1; tp >>= 1) {
        if (n_h % tp === 0) return tp;
    }
    return 1;
}

const RUNTIME_SLIDERS = ['B', 'S', 'S_q'];

const VARIANT_SLIDERS = {
    mha: ['n_h', 'd_h'],
    gqa: ['n_h', 'd_h', 'n_kv'],
    mqa: ['n_h', 'd_h'],
    mla: ['n_h', 'd_h', 'd_c', 'd_q', 'd_r'],
};

// Model presets
const PRESETS = [
    { name: 'Custom', variant: null },
    { name: 'GPT-2 (124M)', variant: 'mha', n_h: 12, d_h: 64 },
    { name: 'GPT-2 XL (1.5B)', variant: 'mha', n_h: 25, d_h: 64 },
    { name: 'Llama 3.1 8B', variant: 'gqa', n_h: 32, d_h: 128, n_kv: 8 },
    { name: 'Llama 3.1 70B', variant: 'gqa', n_h: 64, d_h: 128, n_kv: 8 },
    { name: 'Llama 3.1 405B', variant: 'gqa', n_h: 128, d_h: 128, n_kv: 8 },
    { name: 'Mistral 7B', variant: 'gqa', n_h: 32, d_h: 128, n_kv: 8 },
    { name: 'Qwen 2.5 72B', variant: 'gqa', n_h: 64, d_h: 128, n_kv: 8 },
    { name: 'Qwen 3 235B (MoE)', variant: 'gqa', n_h: 64, d_h: 192, n_kv: 4 },
    { name: 'Gemma 3 27B', variant: 'gqa', n_h: 32, d_h: 128, n_kv: 16 },
    { name: 'Phi-4 14B', variant: 'gqa', n_h: 40, d_h: 128, n_kv: 10 },
    { name: 'Command A (111B, MoE)', variant: 'gqa', n_h: 64, d_h: 128, n_kv: 8 },
    { name: 'StarCoder (15B)', variant: 'mqa', n_h: 48, d_h: 128 },
    { name: 'DeepSeek R1', variant: 'mla', n_h: 128, d_h: 128, d_c: 512, d_q: 1536, d_r: 64 },
];

// Default preset index per variant
const VARIANT_DEFAULT_PRESETS = {
    mha: 1,   // GPT-2 (124M)
    gqa: 3,   // Llama 3.1 8B
    mqa: 12, // StarCoder (15B)
    mla: 13, // DeepSeek R1
};

// --- State ---

let params = {};
for (const [k, v] of Object.entries(SLIDER_DEFS)) {
    params[k] = v.default;
}
params.pagedAttn = false;
params.flashAttn = false;
params.block_q = 128;
params.block_kv = 128;
params.splitKV = false;
params.packGQA = false;
params.seqLens = [params.S];     // per-request total S (cached + new)
params.queryLens = [params.S_q]; // per-request S_q (new query tokens)

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
                hideDetail();
                currentVariant = v.id;
                container.selectAll('button').classed('active', false);
                container.select(`[data-variant="${v.id}"]`).classed('active', true);
                // Apply default preset for this variant
                const presetIdx = VARIANT_DEFAULT_PRESETS[v.id] || 0;
                const preset = PRESETS[presetIdx];
                if (preset && preset.variant) {
                    for (const key of ['B', 'S', 'S_q', 'n_h', 'd_h', 'n_kv', 'd_c', 'd_q', 'd_r']) {
                        if (preset[key] != null) params[key] = preset[key];
                    }
                }
                d3.select('#preset-select').property('value', String(presetIdx));
                updateVariantDesc();
                buildSliders();
                update();
                setTimeout(fitToView, 300);
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

        hideDetail();
        currentVariant = preset.variant;
        for (const key of ['B', 'S', 'S_q', 'n_h', 'd_h', 'n_kv', 'd_c', 'd_q', 'd_r']) {
            if (preset[key] != null) params[key] = preset[key];
        }

        d3.select('#variant-tabs').selectAll('button').classed('active', false);
        d3.select('#variant-tabs').select(`[data-variant="${currentVariant}"]`).classed('active', true);
        updateVariantDesc();
        buildSliders();
        update();
        setTimeout(fitToView, 300);
    });
}

// --- Toggles ---

function setupToggles() {
    // Paged Attention toggle
    d3.select('#toggle-paged').on('click', function() {
        params.pagedAttn = !params.pagedAttn;
        d3.select(this).classed('active', params.pagedAttn);
        buildSliders();
        update();
    });

    // FlashAttention toggle
    d3.select('#toggle-flash').on('click', function() {
        params.flashAttn = !params.flashAttn;
        d3.select(this).classed('active', params.flashAttn);
        update();
    });
}

// --- Sliders ---

function buildSlider(container, key) {
    const def = SLIDER_DEFS[key];
    const group = container.append('div').attr('class', 'slider-group');
    const header = group.append('div').attr('class', 'slider-header');
    header.append('span').attr('class', 'dim-name').text(def.label);

    const isLog2 = key === 'tp_size' || key === 'dp_size' || key === 'block_size';
    const isLogScale = def.logScale;
    const effectiveMax = key === 'tp_size' ? maxTpForHeads(params.n_h) : def.max;

    // Log-scale helpers: slider 0–1000 maps to min..max via log interpolation
    const LOG_STEPS = 1000;
    function valToSlider(v) {
        if (!isLogScale) return isLog2 ? Math.log2(v) : v;
        const minV = Math.max(1, def.min);
        return Math.round(LOG_STEPS * Math.log(v / minV) / Math.log(effectiveMax / minV));
    }
    function sliderToVal(s) {
        if (!isLogScale) return isLog2 ? Math.pow(2, s) : s;
        const minV = Math.max(1, def.min);
        return Math.round(minV * Math.pow(effectiveMax / minV, s / LOG_STEPS));
    }

    const numInput = header.append('input')
        .attr('class', 'dim-input')
        .attr('type', 'number')
        .attr('min', def.min)
        .attr('max', effectiveMax)
        .attr('step', def.step)
        .attr('value', params[key]);

    const rangeInput = group.append('input')
        .attr('type', 'range')
        .attr('min', isLogScale ? 0 : (isLog2 ? Math.log2(Math.max(1, def.min)) : def.min))
        .attr('max', isLogScale ? LOG_STEPS : (isLog2 ? Math.log2(effectiveMax) : effectiveMax))
        .attr('step', isLogScale ? 1 : (isLog2 ? 1 : def.step))
        .attr('value', valToSlider(params[key]));

    function onSliderChange(newVal) {
        let v = +newVal;
        if (isLog2) {
            const curMax = key === 'tp_size' ? maxTpForHeads(params.n_h) : def.max;
            v = Math.max(def.min, Math.min(curMax, v));
        }
        // Snap n_h to multiples of 4 (with 1 as a special case)
        if (key === 'n_h') {
            v = v <= 2 ? 1 : Math.round(v / 4) * 4;
        }
        params[key] = v;

        // Switch preset to "Custom" when a model architecture param changes
        if (['n_h', 'd_h', 'n_kv', 'd_c', 'd_q', 'd_r'].includes(key)) {
            d3.select('#preset-select').property('value', '0');
        }

        if (key === 'n_kv' && currentVariant === 'gqa') {
            // Drag n_h up if n_kv exceeds it
            if (params.n_kv > params.n_h) {
                params.n_h = params.n_kv <= 2 ? 1 : Math.ceil(params.n_kv / 4) * 4;
                if (params.n_h < params.n_kv) params.n_h = params.n_kv;
            }
            params.n_kv = nearestDivisor(params.n_h, params.n_kv);
        }
        if (key === 'n_h' && currentVariant === 'gqa') {
            // Drag n_kv down if it exceeds n_h, then snap to nearest divisor
            params.n_kv = Math.min(params.n_kv, params.n_h);
            params.n_kv = nearestDivisor(params.n_h, params.n_kv);
        }
        if (key === 'n_h' && params.tp_size > 1) {
            const validTp = maxTpForHeads(params.n_h);
            if (params.tp_size > validTp) params.tp_size = validTp;
        }
        if (key === 'S_q' && params.S_q > params.S) {
            params.S = params.S_q;
        }
        if (key === 'S' && params.S_q > params.S) {
            params.S_q = params.S;
        }

        numInput.property('value', params[key]);
        rangeInput.property('value', valToSlider(params[key]));

        // When B changes, rebuild the S/S_q section without touching the B slider itself
        if (key === 'B') {
            rebuildSeqSliders();
            updateDerived();
            update();
            return;
        }

        updateDerived();
        update();

        // Update dependent slider UIs when their constraint or value changes
        if (key === 'n_h' || key === 'n_kv') {
            d3.selectAll('#sliders .slider-group').each(function() {
                const label = d3.select(this).select('.dim-name').text();
                const range = d3.select(this).select('input[type="range"]');
                const num = d3.select(this).select('input[type="number"]');
                if (label.includes('n_kv')) {
                    range.property('value', params.n_kv);
                    num.property('value', params.n_kv);
                } else if (label.includes('n_h') && label.includes('heads')) {
                    range.property('value', params.n_h);
                    num.property('value', params.n_h);
                }
            });
            // Update TP slider max in the parallelism section
            d3.selectAll('#parallelism-sliders .slider-group').each(function() {
                const label = d3.select(this).select('.dim-name').text();
                if (label.includes('TP')) {
                    const tpMax = maxTpForHeads(params.n_h);
                    d3.select(this).select('input[type="range"]').attr('max', Math.log2(tpMax));
                    d3.select(this).select('input[type="number"]').attr('max', tpMax);
                }
            });
        }
        // When B changes, the per-request sliders need rebuilding
        // (S/S_q sync for B=1 is handled in buildPerRequestSlider's applyValue)
    }

    rangeInput.on('input', function() {
        onSliderChange(sliderToVal(+this.value));
    });
    numInput.on('change', function() {
        const effMax = key === 'tp_size' ? maxTpForHeads(params.n_h) : def.max;
        let v = +this.value || def.min;
        if (isLog2) {
            const prev = params[key];
            if (v > prev) {
                v = Math.pow(2, Math.floor(Math.log2(prev)) + 1);
            } else if (v < prev) {
                v = Math.pow(2, Math.ceil(Math.log2(prev)) - 1);
            }
        }
        v = Math.max(def.min, Math.min(effMax, v));
        this.value = v;
        onSliderChange(v);
    });
}

// Rebuild only the S/S_q slider area (below the B slider) without touching B itself
function rebuildSeqSliders() {
    const rtContainer = d3.select('#runtime-sliders');
    rtContainer.selectAll('.seq-slider-area').remove();
    const area = rtContainer.append('div').attr('class', 'seq-slider-area');
    buildSeqLengthInputs(area);
}

function buildSliders() {
    // Runtime sliders — always visible, variant-independent
    const rtContainer = d3.select('#runtime-sliders');
    rtContainer.selectAll('*').remove();

    // B slider always shown
    buildSlider(rtContainer, 'B');

    // S/S_q area — always use compact per-request sliders
    const area = rtContainer.append('div').attr('class', 'seq-slider-area');
    buildSeqLengthInputs(area);

    // Model architecture sliders (variant-specific)
    const dimContainer = d3.select('#sliders');
    dimContainer.selectAll('*').remove();
    for (const key of VARIANT_SLIDERS[currentVariant]) {
        if (currentVariant === 'gqa' && key === 'n_h') {
            // Render n_h and n_kv side by side
            const pair = dimContainer.append('div').attr('class', 'slider-pair');
            buildSlider(pair, 'n_h');
            buildSlider(pair, 'n_kv');
            continue;
        }
        if (currentVariant === 'gqa' && key === 'n_kv') continue; // already built above
        buildSlider(dimContainer, key);
    }

    // Parallelism sliders (TP + DP)
    const parContainer = d3.select('#parallelism-sliders');
    parContainer.selectAll('*').remove();
    const parPair = parContainer.append('div').attr('class', 'slider-pair');
    buildSlider(parPair, 'tp_size');
    buildSlider(parPair, 'dp_size');

    // Paged attention sliders (separate section)
    const pagedContainer = d3.select('#paged-sliders');
    pagedContainer.selectAll('*').remove();
    pagedContainer.classed('visible', params.pagedAttn);
    if (params.pagedAttn) {
        pagedContainer.append('div').attr('class', 'slider-section-label').text('Paged Attention');
        buildSlider(pagedContainer, 'block_size');
    }

    // Hide old seq-lengths container (no longer used)
    d3.select('#seq-lengths').classed('visible', false);

    updateDerived();
}

function buildSeqLengthInputs(container) {
    // Ensure arrays match B
    while (params.seqLens.length < params.B) params.seqLens.push(params.S);
    while (params.seqLens.length > params.B) params.seqLens.pop();
    while (params.queryLens.length < params.B) params.queryLens.push(params.S_q || 1);
    while (params.queryLens.length > params.B) params.queryLens.pop();
    for (let i = 0; i < params.B; i++) {
        if (params.queryLens[i] > params.seqLens[i]) {
            params.seqLens[i] = params.queryLens[i];
        }
    }

    for (let i = 0; i < params.B; i++) {
        const reqLabel = container.append('div').attr('class', 'slider-section-label')
            .style('margin-top', '4px').style('margin-bottom', '0').style('padding-top', '4px')
            .text(`Req ${i}`);
        const typeLabel = reqLabel.append('span')
            .style('margin-left', '6px').style('font-weight', '400');
        updateReqTypeLabel(i, typeLabel);

        buildPerRequestSlider(container, i, 'S', params.seqLens, typeLabel);
        buildPerRequestSlider(container, i, 'S_q', params.queryLens, typeLabel);
    }
}

function updateReqTypeLabel(i, typeLabel) {
    const sq = params.queryLens[i];
    const s = params.seqLens[i];
    const type = sq === 1 ? 'decode' : sq >= s ? 'prefill' : sq < 16 ? 'spec decode' : 'extend';
    const color = sq === 1 ? '#3498db' : sq >= s ? '#f39c12' : sq < 16 ? '#8e44ad' : '#2ecc71';
    typeLabel.style('color', color).text(type);
}

function buildPerRequestSlider(container, reqIdx, dimKey, arr, typeLabel) {
    const def = SLIDER_DEFS[dimKey];
    const row = container.append('div').attr('class', 'slider-group')
        .attr('data-req', reqIdx).attr('data-dim', dimKey)
        .style('display', 'flex').style('align-items', 'center')
        .style('gap', '6px').style('margin-bottom', '2px');

    row.append('span').attr('class', 'dim-name')
        .style('width', '20px').style('flex-shrink', '0')
        .style('font-size', '11px').style('text-align', 'right')
        .text(dimKey === 'S_q' ? 'S_q' : 'S');

    const LOG_STEPS = 1000;
    const effectiveMax = def.max;
    function valToSlider(v) {
        const minV = Math.max(1, def.min);
        return Math.round(LOG_STEPS * Math.log(v / minV) / Math.log(effectiveMax / minV));
    }
    function sliderToVal(s) {
        const minV = Math.max(1, def.min);
        return Math.round(minV * Math.pow(effectiveMax / minV, s / LOG_STEPS));
    }

    const rangeInput = row.append('input')
        .attr('type', 'range')
        .attr('min', 0).attr('max', LOG_STEPS).attr('step', 1)
        .property('value', valToSlider(arr[reqIdx]))
        .style('flex', '1');

    const numInput = row.append('input')
        .attr('class', 'dim-input')
        .attr('type', 'number')
        .attr('min', def.min).attr('max', effectiveMax)
        .attr('step', def.step)
        .property('value', arr[reqIdx]);

    function syncPairedSlider() {
        // Find and update the paired slider's DOM elements
        const pairedDim = dimKey === 'S_q' ? 'S' : 'S_q';
        const pairedVal = dimKey === 'S_q' ? params.seqLens[reqIdx] : params.queryLens[reqIdx];
        const paired = container.select(`[data-req="${reqIdx}"][data-dim="${pairedDim}"]`);
        if (!paired.empty()) {
            paired.select('input[type="range"]').property('value', valToSlider(pairedVal));
            paired.select('input[type="number"]').property('value', pairedVal);
        }
    }

    function applyValue(v) {
        arr[reqIdx] = v;
        // When B=1, keep global S/S_q in sync with the per-request values
        if (params.B === 1) {
            if (dimKey === 'S') params.S = v;
            if (dimKey === 'S_q') params.S_q = v;
        }
        if (dimKey === 'S_q' && params.queryLens[reqIdx] > params.seqLens[reqIdx]) {
            params.seqLens[reqIdx] = params.queryLens[reqIdx];
            if (params.B === 1) params.S = params.seqLens[reqIdx];
            syncPairedSlider();
        }
        if (dimKey === 'S' && params.queryLens[reqIdx] > params.seqLens[reqIdx]) {
            params.queryLens[reqIdx] = params.seqLens[reqIdx];
            if (params.B === 1) params.S_q = params.queryLens[reqIdx];
            syncPairedSlider();
        }
        numInput.property('value', arr[reqIdx]);
        rangeInput.property('value', valToSlider(arr[reqIdx]));
        updateReqTypeLabel(reqIdx, typeLabel);
        updateDerived();
        update();
    }

    rangeInput.on('input', function() {
        applyValue(sliderToVal(+this.value));
    });
    numInput.on('change', function() {
        let v = +this.value || def.min;
        v = Math.max(def.min, Math.min(effectiveMax, v));
        this.value = v;
        applyValue(v);
    });
}

function updateDerived() {
    // Sync seqLens/queryLens arrays with B
    while (params.seqLens.length < params.B) params.seqLens.push(params.S);
    while (params.seqLens.length > params.B) params.seqLens.pop();
    while (params.queryLens.length < params.B) params.queryLens.push(params.S_q || 1);
    while (params.queryLens.length > params.B) params.queryLens.pop();
    // When B=1, keep synced to global S/S_q sliders
    if (params.B === 1) {
        params.seqLens[0] = params.S;
        params.queryLens[0] = params.S_q;
    }
    // Enforce S_q <= S per request
    for (let i = 0; i < params.B; i++) {
        if (params.queryLens[i] > params.seqLens[i]) {
            params.seqLens[i] = params.queryLens[i];
        }
    }
    // Compute totals across all requests
    params.sumSq = params.queryLens.slice(0, params.B).reduce((a, b) => a + b, 0);
    params.sumS = params.seqLens.slice(0, params.B).reduce((a, b) => a + b, 0);

    // Model architecture derived values
    const D = params.n_h * params.d_h;
    let archHtml = `<div class="derived-dim">D = n_h × d_h = <span>${D}</span></div>`;
    if (currentVariant === 'mla') {
        const dr = params.d_r || 64;
        const totalCache = params.d_c + dr;
        const ratio = (2 * params.n_h * params.d_h / totalCache).toFixed(1);
        archHtml += `<div class="derived-dim">Cache per token: <span>d_c + d_r = ${totalCache}</span></div>`;
        archHtml += `<div class="derived-dim">KV cache reduction: <span>${ratio}×</span></div>`;
    }
    if (currentVariant === 'gqa') {
        const gpc = Math.floor(params.n_h / params.n_kv);
        const ratio = (params.n_h / params.n_kv).toFixed(1);
        archHtml += `<div class="derived-dim">Heads per group: <span>${gpc}</span> (${ratio}× reduction)</div>`;
    }
    d3.select('#derived').html(archHtml);

    // Parallelism derived values
    let parHtml = '';
    if (params.tp_size > 1) {
        const headsPerRank = Math.floor(params.n_h / params.tp_size);
        parHtml += `<div class="derived-dim">Heads per TP rank: <span>${headsPerRank}</span></div>`;
    }
    if (params.dp_size > 1 && params.dp_size > params.B) {
        const effectiveDp = Math.min(params.dp_size, params.B);
        const idleRanks = params.dp_size - effectiveDp;
        parHtml += `<div class="derived-dim" style="color:#e67e22">Effective DP: <span>${effectiveDp}</span> (${idleRanks} idle rank${idleRanks !== 1 ? 's' : ''})</div>`;
    }
    if (params.dp_size > 1 || params.tp_size > 1) {
        const totalRanks = Math.max(1, params.dp_size) * Math.max(1, params.tp_size);
        if (totalRanks > 1) {
            parHtml += `<div class="derived-dim">Total ranks (DP\u00d7TP): <span>${totalRanks}</span></div>`;
        }
    }
    d3.select('#derived-parallelism').html(parHtml);

    // Runtime derived values
    let rtHtml = '';
    if (params.B > 1) {
        rtHtml += `<div class="derived-dim">\u03a3S_q = <span>${params.sumSq}</span> (\u03a3S = ${params.sumS})</div>`;
    }
    if (params.pagedAttn) {
        const sLens = params.seqLens.slice(0, params.B);
        const blocksPerSeq = sLens.map(s => Math.ceil(s / params.block_size));
        const totalBlocks = blocksPerSeq.reduce((a, b) => a + b, 0);
        rtHtml += `<div class="derived-dim">Blocks per req: <span>[${blocksPerSeq.join(', ')}]</span></div>`;
        rtHtml += `<div class="derived-dim">Total KV blocks: <span>${totalBlocks}</span></div>`;
    }
    d3.select('#derived-runtime').html(rtHtml);

    const sharded = params.tp_size > 1 || params.dp_size > 1;
    d3.select('#legend-qkv-separate').style('display', sharded ? 'none' : null);
    d3.select('#legend-qkv-unified').style('display', sharded ? null : 'none');
}

// --- Pipeline stats overlay ---

function fmtTime(seconds) {
    if (seconds < 1e-6) return (seconds * 1e9).toFixed(1) + ' ns';
    if (seconds < 1e-3) return (seconds * 1e6).toFixed(1) + ' \u00b5s';
    if (seconds < 1) return (seconds * 1e3).toFixed(2) + ' ms';
    return seconds.toFixed(3) + ' s';
}

// Compute the crossover S_q between MHA-style and MQA-style MLA, for the current S.
// MQA avoids K/V expansion (lower S_q-independent cost) but works in d_c space
// (higher per-S_q cost since d_c >> d_h). So MQA wins at small S_q (decode) and
// MHA wins at large S_q (prefill). We find where the two lines cross.
function computeMlaCrossover(params) {
    // Use the max context length across all requests in the batch.
    const S = Math.max(...(params.seqLens || []).slice(0, params.B || 1), params.S || 1);
    // Evaluate at two S_q points to extract the linear relationship.
    // Use S_q = 1 and S_q = max(2, min(S, 201)). When S=1, bump effective S
    // to 2 so we have two valid sample points.
    // Force B=1 and clear batch totals so swept S_q actually takes effect
    // (graph functions prefer sumSq/sumS over S_q/S).
    const effS = Math.max(S, 2);
    const sqLo = 1;
    const sqHi = Math.min(effS, 201);
    const base = { ...params, B: 1, sumSq: undefined, sumS: undefined };
    const p1 = { ...base, S: effS, S_q: sqLo };
    const p2 = { ...base, S: effS, S_q: sqHi };
    const dSq = sqHi - sqLo;

    const g_up1 = mlaUpprojGraph(p1);
    const g_up2 = mlaUpprojGraph(p2);
    const g_ab1 = mlaAbsorbedGraph(p1);
    const g_ab2 = mlaAbsorbedGraph(p2);
    // Apply FlashAttention annotations so SRAM-only tensors are zeroed in cost
    if (params.flashAttn) {
        addFlashAttnAnnotations(g_up1, p1);
        addFlashAttnAnnotations(g_up2, p2);
        addFlashAttnAnnotations(g_ab1, p1);
        addFlashAttnAnnotations(g_ab2, p2);
    }
    const upproj1 = computePipelineTotals(g_up1);
    const upproj2 = computePipelineTotals(g_up2);
    const absorbed1 = computePipelineTotals(g_ab1);
    const absorbed2 = computePipelineTotals(g_ab2);

    // Bytes as a function of S_q: total = slope * S_q + intercept
    const byteSlopeUpproj = (upproj2.totalBytes - upproj1.totalBytes) / dSq;
    const byteSlopeAbsorbed = (absorbed2.totalBytes - absorbed1.totalBytes) / dSq;
    let byteInterceptUpproj = upproj1.totalBytes - byteSlopeUpproj * sqLo;
    let byteInterceptAbsorbed = absorbed1.totalBytes - byteSlopeAbsorbed * sqLo;

    // Weight amortization: weights are loaded once per batch from HBM, so the
    // per-request weight transfer cost is weightBytes/B. Adjust the byte
    // intercepts (where weight reads live) to reflect this.
    const B = params.B || 1;
    if (B > 1) {
        function graphWeightBytes(graph) {
            return graph.tensors
                .filter(t => t.type === 'weight')
                .reduce((sum, t) => sum + tensorBytes(t.shape), 0);
        }
        const wUp = graphWeightBytes(g_up1);
        const wAb = graphWeightBytes(g_ab1);
        // Remove the (1 - 1/B) fraction of weight bytes that's shared
        byteInterceptUpproj -= wUp * (1 - 1 / B);
        byteInterceptAbsorbed -= wAb * (1 - 1 / B);
    }

    // FLOPs as a function of S_q (not affected by weight amortization —
    // each request still requires the same compute)
    const flopSlopeUpproj = (upproj2.totalFlops - upproj1.totalFlops) / dSq;
    const flopSlopeAbsorbed = (absorbed2.totalFlops - absorbed1.totalFlops) / dSq;
    const flopInterceptUpproj = upproj1.totalFlops - flopSlopeUpproj * sqLo;
    const flopInterceptAbsorbed = absorbed1.totalFlops - flopSlopeAbsorbed * sqLo;

    // Crossover: where upproj cost = absorbed cost, solving for S_q.
    // diff(S_q) = (interceptAbsorbed - interceptUpproj) + (slopeAbsorbed - slopeUpproj) * S_q
    // MQA wins when diff < 0. Crossover at diff = 0 → S_q = interceptDiff / slopeDiff
    // where slopeDiff = slopeUpproj - slopeAbsorbed.
    // slopeDiff > 0: MQA slope lower → MQA wins above crossover (unusual)
    // slopeDiff < 0: MQA slope higher → MQA wins below crossover (typical: decode)
    function findCrossover(interceptUp, interceptAbs, slopeUp, slopeAbs) {
        const iDiff = interceptAbs - interceptUp;
        const sDiff = slopeUp - slopeAbs;
        if (Math.abs(sDiff) < 1e-6) return null; // parallel lines
        const sq = iDiff / sDiff;
        if (sq < 1) return null; // crossover below S_q=1, one always wins
        return { sq: Math.round(sq), mqaWinsBelow: sDiff < 0 };
    }

    const bytesCrossover = findCrossover(byteInterceptUpproj, byteInterceptAbsorbed,
        byteSlopeUpproj, byteSlopeAbsorbed);
    const flopsCrossover = findCrossover(flopInterceptUpproj, flopInterceptAbsorbed,
        flopSlopeUpproj, flopSlopeAbsorbed);

    // Determine which always wins when there's no crossover
    function alwaysWinner(interceptUp, interceptAbs, slopeUp, slopeAbs) {
        // Check at S_q=1: whichever is lower there wins everywhere
        return (interceptAbs + slopeAbs) < (interceptUp + slopeUp) ? 'mqa' : 'mha';
    }

    return {
        bytesCrossover,
        flopsCrossover,
        bytesAlwaysWinner: !bytesCrossover ? alwaysWinner(byteInterceptUpproj, byteInterceptAbsorbed,
            byteSlopeUpproj, byteSlopeAbsorbed) : null,
        flopsAlwaysWinner: !flopsCrossover ? alwaysWinner(flopInterceptUpproj, flopInterceptAbsorbed,
            flopSlopeUpproj, flopSlopeAbsorbed) : null,
        S, B,
        // per-query-token costs (slope vs S_q)
        bytesPerQueryUpproj: byteSlopeUpproj,
        bytesPerQueryAbsorbed: byteSlopeAbsorbed,
        flopsPerQueryUpproj: flopSlopeUpproj,
        flopsPerQueryAbsorbed: flopSlopeAbsorbed,
        // S-dependent fixed costs (intercept: weights + cache ops, independent of S_q)
        fixedBytesUpproj: byteInterceptUpproj,
        fixedBytesAbsorbed: byteInterceptAbsorbed,
        fixedFlopsUpproj: flopInterceptUpproj,
        fixedFlopsAbsorbed: flopInterceptAbsorbed,
    };
}

function updateStatsOverlay(graphs, labels, crossover) {
    const container = d3.select('#stats-overlay');
    container.html('');

    const gpuKeys = Object.keys(GPU_SPECS);

    graphs.forEach((graph, i) => {
        const totals = computePipelineTotals(graph);

        if (labels && labels[i]) {
            container.append('div')
                .style('font-weight', '600')
                .style('color', '#bbb')
                .style('margin-bottom', '6px')
                .style('font-size', '11px')
                .text(labels[i]);
        }

        // Totals row
        const totalsRow = container.append('div')
            .style('display', 'flex')
            .style('gap', '12px')
            .style('align-items', 'center')
            .style('flex-wrap', 'wrap')
            .style('margin-bottom', '8px');

        if (totals.totalFlops > 0) {
            totalsRow.append('span').html(
                `<span style="color:#888">FLOPs:</span> <span style="color:#7c8cf8;font-weight:600">${fmtNum(totals.totalFlops)}</span>`
            );
        }
        totalsRow.append('span').html(
            `<span style="color:#888">Read:</span> <span style="color:#aaa">${fmtBytes(totals.totalReadBytes)}</span>`
        );
        totalsRow.append('span').html(
            `<span style="color:#888">Write:</span> <span style="color:#aaa">${fmtBytes(totals.totalWriteBytes)}</span>`
        );

        // GPU table
        const table = container.append('table')
            .style('width', '100%')
            .style('border-collapse', 'collapse')
            .style('font-size', '11px');

        // Header
        const thead = table.append('tr');
        thead.append('td').style('color', '#666').style('padding', '2px 6px 2px 0').text('GPU');
        thead.append('td').style('color', '#666').style('padding', '2px 6px').style('text-align', 'right').text('Compute');
        thead.append('td').style('color', '#666').style('padding', '2px 6px').style('text-align', 'right').text('Memory');
        thead.append('td').style('color', '#666').style('padding', '2px 6px').style('text-align', 'right').text('Bound');
        thead.append('td').style('color', '#666').style('padding', '2px 0 2px 6px').style('text-align', 'right').text('Time');

        for (const key of gpuKeys) {
            const gpu = GPU_SPECS[key];
            const computeTime = totals.totalFlops > 0 ? totals.totalFlops / (gpu.peakTFLOPS_bf16 * 1e12) : 0;
            const memTime = totals.totalBytes / (gpu.bandwidthTBs * 1e12);
            const bottleneck = totals.totalFlops === 0 ? 'MEM'
                : computeTime >= memTime ? 'COMPUTE' : 'MEM';
            const bottleneckTime = Math.max(computeTime, memTime);
            const bnColor = bottleneck === 'COMPUTE' ? '#2ecc71' : '#e74c3c';

            const row = table.append('tr');
            row.append('td').style('color', '#bbb').style('padding', '2px 6px 2px 0').style('font-weight', '500').text(key);
            row.append('td').style('color', '#aaa').style('padding', '2px 6px').style('text-align', 'right')
                .text(computeTime > 0 ? fmtTime(computeTime) : '\u2014');
            row.append('td').style('color', '#aaa').style('padding', '2px 6px').style('text-align', 'right')
                .text(fmtTime(memTime));
            row.append('td').style('padding', '2px 6px').style('text-align', 'right')
                .style('color', bnColor).style('font-weight', '600')
                .text(bottleneck);
            row.append('td').style('color', '#7c8cf8').style('padding', '2px 0 2px 6px').style('text-align', 'right')
                .style('font-weight', '600')
                .text(fmtTime(bottleneckTime));
        }

        if (i < graphs.length - 1) {
            container.append('div')
                .style('border-top', '1px solid #2a2d3a')
                .style('margin', '8px 0');
        }
    });

    // Show crossover analysis for MLA dual-graph view
    if (crossover != null && graphs.length === 2) {
        container.append('div')
            .style('border-top', '1px solid #2a2d3a')
            .style('margin', '8px 0');

        const heading = container.append('div')
            .style('font-weight', '600')
            .style('color', '#bbb')
            .style('font-size', '11px')
            .style('margin-bottom', '4px')
            .text(`Crossover analysis (S\u2009=\u2009${crossover.S}, B\u2009=\u2009${crossover.B})`);

        container.append('div')
            .style('font-size', '10px')
            .style('color', '#888')
            .style('margin-bottom', '6px')
            .style('font-style', 'italic')
            .text('cost(S_q) = m \u00b7 S_q + b');

        const grid = container.append('div')
            .style('display', 'grid')
            .style('grid-template-columns', 'auto auto auto')
            .style('gap', '1px 8px')
            .style('font-size', '10px')
            .style('margin-bottom', '6px');

        function addRow(label, v1, v2, opts) {
            const highlight = opts?.highlight;
            grid.append('span').style('color', '#888').text(label);
            const s1 = grid.append('span').style('text-align', 'right');
            const s2 = grid.append('span').style('text-align', 'right');
            if (highlight) {
                const better = v1 < v2 ? s1 : s2;
                const worse = v1 < v2 ? s2 : s1;
                better.style('color', '#2ecc71');
                worse.style('color', '#aaa');
            } else {
                s1.style('color', '#aaa');
                s2.style('color', '#aaa');
            }
            s1.text(opts?.fmt(v1) || v1);
            s2.text(opts?.fmt(v2) || v2);
        }

        // Header
        grid.append('span').style('color', '#666').text('');
        grid.append('span').style('color', '#666').style('text-align', 'right').text('MHA-st.');
        grid.append('span').style('color', '#666').style('text-align', 'right').text('MQA-st.');

        // Intercept: costs independent of S_q (weights, K/V expansion)
        addRow('b (xfer)', crossover.fixedBytesUpproj, crossover.fixedBytesAbsorbed,
            { fmt: fmtBytes, highlight: true });
        addRow('b (FLOPs)', crossover.fixedFlopsUpproj, crossover.fixedFlopsAbsorbed,
            { fmt: fmtNum, highlight: true });

        // Spacer
        grid.append('span').html('&nbsp;');
        grid.append('span');
        grid.append('span');

        // Slope: marginal cost per query token
        addRow('m (xfer)', crossover.bytesPerQueryUpproj, crossover.bytesPerQueryAbsorbed,
            { fmt: v => fmtBytes(v) + '/q', highlight: true });
        addRow('m (FLOPs)', crossover.flopsPerQueryUpproj, crossover.flopsPerQueryAbsorbed,
            { fmt: v => fmtNum(v) + '/q', highlight: true });

        // Crossover results — highlighted card
        const resultBox = container.append('div')
            .style('margin-top', '8px')
            .style('padding', '8px 10px')
            .style('border', '1px solid #3a3d4a')
            .style('border-radius', '6px')
            .style('background', '#1a1c2a');

        const atCtx = `S\u2009=\u2009${crossover.S}, B\u2009=\u2009${crossover.B}` +
            (crossover.B > 1 ? ' (weights amortized)' : '');

        function crossoverLine(parent, label, xover, alwaysWinner) {
            const line = parent.append('div')
                .style('font-size', '11px')
                .style('line-height', '1.6');
            if (xover) {
                const dir = xover.mqaWinsBelow ? '\u2264' : '\u2265';
                line.html(`<span style="color:#999">${label}:</span> ` +
                    `<span style="color:#ccc">MQA-style wins at</span> ` +
                    `<span style="color:#7c8cf8;font-weight:600">S_q ${dir} ${xover.sq}</span>`);
            } else {
                const winner = alwaysWinner === 'mqa' ? 'MQA-style' : 'MHA-style';
                const color = alwaysWinner === 'mqa' ? '#2ecc71' : '#e67e22';
                line.html(`<span style="color:#999">${label}:</span> ` +
                    `<span style="color:${color};font-weight:600">${winner} wins for any S_q</span>`);
            }
        }

        crossoverLine(resultBox, 'Xfer', crossover.bytesCrossover, crossover.bytesAlwaysWinner);
        crossoverLine(resultBox, 'FLOPs', crossover.flopsCrossover, crossover.flopsAlwaysWinner);

        resultBox.append('div')
            .style('font-size', '9px')
            .style('color', '#666')
            .style('margin-top', '4px')
            .text(atCtx);
    }
}

// --- Update ---

function annotateGraph(graph) {
    if (params.B > 1) addMultiRequestAnnotations(graph, params);
    if (params.tp_size > 1) addTpAnnotations(graph, params);
    if (params.dp_size > 1) addDpAnnotations(graph, params);
    if (params.pagedAttn) addPagedAnnotations(graph, params);
    if (params.flashAttn) addFlashAttnAnnotations(graph, params);
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
    setDimScaleContext(graph);

    renderGraph(scene, graph, params,
        (op) => showDetail(op, graph, params),
        (tensor) => showTensorDetail(tensor, params),
        undefined, undefined,
        (group) => showGroupDetail(group)
    );

    updateStatsOverlay([graph]);

    refreshDetail([graph], params);
}

function renderMlaStacked() {
    const upprojGraph = mlaUpprojGraph(params);
    const absorbedGraph = mlaAbsorbedGraph(params);
    annotateGraph(upprojGraph);
    annotateGraph(absorbedGraph);
    setDimScaleContext(upprojGraph, absorbedGraph);

    // Compute shared stage positions so both paths align horizontally
    const sharedStageX = computeSharedStagePositions(upprojGraph, absorbedGraph);

    // Render prefill path
    const prefillGroup = scene.append('g');
    const onTensorClick = (tensor) => showTensorDetail(tensor, params);
    const onGroupClick = (group) => showGroupDetail(group);

    renderGraph(prefillGroup, upprojGraph, params,
        (op) => showDetail(op, upprojGraph, params),
        onTensorClick,
        scene,
        sharedStageX,
        onGroupClick
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
        .text('MHA-style (up-projected)');


    // Render decode path below
    const gap = 80;
    const offsetY = bbox1.y + bbox1.height + gap;

    const decodeGroup = scene.append('g')
        .attr('transform', `translate(0, ${offsetY})`);
    renderGraph(decodeGroup, absorbedGraph, params,
        (op) => showDetail(op, absorbedGraph, params),
        onTensorClick,
        scene,
        sharedStageX,
        onGroupClick
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
        .text('MQA-style (absorbed)');

    const crossover = computeMlaCrossover(params);
    updateStatsOverlay([upprojGraph, absorbedGraph], ['MHA-style (up-projected)', 'MQA-style (absorbed)'], crossover);

    refreshDetail([upprojGraph, absorbedGraph], params);
}

// --- TP annotations ---

function addTpAnnotations(graph, params) {
    const tp = params.tp_size;
    const tensorMap = {};
    for (const t of graph.tensors) tensorMap[t.id] = t;

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

    // Annotate 2D weight matrices and their activation inputs/outputs with TP sharding
    for (const op of graph.ops) {
        if (op.type === 'compress') continue;
        if (op.type !== 'matmul' && op.type !== 'decompress') continue;

        for (const inputId of op.inputs) {
            const t = tensorMap[inputId];
            if (!t || t.type !== 'weight' || t.shape.length !== 2 || t.tpSharded) continue;

            if (op.tpAllReduce) {
                t.tpSharded = true;
                t.tpSize = tp;
                t.tpDim = 0;
            } else {
                const directOut = tensorMap[op.output];
                const directHasHeads = directOut && directOut.dimNames &&
                    (directOut.dimNames.includes('n_h') || directOut.dimNames.includes('n_kv'));

                let nextHasHeads = false;
                if (!directHasHeads) {
                    nextHasHeads = graph.ops.some(o =>
                        o.inputs.includes(op.output) && tensorMap[o.output] &&
                        tensorMap[o.output].dimNames &&
                        (tensorMap[o.output].dimNames.includes('n_h') ||
                         tensorMap[o.output].dimNames.includes('n_kv'))
                    );
                }

                if ((directHasHeads || nextHasHeads) && t.shape[1] % tp === 0) {
                    t.tpSharded = true;
                    t.tpSize = tp;
                    t.tpDim = 1;
                }
            }
        }
    }

    // Propagate TP to 2D activation tensors connected to TP-sharded matmuls
    for (const op of graph.ops) {
        if (op.type === 'compress') continue;
        if (op.type !== 'matmul' && op.type !== 'decompress') continue;

        const weightInput = op.inputs.map(id => tensorMap[id]).find(t => t && t.type === 'weight' && t.tpSharded);
        if (!weightInput) continue;

        if (weightInput.tpDim === 1) {
            // Column-parallel: output's last dim is sharded
            const out = tensorMap[op.output];
            if (out && out.shape.length === 2 && !out.tpSharded) {
                out.tpSharded = true;
                out.tpSize = tp;
                out.tpDim = 1;
            }
        } else if (weightInput.tpDim === 0) {
            // Row-parallel: the non-weight input's last dim is sharded
            for (const inputId of op.inputs) {
                const t = tensorMap[inputId];
                if (t && t.type !== 'weight' && t.shape.length === 2 && !t.tpSharded) {
                    t.tpSharded = true;
                    t.tpSize = tp;
                    t.tpDim = 1;
                }
            }
        }
    }

    // Propagate TP through reshape/view ops (2D TP-sharded → 3D/4D or vice versa)
    for (const op of graph.ops) {
        if (op.type !== 'reshape') continue;
        const inT = tensorMap[op.inputs[0]];
        const outT = tensorMap[op.output];
        if (!inT || !outT) continue;

        if (inT.tpSharded && !outT.tpSharded && outT.shape.length === 2) {
            outT.tpSharded = true;
            outT.tpSize = tp;
            outT.tpDim = 1;
        } else if (outT.tpSharded && !inT.tpSharded && inT.shape.length === 2) {
            inT.tpSharded = true;
            inT.tpSize = tp;
            inT.tpDim = 1;
        }
    }
}

// --- DP annotations ---

function addDpAnnotations(graph, params) {
    const dp = params.dp_size;
    const B = params.B || 1;
    const effectiveDp = Math.min(dp, B);
    const sLens = params.seqLens ? params.seqLens.slice(0, B) : [params.S];
    const sqLens = params.queryLens ? params.queryLens.slice(0, B) : [params.S_q];

    for (const t of graph.tensors) {
        if (t.type === 'weight' || t.type === 'mask') continue;
        if (!t.dimNames) continue;

        let seqDim = null;
        for (const d of t.dimNames) {
            if (d === 'S_q' || d === '\u03a3S_q' || d === 'S' || d === '\u03a3S') {
                seqDim = d;
                break;
            }
        }
        if (!seqDim) continue;

        t.dpSharded = seqDim;
        t.dpSize = effectiveDp;
        if (dp > B) t.dpIdleRanks = dp - B;

        if (B > 1) {
            const lens = (seqDim === 'S_q' || seqDim === '\u03a3S_q') ? sqLens : sLens;
            const total = lens.reduce((a, b) => a + b, 0);
            const dpFracs = [0];
            let cum = 0;
            for (let r = 0; r < B; r++) {
                cum += lens[r];
                const curRank = Math.floor(r * effectiveDp / B);
                const nextRank = (r + 1 < B) ? Math.floor((r + 1) * effectiveDp / B) : effectiveDp;
                if (nextRank > curRank) {
                    dpFracs.push(cum / total);
                }
            }
            t.dpBoundaryFracs = dpFracs;
        }
    }
}

// --- FlashAttention annotations ---

// Intermediate tensors that stay in SRAM (never materialized in HBM)
const FLASH_SRAM_TENSORS = new Set(['scores', 'attn', 's_content', 's_rope']);

// Ops that get fused into the FlashAttention kernel
const FLASH_FUSED_OPS_STANDARD = new Set(['qkt', 'masking', 'attn_v']);
const FLASH_FUSED_OPS_MLA_UPPROJ = new Set(['content_qk', 'rope_qk', 'add_scores', 'masking', 'attn_v']);
const FLASH_FUSED_OPS_MLA_ABSORBED = new Set(['content_qk', 'rope_qk', 'add_scores', 'masking', 'latent_attn_v']);

function addFlashAttnAnnotations(graph, params) {
    // Identify which ops to fuse based on what's present in the graph
    const opIds = new Set(graph.ops.map(o => o.id));
    let fusedOpIds;
    if (opIds.has('latent_attn_v')) {
        fusedOpIds = FLASH_FUSED_OPS_MLA_ABSORBED;
    } else if (opIds.has('content_qk')) {
        fusedOpIds = FLASH_FUSED_OPS_MLA_UPPROJ;
    } else {
        fusedOpIds = FLASH_FUSED_OPS_STANDARD;
    }

    // Find the fused ops and extract external inputs/output
    const fusedOps = graph.ops.filter(o => fusedOpIds.has(o.id));
    if (fusedOps.length === 0) return;

    // Collect all tensor IDs that are internal to the fused region
    const fusedOutputIds = new Set(fusedOps.map(o => o.output));
    const fusedInputIds = new Set(fusedOps.flatMap(o => o.inputs));

    // External inputs: consumed by fused ops but not produced by them
    const externalInputs = [...fusedInputIds].filter(id => !fusedOutputIds.has(id));
    // Final output: produced by fused ops but not consumed by any fused op
    const fusedConsumed = new Set(fusedOps.flatMap(o => o.inputs));
    const finalOutputId = [...fusedOutputIds].find(id => !fusedConsumed.has(id));
    // Intermediate tensors: produced AND consumed within fused ops
    const intermediateTensorIds = new Set([...fusedOutputIds].filter(id => id !== finalOutputId));

    // Determine Q, K, V inputs for the fused op description
    let qId, kId, vId;
    const qktOp = fusedOps.find(o => o.id === 'qkt');
    const contentQkOp = fusedOps.find(o => o.id === 'content_qk');
    const attnVOp = fusedOps.find(o => o.id === 'attn_v' || o.id === 'latent_attn_v');
    if (qktOp) {
        qId = qktOp.inputs[0];
        kId = qktOp.inputs[1];
    } else if (contentQkOp) {
        qId = contentQkOp.inputs[0];
        kId = contentQkOp.inputs[1];
    }
    if (attnVOp) {
        vId = attnVOp.inputs[1];
    }

    // Remove intermediate tensors
    graph.tensors = graph.tensors.filter(t => !intermediateTensorIds.has(t.id));

    // Remove fused ops
    graph.ops = graph.ops.filter(o => !fusedOpIds.has(o.id));

    // Position FlashAttention op midway between the mask and its output tensor.
    // This keeps it centered in the attention region regardless of variant,
    // and naturally aligns both MLA graphs (both resolve to stage 12).
    const outputTensor = graph.tensors.find(t => t.id === finalOutputId);
    const maskTensor = graph.tensors.find(t => t.id === 'mask');
    const flashStage = (outputTensor && maskTensor)
        ? Math.round((maskTensor.stage + outputTensor.stage) / 2)
        : undefined;

    // Insert fused FlashAttention op
    const fusedOp = {
        id: 'flash_attn',
        type: 'flash_attn',
        inputs: externalInputs,
        output: finalOutputId,
        label: 'FlashAttn',
        stage: flashStage,
        desc: `Fused FlashAttention-2 kernel. Q, K, V are tiled into blocks of B_r=${params.block_q} and B_c=${params.block_kv} rows. ` +
              `Each CTA loads one Q tile and iterates over all K/V tiles. Intermediate attention scores and weights stay in SRAM — ` +
              `only the final output O is written back to HBM.`,
    };
    if (contentQkOp && contentQkOp.routeBelow) {
        fusedOp.routeBelow = contentQkOp.routeBelow;
    }
    graph.ops.push(fusedOp);

    // Update ATTENTION group
    for (const group of (graph.groups || [])) {
        if (group.tensors) {
            group.tensors = group.tensors.filter(id => !intermediateTensorIds.has(id));
        }
        if (group.ops) {
            group.ops = group.ops.filter(id => !fusedOpIds.has(id));
            if (group.label.includes('ATTENTION')) {
                group.ops.push('flash_attn');
            }
        }
    }
}

// --- Multi-request annotations (B > 1) ---

function addMultiRequestAnnotations(graph, params) {
    const sLens = [...params.seqLens].slice(0, params.B);
    const sqLens = [...params.queryLens].slice(0, params.B);
    const sumS = sLens.reduce((a, b) => a + b, 0);
    const sumSq = sqLens.reduce((a, b) => a + b, 0);

    for (const t of graph.tensors) {
        // Mark mask as multi-request block-diagonal
        if (t.type === 'mask') {
            t.multiRequest = true;
            t.seqLens = sLens;
            t.queryLens = sqLens;
        }

        // Add request boundary lines to activation tensors with S_q or S dims
        if (t.type === 'weight' || t.type === 'mask') continue;
        const dims = t.dimNames || [];
        const hasSq = dims.some(d => d === 'S_q' || d === '\u03a3S_q');
        const hasS  = dims.some(d => d === 'S' || d === '\u03a3S');
        if (hasSq) {
            const boundaries = [];
            let cum = 0;
            for (let i = 0; i < sqLens.length - 1; i++) {
                cum += sqLens[i];
                boundaries.push(cum);
            }
            t.requestBoundaries = boundaries;
            t.requestBoundaryTotal = sumSq;
        } else if (hasS) {
            const boundaries = [];
            let cum = 0;
            for (let i = 0; i < sLens.length - 1; i++) {
                cum += sLens[i];
                boundaries.push(cum);
            }
            t.requestBoundaries = boundaries;
            t.requestBoundaryTotal = sumS;
        }
    }
}

// --- Paged attention annotations ---

function addPagedAnnotations(graph, params) {
    const bs = params.block_size;
    const sLens = [...params.seqLens].slice(0, params.B);
    const blocksPerSeq = sLens.map(s => Math.ceil(s / bs));
    const totalBlocks = blocksPerSeq.reduce((a, b) => a + b, 0);

    // Build map: cache tensor id → source tensor (the "new" tokens fed into the cache op)
    const tensorMap = {};
    for (const t of graph.tensors) tensorMap[t.id] = t;
    const cacheSourceMap = {};
    for (const op of graph.ops) {
        if (op.type === 'cache' && op.inputs.length > 0) {
            const src = tensorMap[op.inputs[0]];
            if (src) cacheSourceMap[op.output] = src;
        }
    }

    for (const t of graph.tensors) {
        // Mark KV cache tensors with paged layout
        if (t.cache) {
            // Attach source tensor info for the detail view
            const src = cacheSourceMap[t.id];
            if (src) {
                t.cacheSource = { label: src.label, shape: [...src.shape], dimNames: [...(src.dimNames || [])], color: src.color };
            }
            t.badge = 'PAGED';
            // Derive per-token dims from shape (skip the S dimension)
            // 3D tensors like [n_h, S, d_h]: per-token = [n_h, d_h] (indices 0 and 2)
            // 2D tensors like [S, d_c]: per-token = [d_c] (index 1)
            const perTokenDims = t.shape.length === 3
                ? [t.dimNames[0], t.dimNames[2]]
                : t.dimNames.slice(1);
            const perTokenShape = t.shape.length === 3
                ? [t.shape[0], t.shape[2]]
                : t.shape.slice(1);
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
                `New S_q tokens are appended to the last block; a new block is allocated when the current one fills.`;
        }
    }
}

// --- Detail panel close ---

d3.select('#close-detail').on('click', hideDetail);

// Click empty area to deselect and close detail panel
svg.on('click', () => {
    scene.selectAll('.tensor-block').classed('selected', false).attr('filter', null);
    scene.selectAll('.op-node').classed('selected', false);
    scene.selectAll('.group-enclosure').classed('selected', false)
        .each(function() { d3.select(this).selectAll('rect,path').filter(function() { return d3.select(this).attr('stroke-dasharray'); }).attr('stroke-opacity', 0.4).attr('stroke-width', 1.5); });
    hideDetail();
});

// --- Init ---

// Apply default preset for initial variant
const initPresetIdx = VARIANT_DEFAULT_PRESETS[currentVariant] || 0;
const initPreset = PRESETS[initPresetIdx];
if (initPreset && initPreset.variant) {
    for (const key of ['B', 'S', 'S_q', 'n_h', 'd_h', 'n_kv', 'd_c', 'd_q', 'd_r']) {
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

// --- Glossary ---

const GLOSSARY = [
    { section: 'Attention Variants', entries: [
        { term: 'MHA', aka: 'Multi-Head Attention', def: 'The standard attention mechanism used in the original transformer. Each head has its own independent Q, K, and V projections.' },
        { term: 'GQA', aka: 'Grouped-Query Attention', def: 'A variant where multiple query heads share a single key/value head, reducing KV cache memory. Used in Llama, Mistral, Qwen, and others.' },
        { term: 'MQA', aka: 'Multi-Query Attention', def: 'The extreme case of GQA where all query heads share a single key/value head. Used in StarCoder and Falcon.' },
        { term: 'MLA', aka: 'Multi-Head Latent Attention', def: 'A variant (from DeepSeek) that compresses keys and values into a low-rank latent space, reducing cache size. Uses separate pipelines for prefill and decode.' },
    ]},
    { section: 'Dimensions', entries: [
        { term: 'B', def: 'Batch size — the number of independent requests being processed together.' },
        { term: 'S', aka: 'Sequence length', def: 'Total number of tokens in a request\'s context (both cached and new).' },
        { term: 'S_q', aka: 'Query length', def: 'Number of new tokens being processed in this forward pass. During prefill S_q = S; during decode S_q = 1.' },
        { term: 'n_h', aka: 'Number of heads', def: 'How many parallel attention heads the model uses. Each head attends independently, then results are concatenated.' },
        { term: 'n_kv', aka: 'Number of KV heads', def: 'Number of key/value head groups (in GQA). Multiple query heads share each KV head. When n_kv = n_h, it\'s MHA; when n_kv = 1, it\'s MQA.' },
        { term: 'd_h', aka: 'Head dimension', def: 'The dimensionality of each attention head. Q, K, and V vectors within a head have this size.' },
        { term: 'D', aka: 'Model dimension', def: 'The full hidden dimension of the model, equal to n_h \u00d7 d_h. This is the width of the residual stream.' },
        { term: 'd_c', def: 'KV latent dimension in MLA. The compressed representation size for keys and values.' },
        { term: 'd_q', def: 'Q latent dimension in MLA. The compressed representation size for queries.' },
        { term: 'd_r', def: 'RoPE dimension in MLA. The portion of each head dedicated to rotary position embeddings.' },
    ]},
    { section: 'Tensors & Operations', entries: [
        { term: 'Q / K / V', aka: 'Query / Key / Value', def: 'The three projections of the input. Queries ask "what am I looking for?", keys say "what do I contain?", and values carry the actual information to aggregate.' },
        { term: 'QK^T', def: 'The dot product of queries and keys, producing attention scores. Measures how much each query should attend to each key position.' },
        { term: 'Softmax', def: 'Normalizes attention scores into a probability distribution, so they sum to 1 across key positions.' },
        { term: 'Causal mask', def: 'Prevents tokens from attending to future positions by setting those scores to negative infinity before softmax.' },
        { term: 'W_Q / W_K / W_V / W_O', def: 'Learned weight matrices that project the input into Q, K, V spaces, and project the attention output back to model dimension.' },
    ]},
    { section: 'Optimizations', entries: [
        { term: 'FlashAttention', def: 'An algorithm that computes exact attention without materializing the full S \u00d7 S attention matrix. It tiles the computation into blocks that fit in GPU SRAM, dramatically reducing memory usage.' },
        { term: 'PagedAttention', def: 'A memory management technique (from vLLM) that stores the KV cache in fixed-size blocks rather than contiguous memory, reducing fragmentation and enabling efficient batching.' },
        { term: 'RoPE', aka: 'Rotary Position Embeddings', def: 'A method for encoding token position by rotating Q and K vectors. Encodes relative position information directly into the attention computation.' },
        { term: 'Tensor Parallelism', aka: 'TP', def: 'Splits attention heads across multiple GPUs. Each GPU computes a subset of heads, then results are combined with an all-reduce operation.' },
        { term: 'Data Parallelism', aka: 'DP', def: 'Splits the input batch across multiple GPUs along the sequence dimension. Each GPU processes a subset of the data independently using the full model weights.' },
        { term: 'KV cache', def: 'Stores previously computed key and value tensors so they don\'t need to be recomputed for each new token during autoregressive generation.' },
    ]},
    { section: 'Performance', entries: [
        { term: 'FLOPs', aka: 'Floating-point operations', def: 'A measure of computational cost. More FLOPs means more arithmetic the GPU must perform.' },
        { term: 'Memory transfer', def: 'The amount of data moved between GPU global memory (HBM) and compute units. Often the bottleneck for attention.' },
        { term: 'Arithmetic intensity', def: 'The ratio of FLOPs to bytes transferred. Low arithmetic intensity means the operation is bottlenecked by memory bandwidth, not compute.' },
        { term: 'Roofline model', def: 'A performance model that determines whether an operation is limited by compute (FLOP/s) or memory bandwidth, based on its arithmetic intensity.' },
        { term: 'Prefill', def: 'The phase where the model processes all input tokens at once (S_q = S). Typically compute-bound due to large matrix multiplications.' },
        { term: 'Decode', def: 'The phase where the model generates one token at a time (S_q = 1). Typically memory-bound because the KV cache must be read for each token.' },
    ]},
];

function buildGlossary() {
    const body = d3.select('#glossary-body');
    body.selectAll('*').remove();

    for (const section of GLOSSARY) {
        const sec = body.append('div').attr('class', 'glossary-section');
        sec.append('h4').text(section.section);
        for (const e of section.entries) {
            const entry = sec.append('div').attr('class', 'glossary-entry');
            let termText = e.term;
            if (e.aka) termText += ` (${e.aka})`;
            entry.append('span').attr('class', 'glossary-term').text(termText);
            entry.append('span').attr('class', 'glossary-def').text(` — ${e.def}`);
        }
    }
}

d3.select('#glossary-btn').on('click', () => {
    buildGlossary();
    d3.select('#glossary-overlay').classed('visible', true);
    d3.select('#glossary-search').node().value = '';
    d3.select('#glossary-search').node().focus();
});

d3.select('#glossary-close').on('click', () => {
    d3.select('#glossary-overlay').classed('visible', false);
});

d3.select('#glossary-overlay').on('click', function(event) {
    if (event.target === this) d3.select(this).classed('visible', false);
});

d3.select('#glossary-search').on('input', function() {
    const q = this.value.toLowerCase();
    d3.selectAll('.glossary-entry').each(function() {
        const text = this.textContent.toLowerCase();
        d3.select(this).classed('hidden', q && !text.includes(q));
    });
    // Hide section headers if all entries are hidden
    d3.selectAll('.glossary-section').each(function() {
        const visible = d3.select(this).selectAll('.glossary-entry:not(.hidden)').size();
        d3.select(this).classed('hidden', visible === 0);
    });
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') d3.select('#glossary-overlay').classed('visible', false);
});
