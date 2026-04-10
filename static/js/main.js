// main.js — Entry point: D3 setup, zoom/pan, sliders, presets, toggles, update loop

import { renderGraph, computeSharedStagePositions, setDimScaleContext } from './render.js';
import { mhaGraph, gqaGraph, mqaGraph, mlaUpprojGraph, mlaAbsorbedGraph, VARIANT_DESCS } from './graphs.js';
import { showDetail, showTensorDetail, showGroupDetail, hideDetail, refreshDetail } from './details/index.js';
import { computePipelineTotals, fmtNum, fmtBytes, computeRooflineThreshold, GPU_SPECS } from './costs.js';

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
    S:      { label: 'S (seq length)',     min: 1, max: 8192, step: 1,  default: 8, logScale: true },
    S_q:    { label: 'S_q (query len)',   min: 1, max: 8192, step: 1,  default: 8, logScale: true },
    n_h:    { label: 'n_h (heads)',       min: 1, max: 128,  step: 1,  default: 8 },
    d_h:    { label: 'd_h (head dim)',    min: 1, max: 256,  step: 1,  default: 64 },
    n_kv:   { label: 'n_kv (KV heads)',   min: 1, max: 128,  step: 1,  default: 2 },
    d_c:    { label: 'd_c (latent dim)',  min: 1, max: 4096, step: 1,  default: 512 },
    d_r:    { label: 'd_r (RoPE dim)',   min: 1, max: 256,  step: 1,  default: 64 },
    tp_size:{ label: 'TP ranks',          min: 1, max: 8,    step: 1,  default: 1 },
    block_size: { label: 'Block size',    min: 1, max: 128,  step: 1,  default: 16 },
};

const RUNTIME_SLIDERS = ['B', 'S', 'S_q'];

const VARIANT_SLIDERS = {
    mha: ['n_h', 'd_h', 'tp_size'],
    gqa: ['n_h', 'd_h', 'n_kv', 'tp_size'],
    mqa: ['n_h', 'd_h', 'tp_size'],
    mla: ['n_h', 'd_h', 'd_c', 'd_r', 'tp_size'],
};

// Model presets
const PRESETS = [
    { name: 'Custom', variant: null },
    { name: 'GPT-2 (124M)', variant: 'mha', B: 1, S: 12, S_q: 12, n_h: 12, d_h: 64 },
    { name: 'GPT-2 XL (1.5B)', variant: 'mha', B: 1, S: 12, S_q: 12, n_h: 25, d_h: 64 },
    { name: 'Llama 3.1 8B', variant: 'gqa', B: 1, S: 12, S_q: 12, n_h: 32, d_h: 128, n_kv: 8 },
    { name: 'Llama 3.1 70B', variant: 'gqa', B: 1, S: 12, S_q: 12, n_h: 64, d_h: 128, n_kv: 8 },
    { name: 'Llama 3.1 405B', variant: 'gqa', B: 1, S: 12, S_q: 12, n_h: 128, d_h: 128, n_kv: 8 },
    { name: 'Mistral 7B', variant: 'gqa', B: 1, S: 12, S_q: 12, n_h: 32, d_h: 128, n_kv: 8 },
    { name: 'Qwen 2.5 72B', variant: 'gqa', B: 1, S: 12, S_q: 12, n_h: 64, d_h: 128, n_kv: 8 },
    { name: 'StarCoder (15B)', variant: 'mqa', B: 1, S: 12, S_q: 12, n_h: 48, d_h: 128 },
    { name: 'DeepSeek R1', variant: 'mla', B: 1, S: 12, S_q: 12, n_h: 128, d_h: 128, d_c: 512, d_r: 64 },
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
params.flashAttn = false;
params.block_q = 128;
params.block_kv = 128;
params.splitKV = false;
params.seqLens = [8, 9];     // per-request total S (cached + new)
params.queryLens = [4, 1];   // per-request S_q (new query tokens)

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
                    for (const key of ['B', 'S', 'S_q', 'n_h', 'd_h', 'n_kv', 'd_c', 'd_r']) {
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
        for (const key of ['B', 'S', 'S_q', 'n_h', 'd_h', 'n_kv', 'd_c', 'd_r']) {
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

    const isLog2 = key === 'B' || key === 'tp_size';
    const isLogScale = def.logScale;
    const effectiveMax = key === 'n_kv' ? params.n_h : key === 'tp_size' ? Math.min(8, params.n_h) : def.max;

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
            const curMax = key === 'tp_size' ? Math.min(8, params.n_h) : def.max;
            v = Math.max(def.min, Math.min(curMax, v));
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
        if (key === 'S_q' && params.S_q > params.S) {
            params.S = params.S_q;
        }
        if (key === 'S' && params.S_q > params.S) {
            params.S_q = params.S;
        }

        numInput.property('value', params[key]);
        rangeInput.property('value', valToSlider(params[key]));

        // Extend seqLens/queryLens arrays BEFORE update() so addPagedAnnotations sees the right length
        if ((key === 'B' || key === 'S') && params.pagedAttn) buildSeqLengthInputs();

        updateDerived();
        update();

        // Update dependent slider maxes when their constraint changes (don't rebuild — that kills the drag)
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
        // Keep S and S_q sliders visually in sync when one bumps the other
        if (key === 'S' || key === 'S_q') {
            d3.selectAll('#runtime-sliders input[type="range"]').each(function() {
                const group = this.parentNode;
                const label = d3.select(group).select('.dim-name').text();
                if (key === 'S' && label.includes('S_q')) {
                    d3.select(this).property('value', valToSlider(params.S_q));
                    d3.select(group).select('input[type="number"]').property('value', params.S_q);
                } else if (key === 'S_q' && label.includes('S (')) {
                    d3.select(this).property('value', valToSlider(params.S));
                    d3.select(group).select('input[type="number"]').property('value', params.S);
                }
            });
        }
    }

    rangeInput.on('input', function() {
        onSliderChange(sliderToVal(+this.value));
    });
    numInput.on('change', function() {
        const effMax = key === 'n_kv' ? params.n_h : key === 'tp_size' ? Math.min(8, params.n_h) : def.max;
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
        .style('margin-bottom', '4px').text('Per-request sequence lengths:');

    // Ensure arrays match B
    while (params.seqLens.length < params.B) params.seqLens.push(params.S);
    while (params.seqLens.length > params.B) params.seqLens.pop();
    while (params.queryLens.length < params.B) params.queryLens.push(params.S_q || 1);
    while (params.queryLens.length > params.B) params.queryLens.pop();
    // Enforce S_q <= S for all entries
    for (let i = 0; i < params.B; i++) {
        if (params.queryLens[i] > params.seqLens[i]) {
            params.seqLens[i] = params.queryLens[i];
        }
    }

    // Header row
    const header = container.append('div').attr('class', 'seq-row')
        .style('color', '#666').style('font-size', '9px');
    header.append('span').style('width', '42px').text('');
    header.append('span').style('width', '50px').style('text-align', 'center').text('S');
    header.append('span').style('width', '50px').style('text-align', 'center').text('S_q');
    header.append('span').style('font-style', 'italic').text('Type');

    for (let i = 0; i < params.B; i++) {
        const row = container.append('div').attr('class', 'seq-row');
        row.append('span').style('width', '42px').text(`Req ${i}:`);

        // S (total KV length per request)
        const sInp = row.append('input')
            .attr('type', 'number')
            .attr('min', 1)
            .attr('step', 1)
            .property('value', params.seqLens[i]);

        // S_q (new query tokens per request)
        const qInp = row.append('input')
            .attr('type', 'number')
            .attr('min', 1)
            .attr('step', 1)
            .property('value', params.queryLens[i]);

        sInp.on('input', function() {
            let v = parseInt(this.value, 10);
            if (isNaN(v)) return;
            v = Math.max(1, v);
            params.seqLens[i] = v;
            if (params.queryLens[i] > v) {
                params.queryLens[i] = v;
                qInp.property('value', v);
            }
            updateTypeLabel();
            updateDerived();
        });
        sInp.on('change', function() {
            let v = parseInt(this.value, 10);
            if (isNaN(v)) v = 1;
            v = Math.max(1, v);
            params.seqLens[i] = v;
            if (params.queryLens[i] > v) {
                params.queryLens[i] = v;
                qInp.property('value', v);
            }
            this.value = v;
            updateTypeLabel();
            updateDerived();
            update();
        });

        qInp.on('input', function() {
            let v = parseInt(this.value, 10);
            if (isNaN(v)) return;
            v = Math.max(1, v);
            if (v > params.seqLens[i]) {
                params.seqLens[i] = v;
                sInp.property('value', v);
            }
            params.queryLens[i] = v;
            updateTypeLabel();
            updateDerived();
        });
        qInp.on('change', function() {
            let v = parseInt(this.value, 10);
            if (isNaN(v)) v = 1;
            v = Math.max(1, v);
            if (v > params.seqLens[i]) {
                params.seqLens[i] = v;
                sInp.property('value', v);
            }
            params.queryLens[i] = v;
            this.value = v;
            updateTypeLabel();
            updateDerived();
            update();
        });

        // Type label (auto-detect prefill vs decode)
        const typeLabel = row.append('span')
            .style('font-size', '9px');

        function updateTypeLabel() {
            const sq = params.queryLens[i];
            const s = params.seqLens[i];
            const type = sq === 1 ? 'decode' : sq >= s ? 'prefill' : 'extend';
            const color = sq === 1 ? '#3498db' : sq >= s ? '#f39c12' : '#2ecc71';
            typeLabel.style('color', color).text(type);
        }
        updateTypeLabel();
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
        const sLens = params.seqLens.slice(0, params.B);
        const blocksPerSeq = sLens.map(s => Math.ceil(s / params.block_size));
        const totalBlocks = blocksPerSeq.reduce((a, b) => a + b, 0);
        html += `<div class="derived-dim">S per req: <span>[${sLens.join(', ')}]</span></div>`;
        html += `<div class="derived-dim">Blocks per req: <span>[${blocksPerSeq.join(', ')}]</span></div>`;
        html += `<div class="derived-dim">Total KV blocks: <span>${totalBlocks}</span></div>`;
    }

    d3.select('#derived').html(html);
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
    const S = params.S || 1;
    // Evaluate at two S_q points to extract the linear relationship.
    // Use S_q = 1 and S_q = max(2, min(S, 201)). When S=1, bump effective S
    // to 2 so we have two valid sample points.
    const effS = Math.max(S, 2);
    const sqLo = 1;
    const sqHi = Math.min(effS, 201);
    const p1 = { ...params, S: effS, S_q: sqLo };
    const p2 = { ...params, S: effS, S_q: sqHi };
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
    const byteInterceptUpproj = upproj1.totalBytes - byteSlopeUpproj * sqLo;
    const byteInterceptAbsorbed = absorbed1.totalBytes - byteSlopeAbsorbed * sqLo;

    // FLOPs as a function of S_q
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
        S,
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
            .text(`Crossover analysis (S\u2009=\u2009${crossover.S})`);

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

        // Crossover results
        const atS = `at S=${crossover.S}, `;

        function crossoverLine(label, xover, alwaysWinner) {
            const line = container.append('div')
                .style('font-size', '11px')
                .style('color', '#999')
                .style('line-height', '1.4')
                .style('margin-top', label === 'Xfer' ? '6px' : '0');
            if (xover) {
                if (xover.mqaWinsBelow) {
                    line.html(`${label}: ${atS}MQA wins at <span style="color:#7c8cf8;font-weight:600">S_q \u2264 ${xover.sq}</span>`);
                } else {
                    line.html(`${label}: ${atS}MQA wins at <span style="color:#7c8cf8;font-weight:600">S_q \u2265 ${xover.sq}</span>`);
                }
            } else {
                const winner = alwaysWinner === 'mqa' ? 'MQA' : 'MHA';
                const color = alwaysWinner === 'mqa' ? '#2ecc71' : '#e74c3c';
                line.html(`${label}: ${atS}<span style="color:${color}">${winner} always wins</span>`);
            }
        }

        crossoverLine('Xfer', crossover.bytesCrossover, crossover.bytesAlwaysWinner);
        crossoverLine('FLOPs', crossover.flopsCrossover, crossover.flopsAlwaysWinner);
    }
}

// --- Update ---

function annotateGraph(graph) {
    if (params.tp_size > 1) addTpAnnotations(graph, params);
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

    // Insert fused FlashAttention op
    const fusedOp = {
        id: 'flash_attn',
        type: 'flash_attn',
        inputs: externalInputs,
        output: finalOutputId,
        label: 'FlashAttn',
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

// --- Paged attention annotations ---

function addPagedAnnotations(graph, params) {
    const bs = params.block_size;
    const sLens = [...params.seqLens].slice(0, params.B);   // S per request (total KV length)
    const sqLens = [...params.queryLens].slice(0, params.B); // S_q per request (new query tokens)
    const blocksPerSeq = sLens.map(s => Math.ceil(s / bs));
    const totalBlocks = blocksPerSeq.reduce((a, b) => a + b, 0);

    for (const t of graph.tensors) {
        if (t.type === 'mask') {
            t.pagedMask = true;
            t.seqLens = sLens;
            t.queryLens = sqLens;
            const totalS = sLens.reduce((a, b) => a + b, 0);
            const totalSq = sqLens.reduce((a, b) => a + b, 0);
            t.shape = [totalSq, totalS];
            t.dimNames = ['\u03a3S_q', '\u03a3S'];
            const reqDescs = sLens.map((s, i) => `req${i}: S_q=${sqLens[i]}, S=${s}`).join(', ');
            t.desc = `Variable-length causal mask for paged attention. ${reqDescs}. Each request attends only within its own sequence (block-diagonal) and causally.`;
        }
        // Mark KV cache tensors with paged layout
        // Tensors with cache: true are the actual KV cache entries
        if (t.cache) {
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
    for (const key of ['B', 'S', 'S_q', 'n_h', 'd_h', 'n_kv', 'd_c', 'd_r']) {
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
