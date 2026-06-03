// flash-attn.js — FlashAttention-2 tiling detail visualization
// Shows CTA assignment, tile computation, memory hierarchy, and split-KV reduction.

import { drawDetailBlock, RANK_COLORS } from './shared.js';
import { fmtBytes, fmtNum, tensorBytes, computeOpCost, GPU_SPECS, computeRooflineThreshold } from '../costs.js';

const PAD = 20;

// CTA palette (distinct from TP colors)
const CTA_COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12',
    '#9b59b6', '#1abc9c', '#e67e22', '#c0392b',
    '#2980b9', '#27ae60', '#d35400', '#8e44ad',
    '#16a085', '#f1c40f', '#e84393', '#00b894',
];

export function drawFlashAttnDetail(svg, op, tensorMap, params) {
    const w = 590;
    const pad = PAD;

    // Extract tensor info from the fused op's inputs
    const inputTensors = op.inputs.map(id => tensorMap[id]).filter(Boolean);
    const outputTensor = tensorMap[op.output];

    // Identify Q, K, V, mask from the fused op's inputs
    const mask = inputTensors.find(t => t.type === 'mask');
    // Auxiliary RoPE tensors (not primary Q/K/V)
    const auxIds = new Set(['q_r', 'q_rp', 'q_pe', 'k_r']);
    const Q = inputTensors.find(t => !auxIds.has(t.id) && t !== mask &&
        (t.id.startsWith('Q') || t.id === 'q_lat' || t.id === 'q_nope'));
    // K/V: non-Q, non-mask, non-auxiliary tensors
    const kvTensors = inputTensors.filter(t => t !== Q && t !== mask && !auxIds.has(t.id));
    const K = kvTensors[0];
    const V = kvTensors.length > 1 ? kvTensors[1] : K;

    if (!Q || !K || !outputTensor) return;

    // Derive dimensions
    const shQ = Q.shape;
    const n_h = shQ.length >= 3 ? shQ[0] : 1;
    const S_q = shQ.length >= 4 ? shQ[2] : shQ.length >= 3 ? shQ[1] : shQ[0];
    const d_q = shQ[shQ.length - 1];
    const shK = K.shape;
    const S = shK.length >= 4 ? shK[2] : shK.length >= 3 ? shK[1] : shK[0];
    const d_k = shK[shK.length - 1];
    const d_v = V.shape[V.shape.length - 1];

    // Detect GQA: K_cache has fewer heads than Q
    const kCache = tensorMap['K_cache'];
    const n_kv = (kCache && kCache.shape.length >= 3) ? kCache.shape[0] : n_h;
    const isGQA = n_kv < n_h && n_h % n_kv === 0;
    const G = isGQA ? Math.floor(n_h / n_kv) : 1; // heads per KV group

    // State for interactive selection
    let selectedQBlock = 0;
    let selectedKVBlock = 0;

    // --- Build HTML controls above the SVG ---
    const detailBody = d3.select('#detail-body');
    // Remove any prior flash controls
    detailBody.select('.flash-controls').remove();
    const controlsDiv = detailBody.insert('div', '#detail-svg')
        .attr('class', 'flash-controls')
        .style('margin-bottom', '12px');

    buildHTMLControls(controlsDiv, params, isGQA, () => redraw());

    // --- SVG sections ---
    function redraw() {
        svg.selectAll('*').remove();
        const Br = params.block_q;
        const Bc = params.block_kv;
        const nQ = Math.ceil(S_q / Br);
        const nKV = Math.ceil(S / Bc);
        if (selectedQBlock >= nQ) selectedQBlock = nQ - 1;
        if (selectedKVBlock >= nKV) selectedKVBlock = nKV - 1;

        let sY = 10;
        sY = drawPerHeadOverview(svg, pad, sY, w - 2 * pad, n_h, S_q, S, d_q, d_v, Q, K, V, outputTensor, isGQA, G, n_kv, params);

        sY += 16;
        sY = drawCTAGrid(svg, pad, sY, w - 2 * pad, params, S_q, S, nQ, nKV,
            selectedQBlock, selectedKVBlock, isGQA, G, n_kv, (qi, kvi) => {
                selectedQBlock = qi;
                selectedKVBlock = kvi;
                redraw();
            });

        sY += 16;
        sY = drawTensorMapping(svg, pad, sY, w - 2 * pad, params, Q, K, V, outputTensor,
            S_q, S, d_q, d_k, d_v, nQ, nKV, selectedQBlock, selectedKVBlock);

        sY += 16;
        sY = drawTileComputation(svg, pad, sY, w - 2 * pad, params, Q, K, V, outputTensor,
            S_q, S, d_q, d_k, d_v, selectedQBlock, selectedKVBlock, nKV, isGQA, G, n_kv);

        sY += 16;
        sY = drawMemoryHierarchy(svg, pad, sY, w - 2 * pad, params, Q, K, V, outputTensor,
            S_q, S, d_q, d_k, d_v, selectedQBlock, selectedKVBlock, isGQA, G, n_kv);

        if (params.splitKV && nKV > 1) {
            sY += 16;
            sY = drawSplitKVReduction(svg, pad, sY, w - 2 * pad, params, S_q, nKV, selectedQBlock, outputTensor, d_v);
        }

        svg.attr('height', sY + 20);

        // Update stats in controls
        updateControlStats(controlsDiv, params, S_q, S, op, tensorMap, isGQA, G, n_kv);
    }

    redraw();
}

// --- HTML controls (inserted above SVG in detail panel) ---

function buildHTMLControls(container, params, isGQA, onChange) {
    container.selectAll('*').remove();

    const row = container.append('div')
        .style('display', 'flex')
        .style('align-items', 'flex-end')
        .style('gap', '20px')
        .style('flex-wrap', 'wrap');

    // Br slider
    buildHTMLSlider(row, 'B_r (Q tile)', params, 'block_q', 16, 256, 16, onChange);

    // Bc slider
    buildHTMLSlider(row, 'B_c (KV tile)', params, 'block_kv', 16, 256, 16, onChange);

    // Split-KV toggle
    const splitGroup = row.append('div').style('display', 'flex').style('flex-direction', 'column').style('gap', '4px');
    splitGroup.append('span')
        .style('font-size', '12px').style('color', '#bbb').style('font-weight', '500')
        .text('Split-KV');
    const splitToggle = splitGroup.append('div')
        .attr('class', 'toggle-switch')
        .classed('active', params.splitKV)
        .style('cursor', 'pointer');
    splitToggle.on('click', function() {
        params.splitKV = !params.splitKV;
        d3.select(this).classed('active', params.splitKV);
        onChange();
    });

    // PackGQA toggle (only visible for GQA variants)
    if (isGQA) {
        const packGroup = row.append('div').style('display', 'flex').style('flex-direction', 'column').style('gap', '4px');
        packGroup.append('span')
            .style('font-size', '12px').style('color', '#bbb').style('font-weight', '500')
            .text('PackGQA');
        const packToggle = packGroup.append('div')
            .attr('class', 'toggle-switch')
            .classed('active', params.packGQA)
            .style('cursor', 'pointer');
        packToggle.on('click', function() {
            params.packGQA = !params.packGQA;
            d3.select(this).classed('active', params.packGQA);
            onChange();
        });
    }

    // Stats area
    container.append('div')
        .attr('class', 'flash-stats')
        .style('margin-top', '8px')
        .style('font-size', '12px')
        .style('color', '#888');
}

function _fmtTime(seconds) {
    if (seconds < 1e-6) return (seconds * 1e9).toFixed(1) + ' ns';
    if (seconds < 1e-3) return (seconds * 1e6).toFixed(1) + ' \u00b5s';
    if (seconds < 1) return (seconds * 1e3).toFixed(2) + ' ms';
    return seconds.toFixed(3) + ' s';
}

function updateControlStats(container, params, S_q, S, op, tensorMap, isGQA, G, n_kv) {
    const Br = params.block_q, Bc = params.block_kv;
    const nQ = Math.ceil(S_q / Br);
    const nKV = Math.ceil(S / Bc);
    const packed = isGQA && params.packGQA;
    const nCTAs = params.splitKV ? nQ * nKV : nQ;

    const stats = container.select('.flash-stats');
    stats.html('');

    // Tiling summary line
    const tilingLine = stats.append('div').style('margin-bottom', '6px');
    tilingLine.append('span')
        .style('color', '#888')
        .text(`${nQ} Q blocks \u00d7 ${nKV} KV blocks \u2014 `);
    if (packed) {
        tilingLine.append('span')
            .style('color', '#7c8cf8').style('font-weight', '600')
            .text(`${nCTAs} CTAs${params.splitKV ? ' (split-KV)' : ''} per KV group`);
        tilingLine.append('span')
            .style('color', '#e67e22').style('margin-left', '8px').style('font-weight', '600')
            .text(`\u00d7 ${n_kv} groups = ${nCTAs * n_kv} total`);
    } else {
        tilingLine.append('span')
            .style('color', '#7c8cf8').style('font-weight', '600')
            .text(`${nCTAs} CTAs${params.splitKV ? ' (split-KV)' : ''} per head`);
    }
    tilingLine.append('span')
        .style('color', '#666').style('margin-left', '8px')
        .text(params.splitKV ? '(each CTA: 1 Q block \u00d7 1 KV block)' : '(each CTA iterates all KV blocks)');
    if (packed) {
        tilingLine.append('span')
            .style('color', '#e67e22').style('margin-left', '8px')
            .text(`(${G} heads packed per CTA)`);
    }

    // Cost + GPU timing
    const cost = computeOpCost(op, tensorMap);
    if (!cost) return;

    const totalBytes = cost.readBytes + cost.writeBytes;
    const threshold = computeRooflineThreshold('H100 SXM');
    const ai = cost.arithmeticIntensity;
    const regime = cost.flops === 0 ? 'MEMORY-ONLY'
        : ai >= threshold ? 'COMPUTE-BOUND' : 'MEMORY-BOUND';
    const regimeColor = regime === 'COMPUTE-BOUND' ? '#2ecc71'
        : regime === 'MEMORY-BOUND' ? '#e74c3c' : '#888';

    // Cost summary line
    const costLine = stats.append('div')
        .style('display', 'flex').style('gap', '12px').style('flex-wrap', 'wrap')
        .style('margin-bottom', '4px');
    costLine.append('span').html(
        `<span style="color:#888">FLOPs:</span> <span style="color:#7c8cf8;font-weight:600">${fmtNum(cost.flops)}</span>`);
    costLine.append('span').html(
        `<span style="color:#888">Read:</span> <span style="color:#aaa">${fmtBytes(cost.readBytes)}</span>`);
    costLine.append('span').html(
        `<span style="color:#888">Write:</span> <span style="color:#aaa">${fmtBytes(cost.writeBytes)}</span>`);
    costLine.append('span').html(
        `<span style="color:#888">AI:</span> <span style="color:#7c8cf8;font-weight:600">${ai.toFixed(1)}</span> <span style="color:#555">FLOPs/B</span>`);
    costLine.append('span').html(
        `<span style="color:${regimeColor};font-weight:600">${regime}</span>`);

    // GPU table
    const gpuKeys = Object.keys(GPU_SPECS);
    const table = stats.append('table')
        .style('width', '100%').style('border-collapse', 'collapse')
        .style('font-size', '12px').style('margin-top', '4px');
    const thead = table.append('tr');
    for (const h of ['GPU', 'Compute', 'Memory', 'Bound', 'Time']) {
        thead.append('td')
            .style('color', '#555').style('padding', '1px 6px')
            .style('text-align', h === 'GPU' ? 'left' : 'right')
            .text(h);
    }
    for (const key of gpuKeys) {
        const gpu = GPU_SPECS[key];
        const computeTime = cost.flops > 0 ? cost.flops / (gpu.peakTFLOPS_bf16 * 1e12) : 0;
        const memTime = totalBytes / (gpu.bandwidthTBs * 1e12);
        const bottleneck = cost.flops === 0 ? 'MEM' : computeTime >= memTime ? 'COMPUTE' : 'MEM';
        const bnTime = Math.max(computeTime, memTime);
        const bnColor = bottleneck === 'COMPUTE' ? '#2ecc71' : '#e74c3c';

        const row = table.append('tr');
        row.append('td').style('color', '#bbb').style('padding', '1px 6px').style('font-weight', '500').text(key);
        row.append('td').style('color', '#aaa').style('padding', '1px 6px').style('text-align', 'right')
            .text(computeTime > 0 ? _fmtTime(computeTime) : '\u2014');
        row.append('td').style('color', '#aaa').style('padding', '1px 6px').style('text-align', 'right')
            .text(_fmtTime(memTime));
        row.append('td').style('padding', '1px 6px').style('text-align', 'right')
            .style('color', bnColor).style('font-weight', '600').text(bottleneck);
        row.append('td').style('color', '#7c8cf8').style('padding', '1px 6px').style('text-align', 'right')
            .style('font-weight', '600').text(_fmtTime(bnTime));
    }
}

function buildHTMLSlider(container, label, params, key, min, max, step, onChange) {
    const group = container.append('div').attr('class', 'slider-group').style('width', '180px');
    const header = group.append('div').attr('class', 'slider-header');
    header.append('span').attr('class', 'dim-name').text(label);
    const numInput = header.append('input')
        .attr('class', 'dim-input')
        .attr('type', 'number')
        .attr('min', min).attr('max', max).attr('step', step)
        .attr('value', params[key]);
    const rangeInput = group.append('input')
        .attr('type', 'range')
        .attr('min', min).attr('max', max).attr('step', step)
        .attr('value', params[key]);

    rangeInput.on('input', function() {
        params[key] = +this.value;
        numInput.property('value', params[key]);
        onChange();
    });
    numInput.on('change', function() {
        let v = Math.max(min, Math.min(max, +this.value || min));
        v = Math.round(v / step) * step;
        params[key] = v;
        this.value = v;
        rangeInput.property('value', v);
        onChange();
    });
}

// --- Section 0: Per-Head Overview ---
// Uses the same shared-hinge proportional matmul layout as Tensor Mapping,
// but with depth-stacked layers showing n_h heads (front head highlighted).

function drawPerHeadOverview(g, x, y, width, n_h, S_q, S, d_q, d_v, Q, K, V, O, isGQA, G, n_kv, params) {
    const sectionG = g.append('g').attr('transform', `translate(${x}, ${y})`);

    sectionG.append('text')
        .attr('x', 0).attr('y', 14)
        .attr('fill', '#bbb').attr('font-size', '12px').attr('font-weight', '600')
        .text('Per-Head Decomposition');

    const sameKV = K.id === V.id;
    const d_k = K.shape[K.shape.length - 1];

    // Depth stacking params
    const maxLayers = Math.min(n_h, 8);
    const layerOff = 3; // px offset per layer (diagonal)
    const stackW = maxLayers * layerOff;
    const stackH = maxLayers * layerOff;

    // --- Proportional shared-hinge layout (same as Tensor Mapping, scaled down) ---
    //              K^T [d_k × S]      V [S × d_v]
    //  Q [S_q×d_q]  S/P [S_q × S]    O [S_q × d_v]
    const headerH = 28;
    const topLabelH = 20;
    const gapInner = 6;
    const softmaxGap = 20;
    const leftMargin = 50;
    const minPx = 32;
    const maxH = 340;
    const pow = 0.4;

    const rawSq = Math.pow(S_q, pow);
    const rawS  = Math.pow(S, pow);
    const rawDq = Math.pow(d_q, pow);
    const rawDk = Math.pow(d_k, pow);
    const rawDv = Math.pow(d_v, pow);

    // Scale to fit, leaving room for stack depth
    const kW = (width - leftMargin - gapInner - softmaxGap - stackW) / (rawDq + rawS + rawDv);
    const kH = (maxH - topLabelH - gapInner - stackH) / (Math.max(rawDk, rawS) + rawSq);
    const k = Math.min(kW, kH);

    const pxDq = Math.max(minPx, rawDq * k);
    const pxDk = Math.max(minPx, rawDk * k);
    const pxS  = Math.max(minPx, rawS * k);
    const pxSq = Math.max(minPx, rawSq * k);
    const pxDv = Math.max(minPx, rawDv * k);

    // Center the blocks themselves; labels extend left
    const blocksW = pxDq + gapInner + pxS + softmaxGap + pxDv + stackW;
    const col0X = Math.max(leftMargin, (width - blocksW) / 2);
    const col1X = col0X + pxDq + gapInner;
    const col2X = col1X + pxS + softmaxGap;

    const topMaxH = Math.max(pxDk, pxS);
    const resultY = headerH + topLabelH + topMaxH + gapInner + stackH;

    // Tensor positions (front face — layer 0)
    const blocks = [
        { id: 'Q',  bx: col0X, by: resultY,                       bw: pxDq, bh: pxSq, color: '#e74c3c', label: Q.label },
        { id: 'KT', bx: col1X, by: resultY - gapInner - pxDk,     bw: pxS,  bh: pxDk, color: '#2ecc71', label: K.label + '\u1d40' },
        { id: 'SP', bx: col1X, by: resultY,                       bw: pxS,  bh: pxSq, color: '#9b59b6', label: 'S/P', dashed: true },
        ...(!sameKV ? [{ id: 'V', bx: col2X, by: resultY - gapInner - pxS, bw: pxDv, bh: pxS, color: '#f39c12', label: V.label }] : []),
        { id: 'O',  bx: col2X, by: resultY,                       bw: pxDv, bh: pxSq, color: '#3498db', label: 'O', sublabel: O.label !== 'O' ? O.label : null },
    ];

    // Draw each tensor as a depth stack
    for (const blk of blocks) {
        // Back layers (dimmed)
        for (let li = maxLayers - 1; li >= 1; li--) {
            const lx = blk.bx + li * layerOff;
            const ly = blk.by - li * layerOff;
            const rect = sectionG.append('rect')
                .attr('x', lx).attr('y', ly)
                .attr('width', blk.bw).attr('height', blk.bh)
                .attr('fill', blk.color).attr('fill-opacity', 0.06 + (maxLayers - li) * 0.015)
                .attr('stroke', blk.color).attr('stroke-opacity', 0.15)
                .attr('stroke-width', 0.5).attr('rx', 2);
            if (blk.dashed) rect.attr('stroke-dasharray', '4,2');
        }

        // Front layer (highlighted)
        const rect = sectionG.append('rect')
            .attr('x', blk.bx).attr('y', blk.by)
            .attr('width', blk.bw).attr('height', blk.bh)
            .attr('fill', blk.color).attr('fill-opacity', blk.dashed ? 0.12 : 0.25)
            .attr('stroke', blk.color).attr('stroke-opacity', 0.7)
            .attr('stroke-width', 1.5).attr('rx', 2);
        if (blk.dashed) rect.attr('stroke-dasharray', '5,3');

        // Label on front face
        if (blk.bw >= 20 && blk.bh >= 14) {
            const hasSubLabel = blk.sublabel && blk.bh >= 24;
            const mainY = blk.by + blk.bh / 2 + (hasSubLabel ? -1 : 3);
            sectionG.append('text')
                .attr('x', blk.bx + blk.bw / 2).attr('y', mainY)
                .attr('text-anchor', 'middle')
                .attr('fill', '#fff').attr('font-size', Math.min(10, blk.bw / 4) + 'px')
                .attr('font-weight', '600')
                .text(blk.label);
            if (hasSubLabel) {
                sectionG.append('text')
                    .attr('x', blk.bx + blk.bw / 2).attr('y', mainY + 10)
                    .attr('text-anchor', 'middle')
                    .attr('fill', '#aaa').attr('font-size', '11px')
                    .text(`(${blk.sublabel})`);
            }
        }
    }

    // --- Tensor name labels above ---
    const ktBlk = blocks.find(b => b.id === 'KT');
    sectionG.append('text')
        .attr('x', ktBlk.bx + ktBlk.bw / 2).attr('y', ktBlk.by - stackH - 4)
        .attr('text-anchor', 'middle')
        .attr('fill', '#2ecc71').attr('font-size', '11px').attr('font-weight', '600')
        .text(K.label + '\u1d40');

    const vBlk = blocks.find(b => b.id === 'V');
    if (vBlk) {
        sectionG.append('text')
            .attr('x', vBlk.bx + vBlk.bw / 2).attr('y', vBlk.by - stackH - 4)
            .attr('text-anchor', 'middle')
            .attr('fill', '#f39c12').attr('font-size', '11px').attr('font-weight', '600')
            .text(V.label);
    }

    const qBlk = blocks[0];
    sectionG.append('text')
        .attr('x', qBlk.bx - 4).attr('y', qBlk.by + qBlk.bh / 2 + 3)
        .attr('text-anchor', 'end')
        .attr('fill', '#e74c3c').attr('font-size', '11px').attr('font-weight', '600')
        .text(Q.label);

    // --- Dimension labels ---
    sectionG.append('text')
        .attr('x', qBlk.bx - 4).attr('y', qBlk.by + 8)
        .attr('text-anchor', 'end')
        .attr('fill', '#555').attr('font-size', '11px')
        .text(`S_q=${S_q}`);
    sectionG.append('text')
        .attr('x', qBlk.bx + qBlk.bw / 2).attr('y', qBlk.by + qBlk.bh + 10)
        .attr('text-anchor', 'middle')
        .attr('fill', '#555').attr('font-size', '11px')
        .text(`d=${d_q}`);
    const spBlk = blocks.find(b => b.id === 'SP');
    sectionG.append('text')
        .attr('x', spBlk.bx + spBlk.bw / 2).attr('y', spBlk.by + spBlk.bh + 10)
        .attr('text-anchor', 'middle')
        .attr('fill', '#555').attr('font-size', '11px')
        .text(`S=${S}`);

    // --- Softmax arrow in the gap ---
    const softX = spBlk.bx + spBlk.bw + softmaxGap / 2;
    const softY = spBlk.by + spBlk.bh / 2;
    sectionG.append('text')
        .attr('x', softX).attr('y', softY)
        .attr('text-anchor', 'middle')
        .attr('fill', '#f39c12').attr('font-size', '11px').attr('font-weight', '600')
        .attr('transform', `rotate(-90, ${softX}, ${softY})`)
        .text('softmax \u2192');

    // --- n_h bracket along the depth diagonal ---
    const annoY = resultY + pxSq + 16;
    if (n_h > 1) {
        const rightBlk = blocks[blocks.length - 1];
        // Back layer top-right corner
        const backX = rightBlk.bx + rightBlk.bw + stackW;
        const backY = rightBlk.by - stackH;
        // Front layer top-right corner
        const frontX = rightBlk.bx + rightBlk.bw;
        const frontY = rightBlk.by;
        // Bracket offset perpendicular to the diagonal (outward)
        const dx = stackW, dy = -stackH;
        const len = Math.sqrt(dx * dx + dy * dy);
        const px = -dy / len * 6; // perpendicular x (outward from tensor)
        const py = dx / len * 6;  // perpendicular y
        // Bracket line along the diagonal
        sectionG.append('line')
            .attr('x1', backX + px).attr('y1', backY + py)
            .attr('x2', frontX + px).attr('y2', frontY + py)
            .attr('stroke', '#888').attr('stroke-width', 1);
        // End ticks (perpendicular to diagonal)
        sectionG.append('line')
            .attr('x1', backX + px).attr('y1', backY + py)
            .attr('x2', backX + px * 0.5).attr('y2', backY + py * 0.5)
            .attr('stroke', '#888').attr('stroke-width', 1);
        sectionG.append('line')
            .attr('x1', frontX + px).attr('y1', frontY + py)
            .attr('x2', frontX + px * 0.5).attr('y2', frontY + py * 0.5)
            .attr('stroke', '#888').attr('stroke-width', 1);
        // Label
        const midX = (backX + frontX) / 2 + px + 4;
        const midY = (backY + frontY) / 2 + py + 3;
        sectionG.append('text')
            .attr('x', midX).attr('y', midY)
            .attr('fill', '#aaa').attr('font-size', '12px')
            .text(`n_h=${n_h}`);
    }

    // Bottom annotation
    const packed = isGQA && params.packGQA;
    let headText;
    if (packed) {
        headText = `PackGQA: ${G} query heads share 1 KV head \u2014 ${n_kv} groups, each CTA processes ${G} heads`;
    } else if (n_h > 1) {
        headText = `Showing 1 of ${n_h} heads \u2014 repeated independently for each head`;
    } else {
        headText = 'Single head (n_h = 1)';
    }
    sectionG.append('text')
        .attr('x', width / 2).attr('y', annoY)
        .attr('text-anchor', 'middle')
        .attr('fill', packed ? '#e67e22' : '#f39c12').attr('font-size', '12px').attr('font-style', 'italic')
        .text(headText);

    return y + annoY + 8;
}

// --- Section A: CTA Grid ---

function drawCTAGrid(g, x, y, width, params, S_q, S, numQBlocks, numKVBlocks,
                     selectedQ, selectedKV, isGQA, G, n_kv, onSelect) {
    const Br = params.block_q, Bc = params.block_kv;
    const splitKV = params.splitKV;
    const packed = isGQA && params.packGQA;
    const B = params.B || 1;
    const dpSize = (params.dp_size || 1);
    const effectiveDp = Math.min(dpSize, B);

    const sectionG = g.append('g').attr('transform', `translate(${x}, ${y})`);

    // Section title
    const titleText = packed
        ? `CTA Assignment (per KV group \u2014 ${G} heads packed)`
        : 'CTA Assignment';
    sectionG.append('text')
        .attr('x', 0).attr('y', 14)
        .attr('fill', packed ? '#e67e22' : '#bbb').attr('font-size', '12px').attr('font-weight', '600')
        .text(titleText);
    sectionG.append('text')
        .attr('x', 0).attr('y', 28)
        .attr('fill', '#666').attr('font-size', '12px').attr('font-style', 'italic')
        .text('Click a cell to see its tile details');

    const titleH = 38;
    const labelW = 60;
    const labelH = 20;
    const dpLabelMargin = (effectiveDp > 1) ? 36 : 0;

    // Compute cell size to fit — square cells so the grid is square
    const maxGridW = width - 20;
    const maxGridH = 400;
    const cellSize = Math.min(40, maxGridW / numKVBlocks, maxGridH / numQBlocks);
    const cellW = cellSize;
    const cellH = cellSize;
    const gridW = cellW * numKVBlocks;
    const gridH = cellH * numQBlocks;

    // Center the grid itself in the full width; labels are placed relative to it
    const gridX = (width - gridW) / 2;
    const gridY = titleH + labelH;

    // Column headers (KV blocks)
    for (let j = 0; j < numKVBlocks; j++) {
        // Only show labels if cells are wide enough
        if (cellW >= 12 || j % Math.ceil(12 / cellW) === 0) {
            sectionG.append('text')
                .attr('x', gridX + j * cellW + cellW / 2)
                .attr('y', titleH + labelH - 4)
                .attr('text-anchor', 'middle')
                .attr('fill', '#666').attr('font-size', '11px')
                .text(cellW >= 24 ? `KV ${j}` : j);
        }
    }

    // Row labels (Q blocks)
    for (let i = 0; i < numQBlocks; i++) {
        if (cellH >= 12 || i % Math.ceil(12 / cellH) === 0) {
            sectionG.append('text')
                .attr('x', gridX - 6)
                .attr('y', gridY + i * cellH + cellH / 2 + 3)
                .attr('text-anchor', 'end')
                .attr('fill', '#666').attr('font-size', '11px')
                .text(cellH >= 20 ? `Q ${i}` : i);
        }
    }

    // Build cumulative offsets for multi-request block-diagonal masking
    const queryLens = params.queryLens ? params.queryLens.slice(0, B) : [S_q];
    const seqLens = params.seqLens ? params.seqLens.slice(0, B) : [S];
    const cumSq = [0];
    const cumS = [0];
    for (let r = 0; r < queryLens.length; r++) {
        cumSq.push(cumSq[r] + queryLens[r]);
        cumS.push(cumS[r] + seqLens[r]);
    }

    // Check if Q block [qStart, qEnd] has ANY valid attention to KV block [kvStart, kvEnd]
    // under the block-diagonal causal mask (plus the sliding window, when active).
    const swaActive = !!params.slidingWindow;
    const swaW = params.window_size;
    function isCausallyReachable(qStart, qEnd, kvStart, kvEnd) {
        for (let r = 0; r < queryLens.length; r++) {
            const rqS = cumSq[r], rqE = cumSq[r + 1] - 1;
            const rkS = cumS[r], rkE = cumS[r + 1] - 1;
            // Q block must overlap this request's query range
            if (qEnd < rqS || qStart > rqE) continue;
            // KV block must overlap this request's key range
            if (kvEnd < rkS || kvStart > rkE) continue;
            const kvOffset = seqLens[r] - queryLens[r];
            // Local query indices (within this request) covered by the Q block
            const loQ = Math.max(qStart, rqS) - rqS;
            const hiQ = Math.min(qEnd, rqE) - rqS;
            // Causal upper bound: highest key the block's last query can attend
            const maxReachableK = rkS + kvOffset + hiQ;
            // KV indices (global) covered by the KV block within this request
            const kvLo = Math.max(kvStart, rkS);
            const kvHi = Math.min(kvEnd, rkE);
            if (kvLo > maxReachableK) continue; // entirely in the future
            // Window lower bound: lowest key the block's first query can attend
            if (swaActive && swaW < seqLens[r]) {
                const minReachableK = rkS + kvOffset + loQ - swaW + 1;
                if (kvHi < minReachableK) continue; // entirely before the window
            }
            return true;
        }
        return false;
    }

    // Draw grid cells
    for (let i = 0; i < numQBlocks; i++) {
        const qStart = i * Br;

        for (let j = 0; j < numKVBlocks; j++) {
            const kvStart = j * Bc;

            const causallyReachable = isCausallyReachable(
                qStart, Math.min(qStart + Br - 1, S_q - 1),
                kvStart, Math.min(kvStart + Bc - 1, S - 1)
            );

            const isSelected = (i === selectedQ && j === selectedKV);
            const ctaId = splitKV ? (i * numKVBlocks + j) : i;
            // Use a coprime stride to avoid column aliasing when numKVBlocks
            // is a multiple of the palette size (all columns same color).
            const colorIdx = splitKV
                ? (i * 7 + j * 3) % CTA_COLORS.length
                : ctaId % CTA_COLORS.length;
            const color = CTA_COLORS[colorIdx];

            const cell = sectionG.append('rect')
                .attr('x', gridX + j * cellW + 0.5)
                .attr('y', gridY + i * cellH + 0.5)
                .attr('width', cellW - 1)
                .attr('height', cellH - 1)
                .attr('rx', 2)
                .style('cursor', causallyReachable ? 'pointer' : 'default');

            if (!causallyReachable) {
                // Grayed out — causal mask skips this block
                cell.attr('fill', '#1a1d2a').attr('fill-opacity', 0.5)
                    .attr('stroke', '#2a2d3a').attr('stroke-width', 0.5)
                    .attr('pointer-events', 'none');
            } else if (splitKV) {
                // Each cell is a separate CTA
                cell.attr('fill', color).attr('fill-opacity', isSelected ? 0.9 : 0.4)
                    .attr('stroke', isSelected ? '#fff' : d3.color(color).darker(0.5))
                    .attr('stroke-width', isSelected ? 2 : 0.5);
            } else {
                // Row = CTA, columns = iterations
                cell.attr('fill', color).attr('fill-opacity', isSelected ? 0.9 : 0.25)
                    .attr('stroke', isSelected ? '#fff' : d3.color(color).darker(0.5))
                    .attr('stroke-width', isSelected ? 2 : 0.5);
            }

            cell.on('click', () => onSelect(i, j));

            // CTA label inside cell if big enough
            if (cellW >= 28 && cellH >= 16 && causallyReachable) {
                sectionG.append('text')
                    .attr('x', gridX + j * cellW + cellW / 2)
                    .attr('y', gridY + i * cellH + cellH / 2 + 3)
                    .attr('text-anchor', 'middle')
                    .attr('fill', causallyReachable ? '#fff' : '#444')
                    .attr('font-size', '12px')
                    .attr('pointer-events', 'none')
                    .text(splitKV ? `CTA${ctaId}` : (j === 0 ? `CTA${ctaId}` : `iter${j}`));
            }
        }

        // Without split-KV: show iteration arrow along the row
        if (!splitKV && cellW * numKVBlocks > 40) {
            const arrowY = gridY + i * cellH + cellH / 2;
            sectionG.append('line')
                .attr('x1', gridX + 2).attr('y1', arrowY)
                .attr('x2', gridX + gridW - 6).attr('y2', arrowY)
                .attr('stroke', '#555').attr('stroke-width', 0.5)
                .attr('stroke-dasharray', '2,2')
                .attr('pointer-events', 'none');
        }
    }

    // Request boundary lines on the grid (when B > 1)
    if (B > 1) {
        for (let r = 1; r < B; r++) {
            // Horizontal line at cumulative S_q boundary
            const qBoundary = cumSq[r] / S_q * gridH;
            sectionG.append('line')
                .attr('x1', gridX).attr('y1', gridY + qBoundary)
                .attr('x2', gridX + gridW).attr('y2', gridY + qBoundary)
                .attr('stroke', '#e74c3c').attr('stroke-width', 1).attr('stroke-opacity', 0.5)
                .attr('pointer-events', 'none');
            // Vertical line at cumulative S boundary
            const kvBoundary = cumS[r] / S * gridW;
            sectionG.append('line')
                .attr('x1', gridX + kvBoundary).attr('y1', gridY)
                .attr('x2', gridX + kvBoundary).attr('y2', gridY + gridH)
                .attr('stroke', '#e74c3c').attr('stroke-width', 1).attr('stroke-opacity', 0.5)
                .attr('pointer-events', 'none');
        }
    }

    // DP rank outlines on the grid
    if (effectiveDp > 1) {
        for (let dp = 0; dp < effectiveDp; dp++) {
            const rankColor = RANK_COLORS[dp * (params.tp_size || 1) % RANK_COLORS.length];
            const qFrac0 = cumSq[_dpRankStartReq(dp, effectiveDp, B)] / S_q;
            const qFrac1 = cumSq[_dpRankEndReq(dp, effectiveDp, B)] / S_q;
            const kvFrac0 = cumS[_dpRankStartReq(dp, effectiveDp, B)] / S;
            const kvFrac1 = cumS[_dpRankEndReq(dp, effectiveDp, B)] / S;

            const rx = gridX + kvFrac0 * gridW;
            const ry = gridY + qFrac0 * gridH;
            const rw = (kvFrac1 - kvFrac0) * gridW;
            const rh = (qFrac1 - qFrac0) * gridH;

            sectionG.append('rect')
                .attr('x', rx - 1).attr('y', ry - 1)
                .attr('width', rw + 2).attr('height', rh + 2)
                .attr('fill', 'none')
                .attr('stroke', rankColor).attr('stroke-width', 1.5)
                .attr('stroke-dasharray', '4,2').attr('stroke-opacity', 0.8)
                .attr('rx', 2)
                .attr('pointer-events', 'none');

            const labelX = rx + rw + 4;
            const labelY = ry + rh / 2 + 3;
            if (rw >= 8 && rh >= 8) {
                sectionG.append('text')
                    .attr('x', labelX).attr('y', labelY)
                    .attr('fill', rankColor).attr('font-size', '11px').attr('font-weight', '600')
                    .attr('pointer-events', 'none')
                    .text(`DP${dp}`);
            }
        }
    }

    // Legend + selection info
    const legendY = gridY + gridH + 10;
    const gridCenterX = gridX + gridW / 2;
    const legendG = sectionG.append('g').attr('transform', `translate(${gridX}, ${legendY})`);
    legendG.append('rect').attr('width', 8).attr('height', 8).attr('fill', '#1a1d2a').attr('stroke', '#2a2d3a').attr('rx', 1);
    legendG.append('text').attr('x', 12).attr('y', 7).attr('fill', '#666').attr('font-size', '11px')
        .text(swaActive ? 'Skipped (causal + window mask)' : 'Skipped (causal mask)');

    const qRange = `[${selectedQ * Br}..${Math.min((selectedQ + 1) * Br, S_q) - 1}]`;
    const kvRange = `[${selectedKV * Bc}..${Math.min((selectedKV + 1) * Bc, S) - 1}]`;
    let selText = `Selected: Q rows ${qRange}, KV rows ${kvRange}`;
    if (effectiveDp > 1) {
        const qMid = (selectedQ + 0.5) * Br;
        let selDp = null;
        for (let dp = 0; dp < effectiveDp; dp++) {
            const s = cumSq[_dpRankStartReq(dp, effectiveDp, B)];
            const e = cumSq[_dpRankEndReq(dp, effectiveDp, B)];
            if (qMid >= s && qMid < e) { selDp = dp; break; }
        }
        if (selDp != null) selText += ` — DP rank ${selDp}`;
    }
    sectionG.append('text')
        .attr('x', gridCenterX).attr('y', legendY + 20)
        .attr('text-anchor', 'middle')
        .attr('fill', '#aaa').attr('font-size', '12px')
        .text(selText);

    let bottomY = gridY + gridH + 38;

    // DP annotation note
    if (effectiveDp > 1) {
        sectionG.append('text')
            .attr('x', width / 2).attr('y', bottomY)
            .attr('text-anchor', 'middle')
            .attr('fill', '#7c8cf8').attr('font-size', '11px').attr('font-style', 'italic')
            .text(`${effectiveDp} DP ranks — each rank runs CTAs only within its dashed region (no cross-rank attention)`);
        bottomY += 16;
        if (dpSize > B) {
            sectionG.append('text')
                .attr('x', width / 2).attr('y', bottomY)
                .attr('text-anchor', 'middle')
                .attr('fill', '#e67e22').attr('font-size', '11px')
                .text(`${dpSize - B} idle DP rank${dpSize - B !== 1 ? 's' : ''} (DP ≥ ${B} = batch size)`);
            bottomY += 16;
        }
    }

    // PackGQA annotation
    if (packed) {
        sectionG.append('text')
            .attr('x', width / 2).attr('y', bottomY)
            .attr('text-anchor', 'middle')
            .attr('fill', '#e67e22').attr('font-size', '11px').attr('font-style', 'italic')
            .text(`Each CTA processes ${G} query heads (h₀..h₍${G-1}₎) sharing the same KV head \u2014 K/V tiles loaded once per group`);
        bottomY += 16;
    }

    return y + bottomY;
}

// --- Section B: Tensor Mapping ---

function drawTensorMapping(g, x, y, width, params, Q, K, V, O,
                           S_q, S, d_q, d_k, d_v, nQ, nKV, selQ, selKV) {
    const Br = params.block_q, Bc = params.block_kv;
    const actBr = Math.min(Br, S_q - selQ * Br);
    const actBc = Math.min(Bc, S - selKV * Bc);
    const sameKV = K.id === V.id;

    const sectionG = g.append('g').attr('transform', `translate(${x}, ${y})`);
    sectionG.append('text')
        .attr('x', 0).attr('y', 14)
        .attr('fill', '#bbb').attr('font-size', '12px').attr('font-weight', '600')
        .text('Tensor Mapping');

    // Shared-hinge matmul layout — full tensors with tile highlights:
    //
    //              K^T [d_k × S]      V [S × d_v]
    //  Q [S_q×d_q]  S/P [S_q × S]    O [S_q × d_v]
    //
    // Matmul 1: Q @ K^T = S  →  softmax  →  Matmul 2: P @ V = O
    // S/P is the attention matrix — dashed, never materialized to HBM.

    const headerH = 28;
    const topLabelH = 20;
    const gapInner = 6;
    const softmaxGap = 24;
    const leftMargin = 60;
    const minPx = 32;
    const maxH = 420;
    const pow = 0.4;

    // Compressed dimension mapping (power scale to handle S >> d)
    const rawSq = Math.pow(S_q, pow);
    const rawS  = Math.pow(S, pow);
    const rawDq = Math.pow(d_q, pow);
    const rawDk = Math.pow(d_k, pow);
    const rawDv = Math.pow(d_v, pow);

    // Scale to fit width and height
    const kW = (width - leftMargin - gapInner - softmaxGap) / (rawDq + rawS + rawDv);
    const kH = (maxH - topLabelH - gapInner) / (Math.max(rawDk, rawS) + rawSq);
    const k = Math.min(kW, kH);

    const pxDq = Math.max(minPx, rawDq * k);
    const pxDk = Math.max(minPx, rawDk * k);
    const pxS  = Math.max(minPx, rawS * k);
    const pxSq = Math.max(minPx, rawSq * k);
    const pxDv = Math.max(minPx, rawDv * k);

    // Center the blocks themselves; labels extend left
    const blocksW = pxDq + gapInner + pxS + softmaxGap + pxDv;
    const col0X = Math.max(leftMargin, (width - blocksW) / 2);
    const col1X = col0X + pxDq + gapInner;
    const col2X = col1X + pxS + softmaxGap;

    const topMaxH = Math.max(pxDk, pxS);
    const resultY = headerH + topLabelH + topMaxH + gapInner;

    // Block positions (each block = full tensor, tile highlighted within)
    const qBx = col0X, qBy = resultY, qBw = pxDq, qBh = pxSq;
    const ktBx = col1X, ktBy = resultY - gapInner - pxDk, ktBw = pxS, ktBh = pxDk;
    const sBx = col1X, sBy = resultY, sBw = pxS, sBh = pxSq;
    const vBx = col2X, vBy = resultY - gapInner - pxS, vBw = pxDv, vBh = pxS;
    const oBx = col2X, oBy = resultY, oBw = pxDv, oBh = pxSq;

    const qTileRow = selQ * Br;
    const kvTileCol = selKV * Bc;

    // --- Helper: draw full tensor outline + tile highlight + bracket annotations ---
    function drawTensorBlock(bx, by, bw, bh, fullRows, fullCols,
                             tileRow, tileCol, tileRows, tileCols,
                             color, tileLabel, dashed) {
        // Full tensor background
        const rect = sectionG.append('rect')
            .attr('x', bx).attr('y', by)
            .attr('width', bw).attr('height', bh)
            .attr('fill', color).attr('fill-opacity', dashed ? 0.04 : 0.08)
            .attr('stroke', color).attr('stroke-opacity', dashed ? 0.35 : 0.25)
            .attr('stroke-width', 1).attr('rx', 2);
        if (dashed) rect.attr('stroke-dasharray', '5,3');

        // Tile highlight
        const tx = bx + (tileCol / fullCols) * bw;
        const ty = by + (tileRow / fullRows) * bh;
        const tw = Math.max((tileCols / fullCols) * bw, 2);
        const th = Math.max((tileRows / fullRows) * bh, 2);

        sectionG.append('rect')
            .attr('x', tx).attr('y', ty)
            .attr('width', tw).attr('height', th)
            .attr('fill', color).attr('fill-opacity', dashed ? 0.35 : 0.5)
            .attr('stroke', '#fff').attr('stroke-width', 1).attr('rx', 1);

        // Tile label inside if big enough
        if (tw >= 24 && th >= 14) {
            sectionG.append('text')
                .attr('x', tx + tw / 2).attr('y', ty + th / 2 + 3)
                .attr('text-anchor', 'middle')
                .attr('fill', '#fff').attr('font-size', '11px').attr('font-weight', '600')
                .text(tileLabel);
        }

        // Row bracket (right side) — when tile doesn't span full height
        if (tileRows < fullRows && th >= 2) {
            const bkX = bx + bw + 3;
            sectionG.append('line')
                .attr('x1', bkX).attr('y1', ty)
                .attr('x2', bkX).attr('y2', ty + th)
                .attr('stroke', '#888').attr('stroke-width', 1);
            sectionG.append('line')
                .attr('x1', bkX).attr('y1', ty)
                .attr('x2', bkX - 3).attr('y2', ty)
                .attr('stroke', '#888').attr('stroke-width', 1);
            sectionG.append('line')
                .attr('x1', bkX).attr('y1', ty + th)
                .attr('x2', bkX - 3).attr('y2', ty + th)
                .attr('stroke', '#888').attr('stroke-width', 1);
            sectionG.append('text')
                .attr('x', bkX + 3).attr('y', ty + th / 2 + 3)
                .attr('fill', '#aaa').attr('font-size', '11px')
                .text(`${tileRow}:${tileRow + tileRows - 1}`);
        }

        // Column bracket (bottom) — when tile doesn't span full width
        if (tileCols < fullCols && tw >= 2) {
            const bkY = by + bh + 3;
            sectionG.append('line')
                .attr('x1', tx).attr('y1', bkY)
                .attr('x2', tx + tw).attr('y2', bkY)
                .attr('stroke', '#888').attr('stroke-width', 1);
            sectionG.append('line')
                .attr('x1', tx).attr('y1', bkY)
                .attr('x2', tx).attr('y2', bkY - 3)
                .attr('stroke', '#888').attr('stroke-width', 1);
            sectionG.append('line')
                .attr('x1', tx + tw).attr('y1', bkY)
                .attr('x2', tx + tw).attr('y2', bkY - 3)
                .attr('stroke', '#888').attr('stroke-width', 1);
            sectionG.append('text')
                .attr('x', tx + tw / 2).attr('y', bkY + 10)
                .attr('text-anchor', 'middle')
                .attr('fill', '#aaa').attr('font-size', '11px')
                .text(`${tileCol}:${tileCol + tileCols - 1}`);
        }
    }

    // --- Draw tensor blocks ---
    // Q: [S_q × d_q], tile = full-width stripe
    drawTensorBlock(qBx, qBy, qBw, qBh, S_q, d_q,
        qTileRow, 0, actBr, d_q, '#e74c3c', 'Q_i', false);

    // K^T: [d_k × S], tile = full-height stripe
    drawTensorBlock(ktBx, ktBy, ktBw, ktBh, d_k, S,
        0, kvTileCol, d_k, actBc, '#2ecc71', 'K_j\u1d40', false);

    // S/P (attention matrix): [S_q × S], tile = small patch at intersection
    drawTensorBlock(sBx, sBy, sBw, sBh, S_q, S,
        qTileRow, kvTileCol, actBr, actBc, '#9b59b6', 'S/P', true);

    // V: [S × d_v], tile = full-width stripe
    const vColor = sameKV ? '#2ecc71' : '#f39c12';
    drawTensorBlock(vBx, vBy, vBw, vBh, S, d_v,
        kvTileCol, 0, actBc, d_v, vColor, 'V_j', false);

    // O: [S_q × d_v], tile = full-width stripe
    drawTensorBlock(oBx, oBy, oBw, oBh, S_q, d_v,
        qTileRow, 0, actBr, d_v, '#3498db', 'O_i', false);

    // --- K_new / V_new highlight (when S_q < S, the last S_q tokens are "new") ---
    if (S_q < S) {
        const newFrac = S_q / S;

        // K^T: new portion = rightmost S_q columns
        const knewX = ktBx + ktBw * (1 - newFrac);
        const knewW = ktBw * newFrac;
        sectionG.append('rect')
            .attr('x', knewX).attr('y', ktBy)
            .attr('width', knewW).attr('height', ktBh)
            .attr('fill', 'none')
            .attr('stroke', '#fff').attr('stroke-width', 1)
            .attr('stroke-dasharray', '3,2').attr('stroke-opacity', 0.4)
            .attr('rx', 1);
        if (knewW >= 24) {
            sectionG.append('text')
                .attr('x', knewX + knewW / 2).attr('y', ktBy - 2)
                .attr('text-anchor', 'middle')
                .attr('fill', '#888').attr('font-size', '11px')
                .text('K_new');
        }

        // V: new portion = bottom S_q rows
        const vnewH = vBh * newFrac;
        sectionG.append('rect')
            .attr('x', vBx).attr('y', vBy + vBh * (1 - newFrac))
            .attr('width', vBw).attr('height', vnewH)
            .attr('fill', 'none')
            .attr('stroke', '#fff').attr('stroke-width', 1)
            .attr('stroke-dasharray', '3,2').attr('stroke-opacity', 0.4)
            .attr('rx', 1);
        if (vnewH >= 12) {
            sectionG.append('text')
                .attr('x', vBx - 6).attr('y', vBy + vBh * (1 - newFrac) + vnewH / 2 + 3)
                .attr('text-anchor', 'end')
                .attr('fill', '#888').attr('font-size', '11px')
                .text('V_new');
        }
    }

    // --- Tensor name labels ---
    sectionG.append('text')
        .attr('x', ktBx + ktBw / 2).attr('y', ktBy - 5)
        .attr('text-anchor', 'middle')
        .attr('fill', '#2ecc71').attr('font-size', '12px').attr('font-weight', '600')
        .text(K.label + '\u1d40');
    sectionG.append('text')
        .attr('x', vBx + vBw / 2).attr('y', vBy - 5)
        .attr('text-anchor', 'middle')
        .attr('fill', vColor).attr('font-size', '12px').attr('font-weight', '600')
        .text(V.label);
    sectionG.append('text')
        .attr('x', qBx - 6).attr('y', qBy + qBh / 2 + 3)
        .attr('text-anchor', 'end')
        .attr('fill', '#e74c3c').attr('font-size', '12px').attr('font-weight', '600')
        .text(Q.label);

    // --- Dimension labels on edges ---
    sectionG.append('text')
        .attr('x', qBx - 6).attr('y', qBy + qBh - 4)
        .attr('text-anchor', 'end')
        .attr('fill', '#666').attr('font-size', '11px')
        .text(`S_q=${S_q}`);
    sectionG.append('text')
        .attr('x', qBx + qBw / 2).attr('y', qBy + qBh + 12)
        .attr('text-anchor', 'middle')
        .attr('fill', '#666').attr('font-size', '11px')
        .text(`d=${d_q}`);
    sectionG.append('text')
        .attr('x', ktBx - 6).attr('y', ktBy + ktBh / 2 + 3)
        .attr('text-anchor', 'end')
        .attr('fill', '#666').attr('font-size', '11px')
        .text(`d=${d_k}`);
    sectionG.append('text')
        .attr('x', sBx + sBw / 2).attr('y', sBy + sBh + 12)
        .attr('text-anchor', 'middle')
        .attr('fill', '#666').attr('font-size', '11px')
        .text(`S=${S}`);
    sectionG.append('text')
        .attr('x', vBx - 6).attr('y', vBy + vBh / 2 + 3)
        .attr('text-anchor', 'end')
        .attr('fill', '#666').attr('font-size', '11px')
        .text(`S=${S}`);
    sectionG.append('text')
        .attr('x', oBx + oBw / 2).attr('y', oBy + oBh + 12)
        .attr('text-anchor', 'middle')
        .attr('fill', '#666').attr('font-size', '11px')
        .text(`d=${d_v}`);

    // --- Softmax label (rotated, in the gap between S/P and O columns) ---
    const softX = sBx + sBw + softmaxGap / 2;
    const softY = sBy + sBh / 2;
    sectionG.append('text')
        .attr('x', softX).attr('y', softY)
        .attr('text-anchor', 'middle')
        .attr('fill', '#f39c12').attr('font-size', '12px').attr('font-weight', '600')
        .attr('transform', `rotate(-90, ${softX}, ${softY})`)
        .text('softmax \u2192');

    // --- Equation labels below ---
    const bottomY = resultY + pxSq;
    const eqY = bottomY + 30;
    sectionG.append('text')
        .attr('x', width / 2).attr('y', eqY)
        .attr('text-anchor', 'middle')
        .attr('fill', '#888').attr('font-size', '12px')
        .text('S = Q_i \u00d7 K_j\u1d40     \u2192     softmax     \u2192     O_i = P_ij \u00d7 V_j');

    // Note about S/P never being materialized
    let noteY = eqY + 18;
    sectionG.append('text')
        .attr('x', width / 2).attr('y', noteY)
        .attr('text-anchor', 'middle')
        .attr('fill', '#9b59b6').attr('font-size', '11px').attr('font-style', 'italic')
        .text('S/P (dashed) is never materialized to HBM \u2014 computed and consumed in SRAM per tile');

    let totalH = noteY + 8;
    if (sameKV) {
        totalH += 14;
        sectionG.append('text')
            .attr('x', width / 2).attr('y', noteY + 14)
            .attr('text-anchor', 'middle')
            .attr('fill', '#666').attr('font-size', '11px').attr('font-style', 'italic')
            .text('Absorbed MLA \u2014 K and V share the same latent');
    }

    return y + totalH;
}

// --- Section C: Tile Computation View ---

function drawTileComputation(g, x, y, width, params, Q, K, V, O,
                             S_q, S, d_q, d_k, d_v, selQ, selKV, numKVBlocks, isGQA, G, n_kv) {
    const Br = params.block_q, Bc = params.block_kv;
    const actBr = Math.min(Br, S_q - selQ * Br);
    const actBc = Math.min(Bc, S - selKV * Bc);
    const packed = isGQA && params.packGQA;

    const sectionG = g.append('g').attr('transform', `translate(${x}, ${y})`);

    sectionG.append('text')
        .attr('x', 0).attr('y', 14)
        .attr('fill', packed ? '#e67e22' : '#bbb').attr('font-size', '12px').attr('font-weight', '600')
        .text(`Tile Computation — CTA ${params.splitKV ? selQ * numKVBlocks + selKV : selQ}, ` +
              `iteration ${selKV + 1}/${numKVBlocks}`);

    const qRowStart = selQ * Br, qRowEnd = qRowStart + actBr - 1;
    const kvRowStart = selKV * Bc, kvRowEnd = kvRowStart + actBc - 1;
    let tileSubtext = `Q rows ${qRowStart}:${qRowEnd}, KV rows ${kvRowStart}:${kvRowEnd}`;
    const _tileEffDp = Math.min(params.dp_size || 1, params.B || 1);
    if (_tileEffDp > 1) {
        const B = params.B || 1;
        const cumSqTile = [0];
        const sqLens = params.queryLens ? params.queryLens.slice(0, B) : [S_q];
        for (let r = 0; r < sqLens.length; r++) cumSqTile.push(cumSqTile[r] + sqLens[r]);
        const qMid = (selQ + 0.5) * Br;
        for (let dp = 0; dp < _tileEffDp; dp++) {
            const s = cumSqTile[_dpRankStartReq(dp, _tileEffDp, B)];
            const e = cumSqTile[_dpRankEndReq(dp, _tileEffDp, B)];
            if (qMid >= s && qMid < e) { tileSubtext += ` — DP rank ${dp}`; break; }
        }
    }
    sectionG.append('text')
        .attr('x', 0).attr('y', 26)
        .attr('fill', '#666').attr('font-size', '11px')
        .text(tileSubtext);

    const topY = 36;
    const midX = width / 2;
    const sameKV = K.id === V.id;

    const allDims = [actBr, actBc, d_q, d_k, d_v];
    const maxDim = Math.max(...allDims);
    const minPx = 36;
    const maxPx = packed ? 90 : 120;
    function dimToPx(d) { return Math.max(minPx, (d / maxDim) * maxPx); }

    const qW = dimToPx(d_q), qH = dimToPx(actBr);
    const kW = dimToPx(d_k), kH = dimToPx(actBc);
    const vW = dimToPx(d_v), vH = dimToPx(actBc);
    const sW = dimToPx(actBc), sH = dimToPx(actBr);
    const oW = dimToPx(d_v), oH = dimToPx(actBr);

    // --- PackGQA: show G stacked query heads on the left ---
    const numShown = packed ? Math.min(G, 4) : 1; // show up to 4 heads
    const stackOff = packed ? 5 : 0; // px offset per stacked head
    const totalStackH = (numShown - 1) * stackOff;

    // Left: Q_i block(s) (loaded once from HBM)
    const qX = PAD, qY = topY + 10;
    if (packed) {
        // Draw stacked Q tiles for each head in the group
        for (let hi = numShown - 1; hi >= 0; hi--) {
            const ox = hi * stackOff;
            const oy = -hi * stackOff;
            const opacity = hi === 0 ? 0.7 : 0.25;
            drawTileBlock(sectionG, qX + ox, qY + oy, qW, qH, '#e74c3c',
                hi === 0 ? `Q_i^(h0)` : `Q_i^(h${hi})`, hi === 0 ? `[${actBr}, ${d_q}]` : null, opacity);
        }
        sectionG.append('text').attr('x', qX + qW / 2 + totalStackH / 2).attr('y', qY + qH + 14)
            .attr('text-anchor', 'middle').attr('fill', '#e67e22').attr('font-size', '11px')
            .text(`${G} heads \u2192 SMEM`);
    } else {
        drawTileBlock(sectionG, qX, qY, qW, qH, '#e74c3c', `Q_i`, `[${actBr}, ${d_q}]`);
        sectionG.append('text').attr('x', qX + qW / 2).attr('y', qY + qH + 14)
            .attr('text-anchor', 'middle').attr('fill', '#888').attr('font-size', '11px')
            .text('HBM \u2192 SMEM (once)');
    }

    // Right: K_j and V_j blocks (loaded each iter — shared across G heads when packed)
    const kvX = width - Math.max(kW, vW) - PAD;
    const kjY = topY;
    drawTileBlock(sectionG, kvX, kjY, kW, kH, '#2ecc71', `K_j`, `[${actBc}, ${d_k}]`);
    const kvNote = packed ? `HBM \u2192 SMEM (shared \u00d7${G})` : 'HBM \u2192 SMEM (each iter)';
    sectionG.append('text').attr('x', kvX + kW / 2).attr('y', kjY + kH + 14)
        .attr('text-anchor', 'middle').attr('fill', packed ? '#e67e22' : '#888').attr('font-size', '11px')
        .text(kvNote);

    const vjY = kjY + kH + 32;
    if (!sameKV) {
        drawTileBlock(sectionG, kvX, vjY, vW, vH, '#f39c12', `V_j`, `[${actBc}, ${d_v}]`);
        sectionG.append('text').attr('x', kvX + vW / 2).attr('y', vjY + vH + 14)
            .attr('text-anchor', 'middle').attr('fill', packed ? '#e67e22' : '#888').attr('font-size', '11px')
            .text(kvNote);
    }

    // Center: SRAM computation
    const sramX = midX - sW / 2;
    const sramY1 = topY + 5;

    if (packed) {
        // Show stacked S/P tiles for each head
        for (let hi = numShown - 1; hi >= 0; hi--) {
            const ox = hi * stackOff;
            const oy = -hi * stackOff;
            drawTileBlock(sectionG, sramX + ox, sramY1 + oy, sW, sH, '#9b59b6',
                hi === 0 ? `S_ij^(h0)` : '', null, hi === 0 ? 0.7 : 0.25);
        }
    } else {
        drawTileBlock(sectionG, sramX, sramY1, sW, sH, '#9b59b6', `S_ij`, `[${actBr}, ${actBc}]`);
    }
    sectionG.append('text').attr('x', midX).attr('y', sramY1 - 6)
        .attr('text-anchor', 'middle').attr('fill', '#9b59b6').attr('font-size', '11px').attr('font-weight', '600')
        .text('SRAM only');

    // Arrow Q -> S
    drawArrow(sectionG, qX + qW + totalStackH, qY + qH / 2, sramX, sramY1 + sH / 2, '#e74c3c');
    // Arrow K -> S
    drawArrow(sectionG, kvX, kjY + kH * 0.4, sramX + sW + totalStackH, sramY1 + sH / 2, '#2ecc71');

    // Softmax label
    const softY = sramY1 + sH + 8;
    sectionG.append('text').attr('x', midX).attr('y', softY + 10)
        .attr('text-anchor', 'middle').attr('fill', '#f39c12').attr('font-size', '12px').attr('font-weight', '600')
        .text('mask + softmax');
    sectionG.append('line')
        .attr('x1', midX).attr('y1', sramY1 + sH).attr('x2', midX).attr('y2', softY)
        .attr('stroke', '#555').attr('stroke-width', 1);

    // P_ij
    const pY = softY + 18;
    if (packed) {
        for (let hi = numShown - 1; hi >= 0; hi--) {
            const ox = hi * stackOff;
            const oy = -hi * stackOff;
            drawTileBlock(sectionG, sramX + ox, pY + oy, sW, sH, '#9b59b6',
                hi === 0 ? `P_ij^(h0)` : '', null, hi === 0 ? 0.7 : 0.25);
        }
    } else {
        drawTileBlock(sectionG, sramX, pY, sW, sH, '#9b59b6', `P_ij`, `[${actBr}, ${actBc}]`);
    }

    sectionG.append('line')
        .attr('x1', midX).attr('y1', softY + 12).attr('x2', midX).attr('y2', pY)
        .attr('stroke', '#555').attr('stroke-width', 1);

    // Arrow V -> P
    if (!sameKV) {
        drawArrow(sectionG, kvX, vjY + vH * 0.4, sramX + sW + totalStackH, pY + sH * 0.4, '#f39c12');
    } else {
        drawArrow(sectionG, kvX, kjY + kH * 0.6, sramX + sW + totalStackH, pY + sH * 0.4, '#f39c12');
    }

    // O_i accumulator(s)
    const oY = pY + sH + 20;
    const oX = midX - oW / 2;
    if (packed) {
        for (let hi = numShown - 1; hi >= 0; hi--) {
            const ox = hi * stackOff;
            const oy = -hi * stackOff;
            drawTileBlock(sectionG, oX + ox, oY + oy, oW, oH, '#3498db',
                hi === 0 ? `O_i^(h0)` : '', hi === 0 ? `[${actBr}, ${d_v}]` : null, hi === 0 ? 0.7 : 0.25);
        }
        sectionG.append('text').attr('x', midX).attr('y', oY + oH + 14)
            .attr('text-anchor', 'middle').attr('fill', '#e67e22').attr('font-size', '11px')
            .text(`${G} accumulators (registers)`);
    } else {
        drawTileBlock(sectionG, oX, oY, oW, oH, '#3498db', `O_i`, `[${actBr}, ${d_v}]`);
        sectionG.append('text').attr('x', midX).attr('y', oY + oH + 14)
            .attr('text-anchor', 'middle').attr('fill', '#888').attr('font-size', '11px')
            .text('Accumulator (registers)');
    }

    // Arrow P -> O
    sectionG.append('line')
        .attr('x1', midX).attr('y1', pY + sH).attr('x2', midX).attr('y2', oY)
        .attr('stroke', '#555').attr('stroke-width', 1);

    // O += P_ij @ V_j label
    sectionG.append('text').attr('x', midX + Math.max(sW, oW) / 2 + totalStackH + 15).attr('y', pY + sH + 10)
        .attr('fill', '#3498db').attr('font-size', '11px')
        .text(packed ? 'O_i^(h) += P_ij^(h) \u00d7 V_j' : 'O_i += P_ij \u00d7 V_j');

    // Write-back arrow
    const wbY = oY + oH + 30;
    let wbText = params.splitKV ? 'Write partial O_i \u2192 HBM' : 'Write O_i \u2192 HBM (final iter)';
    if (packed) wbText = params.splitKV ? `Write ${G}\u00d7 partial O_i \u2192 HBM` : `Write ${G}\u00d7 O_i \u2192 HBM (final iter)`;
    sectionG.append('text').attr('x', midX).attr('y', wbY + 4)
        .attr('text-anchor', 'middle').attr('fill', '#3498db').attr('font-size', '11px').attr('font-weight', '600')
        .text(wbText);

    // Online softmax state
    const stateX = midX + Math.max(sW, oW) / 2 + totalStackH + 20;
    const stateY = oY;
    sectionG.append('text').attr('x', stateX).attr('y', stateY + 10)
        .attr('fill', '#666').attr('font-size', '11px').attr('font-style', 'italic')
        .text('Online softmax state:');
    sectionG.append('text').attr('x', stateX).attr('y', stateY + 22)
        .attr('fill', '#888').attr('font-size', '11px')
        .text(packed ? `m_i [${G}\u00d7${actBr}] — row max` : `m_i [${actBr}] — row max`);
    sectionG.append('text').attr('x', stateX).attr('y', stateY + 34)
        .attr('fill', '#888').attr('font-size', '11px')
        .text(packed ? `l_i [${G}\u00d7${actBr}] — row sum(exp)` : `l_i [${actBr}] — row sum(exp)`);

    return y + wbY + 16;
}

// --- Section C: Memory Hierarchy ---

function drawMemoryHierarchy(g, x, y, width, params, Q, K, V, O,
                             S_q, S, d_q, d_k, d_v, selQ, selKV, isGQA, G, n_kv) {
    const Br = params.block_q, Bc = params.block_kv;
    const actBr = Math.min(Br, S_q - selQ * Br);
    const actBc = Math.min(Bc, S - selKV * Bc);
    const packed = isGQA && params.packGQA;
    const sectionG = g.append('g').attr('transform', `translate(${x}, ${y})`);

    sectionG.append('text')
        .attr('x', 0).attr('y', 14)
        .attr('fill', '#bbb').attr('font-size', '12px').attr('font-weight', '600')
        .text('Memory Hierarchy');

    const topY = 26;
    const bandH = 76;
    const bandGap = 28;
    const bandW = width;

    const qSmemLabel = packed ? `Q_i \u00d7${G}` : 'Q_i';
    const qSmemShape = packed ? `${G}\u00d7${actBr}\u00d7${d_q}` : `${actBr}\u00d7${d_q}`;
    const qSmemBytes = (packed ? G : 1) * actBr * d_q * 2;

    const sRegLabel = packed ? `S_ij \u00d7${G}` : 'S_ij';
    const sRegShape = packed ? `${G}\u00d7${actBr}\u00d7${actBc}` : `${actBr}\u00d7${actBc}`;
    const sRegBytes = (packed ? G : 1) * actBr * actBc * 2;

    const oRegLabel = packed ? `O_i \u00d7${G}` : 'O_i';
    const oRegShape = packed ? `${G}\u00d7${actBr}\u00d7${d_v}` : `${actBr}\u00d7${d_v}`;
    const oRegBytes = (packed ? G : 1) * actBr * d_v * 2;

    const mlRegLabel = packed ? `m_i, l_i \u00d7${G}` : 'm_i, l_i';
    const mlRegShape = packed ? `2\u00d7${G}\u00d7${actBr}` : `2\u00d7${actBr}`;
    const mlRegBytes = (packed ? G : 1) * actBr * 2 * 2;

    const tiers = [
        {
            label: 'HBM (Global Memory)',
            color: '#1a2a4a',
            border: '#2a4a6a',
            items: [
                { label: Q.label, shape: Q.shape.join('\u00d7'), bytes: tensorBytes(Q.shape), color: '#e74c3c' },
                { label: K.label, shape: K.shape.join('\u00d7'), bytes: tensorBytes(K.shape), color: '#2ecc71' },
                ...(V.id !== K.id ? [{ label: V.label, shape: V.shape.join('\u00d7'), bytes: tensorBytes(V.shape), color: '#f39c12' }] : []),
                { label: O.label, shape: O.shape.join('\u00d7'), bytes: tensorBytes(O.shape), color: '#3498db' },
            ]
        },
        {
            label: packed ? `SMEM (Shared Memory) — ${G} heads packed` : 'SMEM (Shared Memory)',
            color: '#1a3a2a',
            border: packed ? '#5a8a3a' : '#2a5a3a',
            items: [
                { label: qSmemLabel, shape: qSmemShape, bytes: qSmemBytes, color: '#e74c3c', note: 'persistent' },
                { label: 'K_j', shape: `${actBc}\u00d7${d_k}`, bytes: actBc * d_k * 2, color: '#2ecc71', note: packed ? `shared \u00d7${G}` : 'rotated' },
                ...(V.id !== K.id ? [{ label: 'V_j', shape: `${actBc}\u00d7${d_v}`, bytes: actBc * d_v * 2, color: '#f39c12', note: packed ? `shared \u00d7${G}` : 'rotated' }] : []),
            ]
        },
        {
            label: 'Registers',
            color: '#2a1a3a',
            border: '#4a2a5a',
            items: [
                { label: sRegLabel, shape: sRegShape, bytes: sRegBytes, color: '#9b59b6' },
                { label: oRegLabel, shape: oRegShape, bytes: oRegBytes, color: '#3498db', note: 'accum' },
                { label: mlRegLabel, shape: mlRegShape, bytes: mlRegBytes, color: '#888' },
            ]
        },
    ];

    const arrowLabels = [
        packed
            ? `load ${G}\u00d7 Q_i once; stream K_j, V_j once (shared); write ${G}\u00d7 O_i`
            : 'load Q_i once; stream K_j, V_j each iter; write O_i once',
        'compute tiles in registers',
    ];

    tiers.forEach((tier, ti) => {
        const bY = topY + ti * (bandH + bandGap);

        // Band background
        sectionG.append('rect')
            .attr('x', 0).attr('y', bY)
            .attr('width', bandW).attr('height', bandH)
            .attr('rx', 4).attr('fill', tier.color).attr('stroke', tier.border).attr('stroke-width', 1);

        // Tier label
        sectionG.append('text')
            .attr('x', 6).attr('y', bY + 13)
            .attr('fill', '#888').attr('font-size', '11px').attr('font-weight', '600')
            .text(tier.label);

        // Items
        const itemY = bY + 20;
        const itemGap = bandW / (tier.items.length + 1);
        tier.items.forEach((item, ii) => {
            const ix = itemGap * (ii + 1);
            let ly = itemY + 8;
            sectionG.append('text').attr('x', ix).attr('y', ly)
                .attr('text-anchor', 'middle').attr('fill', item.color).attr('font-size', '12px').attr('font-weight', '600')
                .text(item.label);
            ly += 14;
            sectionG.append('text').attr('x', ix).attr('y', ly)
                .attr('text-anchor', 'middle').attr('fill', '#666').attr('font-size', '11px')
                .text(`[${item.shape}]`);
            ly += 12;
            sectionG.append('text').attr('x', ix).attr('y', ly)
                .attr('text-anchor', 'middle').attr('fill', '#888').attr('font-size', '11px')
                .text(fmtBytes(item.bytes));
            if (item.note) {
                ly += 12;
                sectionG.append('text').attr('x', ix).attr('y', ly)
                    .attr('text-anchor', 'middle').attr('fill', '#666').attr('font-size', '10px')
                    .text(`(${item.note})`);
            }
        });

        // Arrow between tiers
        if (ti < tiers.length - 1) {
            const arrowY = bY + bandH + 4;
            sectionG.append('text')
                .attr('x', 12).attr('y', arrowY + 14)
                .attr('fill', '#666').attr('font-size', '18px').attr('font-weight', '700')
                .text('\u2195');
            sectionG.append('text')
                .attr('x', bandW / 2).attr('y', arrowY + 12)
                .attr('text-anchor', 'middle').attr('fill', '#555').attr('font-size', '11px')
                .text(arrowLabels[ti]);
            sectionG.append('text')
                .attr('x', bandW - 12).attr('y', arrowY + 14)
                .attr('text-anchor', 'end').attr('fill', '#666').attr('font-size', '18px').attr('font-weight', '700')
                .text('\u2195');
        }
    });

    const totalH = tiers.length * bandH + (tiers.length - 1) * bandGap + topY;
    return y + totalH + 10;
}

// --- Section D: Split-KV Reduction ---

function drawSplitKVReduction(g, x, y, width, params, S_q, numKVBlocks, selQ, O, d_v) {
    const Br = params.block_q;
    const actBr = Math.min(Br, S_q - selQ * Br);
    const sectionG = g.append('g').attr('transform', `translate(${x}, ${y})`);

    sectionG.append('text')
        .attr('x', 0).attr('y', 14)
        .attr('fill', '#bbb').attr('font-size', '12px').attr('font-weight', '600')
        .text('Split-KV Reduction');

    const topY = 28;
    const blockW = 50;
    const blockH = 30;
    const gap = 8;
    const maxShown = Math.min(numKVBlocks, 8);
    const totalBlocksW = maxShown * (blockW + gap);
    const startX = (width - totalBlocksW - blockW - 40) / 2;

    // Partial O blocks from each KV-split CTA
    for (let j = 0; j < maxShown; j++) {
        const bx = startX + j * (blockW + gap);
        drawTileBlock(sectionG, bx, topY, blockW, blockH, '#3498db',
            `O_${j}`, `[${actBr},${d_v}]`, 0.5);
    }
    if (numKVBlocks > maxShown) {
        sectionG.append('text')
            .attr('x', startX + maxShown * (blockW + gap) - gap / 2)
            .attr('y', topY + blockH / 2 + 3)
            .attr('fill', '#888').attr('font-size', '11px')
            .text(`\u2026 (${numKVBlocks})`);
    }

    // Reduction arrow
    const arrowX = startX + totalBlocksW + 10;
    sectionG.append('text')
        .attr('x', arrowX).attr('y', topY + blockH / 2 + 4)
        .attr('fill', '#888').attr('font-size', '16px')
        .text('\u2192');

    // Final O block
    const finalX = arrowX + 30;
    drawTileBlock(sectionG, finalX, topY, blockW + 10, blockH, '#2980b9', 'O_i', `[${actBr},${d_v}]`, 0.9);

    // Explanation
    sectionG.append('text')
        .attr('x', 0).attr('y', topY + blockH + 18)
        .attr('fill', '#888').attr('font-size', '11px')
        .text('Partial outputs combined via online softmax correction: rescale by max(m) and sum(l) across splits');

    return y + topY + blockH + 28;
}

// --- Helpers ---

function _dpRankStartReq(dp, effectiveDp, B) {
    for (let r = 0; r < B; r++) {
        if (Math.floor(r * effectiveDp / B) === dp) return r;
    }
    return B;
}

function _dpRankEndReq(dp, effectiveDp, B) {
    for (let r = B - 1; r >= 0; r--) {
        if (Math.floor(r * effectiveDp / B) === dp) return r + 1;
    }
    return 0;
}

function drawTileBlock(g, x, y, w, h, color, label, shapeStr, opacity) {
    g.append('rect')
        .attr('x', x).attr('y', y).attr('width', w).attr('height', h)
        .attr('fill', color).attr('fill-opacity', opacity != null ? opacity : 0.7)
        .attr('stroke', d3.color(color).darker(0.3)).attr('stroke-width', 1)
        .attr('rx', 3);
    g.append('text')
        .attr('x', x + w / 2).attr('y', y + h / 2 + (shapeStr ? -1 : 3))
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff').attr('font-size', '12px').attr('font-weight', '600')
        .text(label);
    if (shapeStr) {
        g.append('text')
            .attr('x', x + w / 2).attr('y', y + h / 2 + 11)
            .attr('text-anchor', 'middle')
            .attr('fill', '#ddd').attr('font-size', '12px')
            .text(shapeStr);
    }
}

function drawArrow(g, x1, y1, x2, y2, color) {
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy);
    const ux = dx / len, uy = dy / len;
    // Shorten by 4px on each end
    const sx = x1 + ux * 4, sy = y1 + uy * 4;
    const ex = x2 - ux * 8, ey = y2 - uy * 8;

    g.append('line')
        .attr('x1', sx).attr('y1', sy).attr('x2', ex).attr('y2', ey)
        .attr('stroke', color || '#555').attr('stroke-width', 1.5)
        .attr('stroke-opacity', 0.6);

    // Arrowhead
    const headLen = 6;
    const angle = Math.atan2(ey - sy, ex - sx);
    g.append('polygon')
        .attr('points', [
            `${ex},${ey}`,
            `${ex - headLen * Math.cos(angle - 0.4)},${ey - headLen * Math.sin(angle - 0.4)}`,
            `${ex - headLen * Math.cos(angle + 0.4)},${ey - headLen * Math.sin(angle + 0.4)}`,
        ].join(' '))
        .attr('fill', color || '#555').attr('fill-opacity', 0.6);
}
