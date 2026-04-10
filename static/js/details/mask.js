// mask.js — Standard mask, mask tensor, and paged mask tensor detail visualizations
import { detailMetrics } from './shared.js';
import { drawSoftmaxSection } from './softmax.js';

// Deterministic pseudo-random score for cell (i, j) — looks natural, stable across rerenders
function pseudoScore(i, j) {
    let h = (i * 374761 + j * 668265) ^ 0x5bd1e995;
    h = Math.imul(h ^ (h >>> 13), 0x5bd1e995);
    h = h ^ (h >>> 15);
    return 0.3 + ((h >>> 0) / 0xffffffff) * 1.4;  // range [0.3, 1.7]
}

// Compute cell size and label density for a given grid
function maskLayout(svgW, rows, cols) {
    const maxGridW = svgW - 80;  // leave room for row labels + padding
    const maxGridH = 300;        // don't let it get taller than this
    const cellSize = Math.max(2, Math.min(28, maxGridW / cols, maxGridH / rows));

    // Decide label density based on cell size
    let labelEvery;
    if (cellSize >= 16) labelEvery = 1;
    else if (cellSize >= 8) labelEvery = Math.ceil(5 / cellSize) * 2;
    else if (cellSize >= 4) labelEvery = Math.ceil(20 / cellSize);
    else labelEvery = 0; // no labels at all

    return { cellSize, labelEvery };
}

// Draw axis labels at the given density
function drawAxisLabels(g, rows, cols, cellSize, labelEvery, y0, rowOffset) {
    if (!labelEvery) return;
    const fontSize = Math.min(8, Math.max(5, cellSize * 0.6)) + 'px';
    for (let i = 0; i < rows; i++) {
        if (i % labelEvery !== 0 && i !== rows - 1) continue;
        g.append('text').attr('class', 'dim-label')
            .attr('x', -4).attr('y', y0 + i * cellSize + cellSize / 2 + 3)
            .attr('text-anchor', 'end').attr('font-size', fontSize).text(i + rowOffset);
    }
    for (let j = 0; j < cols; j++) {
        if (j % labelEvery !== 0 && j !== cols - 1) continue;
        g.append('text').attr('class', 'dim-label')
            .attr('x', j * cellSize + cellSize / 2).attr('y', y0 - 3)
            .attr('text-anchor', 'middle').attr('font-size', fontSize).text(j);
    }
}

export function drawMaskDetail(svg, _op, _tensorMap, params) {
    const { w: svgW } = detailMetrics();
    const S = params.S;
    const S_q = params.S_q || S;
    const { cellSize, labelEvery } = maskLayout(svgW, S_q, S);
    const gridW = S * cellSize;
    const gridH = S_q * cellSize;
    const pad = Math.max(30, (svgW - gridW - 60) / 2);

    const g = svg.append('g').attr('transform', `translate(${pad}, 38)`);

    // --- Part 1: Causal mask ---
    g.append('text').attr('class', 'tensor-label')
        .attr('x', gridW / 2).attr('y', -14)
        .text(`Causal Mask (${S_q}\u00d7${S})`);

    // When S_q < S (decode), query positions are at the end of the sequence
    const queryOffset = S - S_q;
    const showCellText = cellSize >= 16;
    for (let i = 0; i < S_q; i++) {
        for (let j = 0; j < S; j++) {
            const allowed = (i + queryOffset) >= j;
            g.append('rect')
                .attr('x', j * cellSize).attr('y', i * cellSize + 10)
                .attr('width', cellSize - (cellSize > 3 ? 1 : 0)).attr('height', cellSize - (cellSize > 3 ? 1 : 0))
                .attr('rx', cellSize >= 6 ? 2 : 0)
                .attr('fill', allowed ? '#1abc9c' : '#2c3e50')
                .attr('fill-opacity', allowed ? 0.85 : 0.5)
                .attr('stroke', cellSize >= 4 ? '#1a1d2a' : 'none').attr('stroke-width', 0.5);

            if (showCellText) {
                g.append('text')
                    .attr('x', j * cellSize + cellSize / 2)
                    .attr('y', i * cellSize + cellSize / 2 + 13)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '8px')
                    .attr('fill', allowed ? '#fff' : '#555')
                    .text(allowed ? '1' : '-\u221e');
            }
        }
    }

    drawAxisLabels(g, S_q, S, cellSize, labelEvery, 10, queryOffset);

    const maskLegendY = gridH + 26;
    g.append('rect').attr('x', 0).attr('y', maskLegendY).attr('width', 12).attr('height', 12)
        .attr('fill', '#1abc9c').attr('fill-opacity', 0.85).attr('rx', 2);
    g.append('text').attr('class', 'dim-label').attr('x', 18).attr('y', maskLegendY + 10)
        .attr('fill', '#aaa').text('Attend (i \u2265 j)');

    g.append('rect').attr('x', 140).attr('y', maskLegendY).attr('width', 12).attr('height', 12)
        .attr('fill', '#2c3e50').attr('fill-opacity', 0.5).attr('rx', 2);
    g.append('text').attr('class', 'dim-label').attr('x', 158).attr('y', maskLegendY + 10)
        .attr('fill', '#aaa').text('Masked (i < j) \u2192 -\u221e');

    // --- Part 2: Attention weights heatmap (after softmax) ---
    let y2 = maskLegendY + 46;

    g.append('text').attr('class', 'tensor-label')
        .attr('x', gridW / 2).attr('y', y2)
        .text('Attention Weights (after softmax)');
    y2 += 26;

    // Use same adaptive layout as the mask grid
    const heat = maskLayout(svgW, S_q, S);
    const heatCell = heat.cellSize;
    const heatLabelEvery = heat.labelEvery;
    const heatGridW = S * heatCell;
    const heatGridH = S_q * heatCell;
    const showWeightText = heatCell >= 20;

    // Compute softmax attention weights for all rows
    const attnWeights = [];
    for (let i = 0; i < S_q; i++) {
        const rawScores = [];
        for (let j = 0; j < S; j++) {
            if (j <= i + queryOffset) {
                rawScores.push(pseudoScore(i, j));
            } else {
                rawScores.push(-Infinity);
            }
        }
        const exps = rawScores.map(s => s === -Infinity ? 0 : Math.exp(s));
        const sumExp = exps.reduce((a, b) => a + b, 0);
        attnWeights.push(exps.map(e => e / sumExp));
    }

    const maxWeight = Math.max(...attnWeights.flat());

    for (let i = 0; i < S_q; i++) {
        for (let j = 0; j < S; j++) {
            const w = attnWeights[i][j];
            const intensity = maxWeight > 0 ? w / maxWeight : 0;
            g.append('rect')
                .attr('x', j * heatCell).attr('y', y2 + i * heatCell)
                .attr('width', heatCell - (heatCell > 3 ? 1 : 0)).attr('height', heatCell - (heatCell > 3 ? 1 : 0))
                .attr('rx', heatCell >= 6 ? 2 : 0)
                .attr('fill', w > 0 ? '#f39c12' : '#1a1d2a')
                .attr('fill-opacity', w > 0 ? 0.15 + intensity * 0.75 : 0.3)
                .attr('stroke', heatCell >= 4 ? '#1a1d2a' : 'none').attr('stroke-width', 0.5);

            if (showWeightText && w > 0.01) {
                g.append('text')
                    .attr('x', j * heatCell + heatCell / 2)
                    .attr('y', y2 + i * heatCell + heatCell / 2 + 3)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '7px')
                    .attr('fill', intensity > 0.5 ? '#fff' : '#ccc')
                    .text(w.toFixed(2));
            }
        }
    }

    drawAxisLabels(g, S_q, S, heatCell, heatLabelEvery, y2, queryOffset);

    // "each row sums to 1" annotation
    g.append('text').attr('class', 'dim-label')
        .attr('x', heatGridW / 2).attr('y', y2 + heatGridH + 14)
        .attr('text-anchor', 'middle').attr('fill', '#f39c12')
        .attr('font-size', '10px')
        .text('each row sums to 1');

    const heatBottom = y2 + heatGridH + 30;

    // Color scale legend
    const heatLegendY = heatBottom + 4;
    g.append('text').attr('class', 'dim-label')
        .attr('x', 0).attr('y', heatLegendY)
        .attr('fill', '#888').attr('font-size', '9px')
        .text('Intensity = attention weight:');

    const gradW = 120;
    const gradY = heatLegendY + 6;
    for (let k = 0; k < 20; k++) {
        g.append('rect')
            .attr('x', k * (gradW / 20)).attr('y', gradY)
            .attr('width', gradW / 20).attr('height', 10)
            .attr('fill', '#f39c12')
            .attr('fill-opacity', 0.15 + (k / 19) * 0.75);
    }
    g.append('text').attr('class', 'dim-label')
        .attr('x', 0).attr('y', gradY + 22)
        .attr('font-size', '8px').attr('fill', '#888').text('0');
    g.append('text').attr('class', 'dim-label')
        .attr('x', gradW).attr('y', gradY + 22)
        .attr('text-anchor', 'end').attr('font-size', '8px').attr('fill', '#888').text('max');

    // --- Part 3: Sample row bar chart ---
    const softmaxBottom = drawSoftmaxSection(g, 0, gradY + 52, S, heatCell, attnWeights);

    svg.attr('height', softmaxBottom + 10);
}

// --- Mask tensor detail (when clicking the mask tensor directly) ---

export function drawMaskTensorDetail(svg, _tensor, params) {
    const { w: svgW } = detailMetrics();
    const S = params.S;
    const S_q = params.S_q || S;
    const { cellSize, labelEvery } = maskLayout(svgW, S_q, S);
    const gridW = S * cellSize;
    const gridH = S_q * cellSize;
    const pad = Math.max(30, (svgW - gridW - 60) / 2);

    const g = svg.append('g').attr('transform', `translate(${pad}, 38)`);

    g.append('text').attr('class', 'tensor-label')
        .attr('x', gridW / 2).attr('y', -14)
        .text(`Causal Mask (${S_q}\u00d7${S})`);

    const queryOffset = S - S_q;
    const showCellText = cellSize >= 16;
    for (let i = 0; i < S_q; i++) {
        for (let j = 0; j < S; j++) {
            const allowed = (i + queryOffset) >= j;
            g.append('rect')
                .attr('x', j * cellSize).attr('y', i * cellSize + 10)
                .attr('width', cellSize - (cellSize > 3 ? 1 : 0)).attr('height', cellSize - (cellSize > 3 ? 1 : 0))
                .attr('rx', cellSize >= 6 ? 2 : 0)
                .attr('fill', allowed ? '#1abc9c' : '#2c3e50')
                .attr('fill-opacity', allowed ? 0.85 : 0.5)
                .attr('stroke', cellSize >= 4 ? '#1a1d2a' : 'none').attr('stroke-width', 0.5);

            if (showCellText) {
                g.append('text')
                    .attr('x', j * cellSize + cellSize / 2)
                    .attr('y', i * cellSize + cellSize / 2 + 13)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '8px')
                    .attr('fill', allowed ? '#fff' : '#555')
                    .text(allowed ? '1' : '-\u221e');
            }
        }
    }

    drawAxisLabels(g, S_q, S, cellSize, labelEvery, 10, queryOffset);

    const descY = gridH + 26;
    g.append('rect').attr('x', 0).attr('y', descY).attr('width', 12).attr('height', 12)
        .attr('fill', '#1abc9c').attr('fill-opacity', 0.85).attr('rx', 2);
    g.append('text').attr('class', 'dim-label').attr('x', 18).attr('y', descY + 10)
        .attr('fill', '#aaa').text('Attend (i \u2265 j): token can see this position');

    g.append('rect').attr('x', 140).attr('y', descY).attr('width', 12).attr('height', 12)
        .attr('fill', '#2c3e50').attr('fill-opacity', 0.5).attr('rx', 2);
    g.append('text').attr('class', 'dim-label').attr('x', 158).attr('y', descY + 10)
        .attr('fill', '#aaa').text('Masked (i < j) \u2192 -\u221e');

    svg.attr('height', descY + 30);
}

// --- Paged mask tensor detail (clicking the mask tensor when paged attention is on) ---

export function drawPagedMaskTensorDetail(svg, _tensor, params) {
    const { w: svgW } = detailMetrics();
    const sLens = params.seqLens.slice(0, params.B);               // S per request (columns)
    const sqLens = params.queryLens.slice(0, params.B).map(q => q || 1); // S_q per request (rows)

    const dispCols = sLens.reduce((a, b) => a + b, 0);
    const dispRows = sqLens.reduce((a, b) => a + b, 0);
    const cellSize = Math.min(22, Math.max(10, (svgW - 100) / dispCols));
    const gridW = dispCols * cellSize;
    const gridH = dispRows * cellSize;
    const pad = Math.max(56, (svgW - gridW - 60) / 2);

    const g = svg.append('g').attr('transform', `translate(${pad}, 38)`);

    g.append('text').attr('class', 'tensor-label')
        .attr('x', gridW / 2).attr('y', -14)
        .text('Variable-Length Causal Mask');

    // Draw block-diagonal mask grid: rows = S_q (query tokens), cols = S (full KV)
    let rowOff = 0;
    for (let si = 0; si < sqLens.length; si++) {
        const nq = sqLens[si];
        let colOff = 0;

        for (let sj = 0; sj < sLens.length; sj++) {
            const ns = sLens[sj];

            for (let i = 0; i < nq; i++) {
                for (let j = 0; j < ns; j++) {
                    let fill, opacity;
                    if (si !== sj) {
                        // Cross-sequence: always blocked
                        fill = '#1a1520'; opacity = 0.8;
                    } else {
                        // Same sequence: causal within the query's view of the full KV
                        // Query token i attends to KV positions 0..(S - S_q + i)
                        const allowed = j <= (ns - nq + i);
                        fill = allowed ? '#1abc9c' : '#2c3e50';
                        opacity = allowed ? 0.85 : 0.5;
                    }

                    g.append('rect')
                        .attr('x', (colOff + j) * cellSize).attr('y', (rowOff + i) * cellSize + 10)
                        .attr('width', cellSize - 1).attr('height', cellSize - 1)
                        .attr('rx', 1)
                        .attr('fill', fill).attr('fill-opacity', opacity)
                        .attr('stroke', '#1a1d2a').attr('stroke-width', 0.3);
                }
            }
            colOff += sLens[sj];
        }

        // Horizontal boundary line after this request's query rows
        if (si < sqLens.length - 1) {
            g.append('line')
                .attr('x1', 0).attr('y1', (rowOff + nq) * cellSize + 10)
                .attr('x2', gridW).attr('y2', (rowOff + nq) * cellSize + 10)
                .attr('stroke', '#e74c3c').attr('stroke-width', 1).attr('stroke-opacity', 0.6);
        }

        rowOff += nq;
    }

    // Vertical boundary lines between sequences
    let vOff = 0;
    for (let sj = 0; sj < sLens.length - 1; sj++) {
        vOff += sLens[sj];
        g.append('line')
            .attr('x1', vOff * cellSize).attr('y1', 10)
            .attr('x2', vOff * cellSize).attr('y2', gridH + 10)
            .attr('stroke', '#e74c3c').attr('stroke-width', 1).attr('stroke-opacity', 0.6);
    }

    // Row labels (S_q side)
    let labelOff = 0;
    for (let si = 0; si < sqLens.length; si++) {
        const nq = sqLens[si];
        const midPos = labelOff + nq / 2;
        g.append('text').attr('class', 'dim-label')
            .attr('x', -8).attr('y', midPos * cellSize + 14)
            .attr('text-anchor', 'end').attr('font-size', '8px').attr('fill', '#aaa')
            .text(`S_q=${sqLens[si]}`);
        labelOff += nq;
    }

    // Column labels (S side)
    let colLabelOff = 0;
    for (let sj = 0; sj < sLens.length; sj++) {
        const ns = sLens[sj];
        const midPos = colLabelOff + ns / 2;
        g.append('text').attr('class', 'dim-label')
            .attr('x', midPos * cellSize).attr('y', 6)
            .attr('text-anchor', 'middle').attr('font-size', '8px').attr('fill', '#aaa')
            .text(`S=${sLens[sj]}`);
        colLabelOff += ns;
    }

    const descY = gridH + 30;

    // Legend
    g.append('rect').attr('x', 0).attr('y', descY).attr('width', 12).attr('height', 12)
        .attr('fill', '#1abc9c').attr('fill-opacity', 0.85).attr('rx', 2);
    g.append('text').attr('class', 'dim-label').attr('x', 18).attr('y', descY + 10)
        .attr('fill', '#aaa').text('Causal attend (same seq, i \u2265 j)');

    g.append('rect').attr('x', 0).attr('y', descY + 18).attr('width', 12).attr('height', 12)
        .attr('fill', '#2c3e50').attr('fill-opacity', 0.5).attr('rx', 2);
    g.append('text').attr('class', 'dim-label').attr('x', 18).attr('y', descY + 28)
        .attr('fill', '#aaa').text('Masked future (same seq, i < j)');

    g.append('rect').attr('x', 0).attr('y', descY + 36).attr('width', 12).attr('height', 12)
        .attr('fill', '#1a1520').attr('fill-opacity', 0.8).attr('rx', 2);
    g.append('text').attr('class', 'dim-label').attr('x', 18).attr('y', descY + 46)
        .attr('fill', '#aaa').text('Cross-sequence (always blocked)');

    g.append('text').attr('class', 'dim-label')
        .attr('x', 0).attr('y', descY + 66)
        .attr('fill', '#777')
        .text(`Per-request: [${sqLens.map((q, i) => `S_q=${q} × S=${sLens[i]}`).join(', ')}]`);

    svg.attr('height', descY + 86);
}
