// mask.js — Standard mask, mask tensor, and paged mask tensor detail visualizations
import { detailMetrics, maskLayout } from './shared.js';

// Deterministic pseudo-random score for cell (i, j) — looks natural, stable across rerenders
function pseudoScore(i, j) {
    let h = (i * 374761 + j * 668265) ^ 0x5bd1e995;
    h = Math.imul(h ^ (h >>> 13), 0x5bd1e995);
    h = h ^ (h >>> 15);
    return -0.8 + ((h >>> 0) / 0xffffffff) * 2.4;  // range [-0.8, 1.6]
}

// Draw a text label with a rounded background pill for readability over busy grids
function drawPillLabel(g, x, y, text, fill) {
    const fontSize = 11;
    const padX = 6, padY = 4;
    const approxW = text.length * fontSize * 0.6 + padX * 2;
    const h = fontSize + padY * 2;
    g.append('rect')
        .attr('x', x - approxW / 2).attr('y', y - h / 2 - 1)
        .attr('width', approxW).attr('height', h)
        .attr('rx', h / 2)
        .attr('fill', '#0e1117').attr('fill-opacity', 0.7);
    g.append('text').attr('x', x).attr('y', y + fontSize * 0.35)
        .attr('text-anchor', 'middle').attr('font-size', fontSize + 'px')
        .attr('fill', fill).attr('fill-opacity', 0.95).text(text);
}

// Draw a schematic causal mask (triangle) instead of individual cells
function drawSchematicCausal(g, gridW, gridH, S_q, S, queryOffset, color, label) {
    const y0 = 10;
    const topAttendX = S_q === S ? 0 : ((queryOffset + 1) / S) * gridW;
    // Background (masked region)
    g.append('rect').attr('x', 0).attr('y', y0)
        .attr('width', gridW).attr('height', gridH)
        .attr('fill', color === 'mask' ? '#2c3e50' : '#1a1d2a')
        .attr('fill-opacity', 0.5).attr('rx', 2);
    // Attend region (lower-left triangle / trapezoid)
    const fill = color === 'mask' ? '#1abc9c' : '#f39c12';
    const opacity = color === 'mask' ? 0.7 : 0.5;
    g.append('polygon')
        .attr('points', `0,${y0} ${topAttendX},${y0} ${gridW},${y0 + gridH} 0,${y0 + gridH}`)
        .attr('fill', fill).attr('fill-opacity', opacity);
    // Diagonal boundary
    g.append('line')
        .attr('x1', topAttendX).attr('y1', y0)
        .attr('x2', gridW).attr('y2', y0 + gridH)
        .attr('stroke', '#fff').attr('stroke-width', 1.5).attr('stroke-opacity', 0.4);
    // Region labels — position at centroids of attend trapezoid / masked triangle
    if (color === 'mask') {
        const attendCx = (topAttendX + gridW) / 4;
        const attendCy = y0 + gridH * 2 / 3;
        drawPillLabel(g, attendCx, attendCy, '1', '#fff');
        if (gridW - topAttendX > 30) {
            const maskedCx = (topAttendX + 2 * gridW) / 3;
            const maskedCy = y0 + gridH / 3;
            drawPillLabel(g, maskedCx, maskedCy, '\u2212\u221e', '#aaa');
        }
    }
    // Corner axis labels
    g.append('text').attr('class', 'dim-label').attr('x', -4).attr('y', y0 + 4)
        .attr('text-anchor', 'end').attr('font-size', '7px').text('0');
    g.append('text').attr('class', 'dim-label').attr('x', -4).attr('y', y0 + gridH)
        .attr('text-anchor', 'end').attr('font-size', '7px').text(S_q - 1);
    g.append('text').attr('class', 'dim-label').attr('x', 0).attr('y', y0 - 3)
        .attr('text-anchor', 'middle').attr('font-size', '7px').text('0');
    g.append('text').attr('class', 'dim-label').attr('x', gridW).attr('y', y0 - 3)
        .attr('text-anchor', 'middle').attr('font-size', '7px').text(S - 1);
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
    // Multi-request: show block-diagonal mask (reuse tensor detail view)
    if (params.B > 1) {
        drawPagedMaskTensorDetail(svg, { type: 'mask' }, params);
        return;
    }
    const { w: svgW } = detailMetrics();
    const S = params.S;
    const S_q = params.S_q || S;
    const { cellSize, labelEvery, schematic } = maskLayout(svgW, S_q, S);
    const gridW = S * cellSize;
    const gridH = S_q * cellSize;
    const pad = Math.max(30, (svgW - gridW - 60) / 2);

    const g = svg.append('g').attr('transform', `translate(${pad}, 38)`);

    const queryOffset = S - S_q;
    const showCellText = cellSize >= 16;

    // Compute scaled scores and softmax weights up front (used by Parts 0 and 2)
    const d_h = params.d_h || 64;
    const scale = 1 / Math.sqrt(d_h);
    const rawScoreGrid = [];
    const attnWeights = [];
    for (let i = 0; i < S_q; i++) {
        const rawRow = [];
        const maskedScores = [];
        for (let j = 0; j < S; j++) {
            const s = pseudoScore(i, j) * scale;
            rawRow.push(s);
            maskedScores.push((i + queryOffset) >= j ? s : -Infinity);
        }
        rawScoreGrid.push(rawRow);
        const exps = maskedScores.map(s => s === -Infinity ? 0 : Math.exp(s));
        const sumExp = exps.reduce((a, b) => a + b, 0);
        attnWeights.push(exps.map(e => e / sumExp));
    }

    if (schematic) {
        // --- Part 0: QK^T scores schematic ---
        g.append('text').attr('class', 'tensor-label')
            .attr('x', gridW / 2).attr('y', -14)
            .text(`QK\u1d40 / \u221a${d_h} (${S_q}\u00d7${S})`);
        g.append('rect').attr('x', 0).attr('y', 10)
            .attr('width', gridW).attr('height', gridH)
            .attr('fill', '#3498db').attr('fill-opacity', 0.5).attr('rx', 2);
        drawPillLabel(g, gridW / 2, 10 + gridH / 2, 'q\u1d62\u00b7k\u2c7c / \u221ad\u2095', '#fff');

        const yOff = gridH + 46;

        // --- Part 1: Causal mask schematic ---
        g.append('text').attr('class', 'tensor-label')
            .attr('x', gridW / 2).attr('y', yOff - 14)
            .text(`Causal Mask (${S_q}\u00d7${S})`);
        const maskG = g.append('g').attr('transform', `translate(0, ${yOff})`);
        drawSchematicCausal(maskG, gridW, gridH, S_q, S, queryOffset, 'mask');

        const legendY = yOff + gridH + 26;
        g.append('rect').attr('x', 0).attr('y', legendY).attr('width', 12).attr('height', 12)
            .attr('fill', '#1abc9c').attr('fill-opacity', 0.85).attr('rx', 2);
        g.append('text').attr('class', 'dim-label').attr('x', 18).attr('y', legendY + 10)
            .attr('fill', '#aaa').text('Attend (i \u2265 j)');
        g.append('rect').attr('x', 140).attr('y', legendY).attr('width', 12).attr('height', 12)
            .attr('fill', '#2c3e50').attr('fill-opacity', 0.5).attr('rx', 2);
        g.append('text').attr('class', 'dim-label').attr('x', 158).attr('y', legendY + 10)
            .attr('fill', '#aaa').text('Masked (i < j) \u2192 -\u221e');

        // --- Part 2: Heatmap schematic ---
        let y2 = legendY + 46;
        g.append('text').attr('class', 'tensor-label')
            .attr('x', gridW / 2).attr('y', y2)
            .text('Attention Weights (after softmax)');
        y2 += 16;
        const hg = g.append('g').attr('transform', `translate(0, ${y2})`);
        drawSchematicCausal(hg, gridW, gridH, S_q, S, queryOffset, 'heat');
        g.append('text').attr('class', 'dim-label')
            .attr('x', gridW / 2).attr('y', y2 + gridH + 24)
            .attr('text-anchor', 'middle').attr('fill', '#f39c12')
            .attr('font-size', '10px').text('each row sums to 1');

        g.append('text').attr('class', 'dim-label')
            .attr('x', gridW / 2).attr('y', y2 + gridH + 42)
            .attr('text-anchor', 'middle').attr('fill', '#555')
            .attr('font-size', '9px').text('Schematic \u2014 reduce S to see individual cells');

        svg.attr('height', y2 + gridH + 64);
        return;
    }

    // --- Part 0: QK^T raw scores heatmap ---
    g.append('text').attr('class', 'tensor-label')
        .attr('x', gridW / 2).attr('y', -14)
        .text(`QK\u1d40 / \u221a${d_h} (${S_q}\u00d7${S})`);

    const maxRaw = Math.max(...rawScoreGrid.flat());
    const minRaw = Math.min(...rawScoreGrid.flat());
    const showScoreText = cellSize >= 20;

    const absMax = Math.max(Math.abs(maxRaw), Math.abs(minRaw)) || 1;

    for (let i = 0; i < S_q; i++) {
        for (let j = 0; j < S; j++) {
            const s = rawScoreGrid[i][j];
            const norm = s / absMax;  // -1 to 1
            const color = s >= 0 ? '#3498db' : '#e74c3c';
            const opacity = 0.15 + Math.abs(norm) * 0.7;
            g.append('rect')
                .attr('x', j * cellSize).attr('y', i * cellSize + 10)
                .attr('width', cellSize - (cellSize > 3 ? 1 : 0))
                .attr('height', cellSize - (cellSize > 3 ? 1 : 0))
                .attr('rx', cellSize >= 6 ? 2 : 0)
                .attr('fill', color)
                .attr('fill-opacity', opacity)
                .attr('stroke', cellSize >= 4 ? '#1a1d2a' : 'none').attr('stroke-width', 0.5);

            if (showScoreText) {
                g.append('text')
                    .attr('x', j * cellSize + cellSize / 2)
                    .attr('y', i * cellSize + cellSize / 2 + 13)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '7px')
                    .attr('fill', Math.abs(norm) > 0.5 ? '#fff' : '#ccc')
                    .text(s.toFixed(2));
            }
        }
    }
    drawAxisLabels(g, S_q, S, cellSize, labelEvery, 10, 0);

    const yOff = gridH + 46;

    // --- Part 1: Causal mask ---
    g.append('text').attr('class', 'tensor-label')
        .attr('x', gridW / 2).attr('y', yOff - 14)
        .text(`Causal Mask (${S_q}\u00d7${S})`);

    for (let i = 0; i < S_q; i++) {
        for (let j = 0; j < S; j++) {
            const allowed = (i + queryOffset) >= j;
            g.append('rect')
                .attr('x', j * cellSize).attr('y', yOff + i * cellSize + 10)
                .attr('width', cellSize - (cellSize > 3 ? 1 : 0)).attr('height', cellSize - (cellSize > 3 ? 1 : 0))
                .attr('rx', cellSize >= 6 ? 2 : 0)
                .attr('fill', allowed ? '#1abc9c' : '#2c3e50')
                .attr('fill-opacity', allowed ? 0.85 : 0.5)
                .attr('stroke', cellSize >= 4 ? '#1a1d2a' : 'none').attr('stroke-width', 0.5);

            if (showCellText) {
                g.append('text')
                    .attr('x', j * cellSize + cellSize / 2)
                    .attr('y', yOff + i * cellSize + cellSize / 2 + 13)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '8px')
                    .attr('fill', allowed ? '#fff' : '#555')
                    .text(allowed ? '1' : '-\u221e');
            }
        }
    }

    drawAxisLabels(g, S_q, S, cellSize, labelEvery, yOff + 10, 0);

    if (!showCellText) {
        const topAttendX = S_q === S ? 0 : ((queryOffset + 1) / S) * gridW;
        const attendCx = (topAttendX + gridW) / 4;
        drawPillLabel(g, attendCx, yOff + 10 + gridH * 2 / 3, '1', '#fff');
        if (gridW - topAttendX > 30) {
            const maskedCx = (topAttendX + 2 * gridW) / 3;
            drawPillLabel(g, maskedCx, yOff + 10 + gridH / 3, '\u2212\u221e', '#aaa');
        }
    }

    const maskLegendY = yOff + gridH + 26;
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

    const maxWeight = Math.max(...attnWeights.flat());

    for (let i = 0; i < S_q; i++) {
        for (let j = 0; j < S; j++) {
            const w = attnWeights[i][j];
            const intensity = maxWeight > 0 ? w / maxWeight : 0;
            g.append('rect')
                .attr('x', j * cellSize).attr('y', y2 + i * cellSize)
                .attr('width', cellSize - (cellSize > 3 ? 1 : 0)).attr('height', cellSize - (cellSize > 3 ? 1 : 0))
                .attr('rx', cellSize >= 6 ? 2 : 0)
                .attr('fill', w > 0 ? '#f39c12' : '#1a1d2a')
                .attr('fill-opacity', w > 0 ? 0.15 + intensity * 0.75 : 0.3)
                .attr('stroke', cellSize >= 4 ? '#1a1d2a' : 'none').attr('stroke-width', 0.5);

            if (showCellText && w > 0.01) {
                g.append('text')
                    .attr('x', j * cellSize + cellSize / 2)
                    .attr('y', y2 + i * cellSize + cellSize / 2 + 3)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '7px')
                    .attr('fill', intensity > 0.5 ? '#fff' : '#ccc')
                    .text(w.toFixed(2));
            }
        }
    }

    drawAxisLabels(g, S_q, S, cellSize, labelEvery, y2, 0);

    // "each row sums to 1" annotation
    g.append('text').attr('class', 'dim-label')
        .attr('x', gridW / 2).attr('y', y2 + gridH + 14)
        .attr('text-anchor', 'middle').attr('fill', '#f39c12')
        .attr('font-size', '10px')
        .text('each row sums to 1');

    const heatBottom = y2 + gridH + 30;

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

    svg.attr('height', gradY + 40);
}

// --- Mask tensor detail (when clicking the mask tensor directly) ---

export function drawMaskTensorDetail(svg, _tensor, params) {
    const { w: svgW } = detailMetrics();
    const S = params.S;
    const S_q = params.S_q || S;
    const { cellSize, labelEvery, schematic } = maskLayout(svgW, S_q, S);
    const gridW = S * cellSize;
    const gridH = S_q * cellSize;
    const pad = Math.max(30, (svgW - gridW - 60) / 2);

    const g = svg.append('g').attr('transform', `translate(${pad}, 38)`);

    g.append('text').attr('class', 'tensor-label')
        .attr('x', gridW / 2).attr('y', -14)
        .text(`Causal Mask (${S_q}\u00d7${S})`);

    if (schematic) {
        const queryOffset = S - S_q;
        drawSchematicCausal(g, gridW, gridH, S_q, S, queryOffset, 'mask');

        const descY = gridH + 26;
        g.append('rect').attr('x', 0).attr('y', descY).attr('width', 12).attr('height', 12)
            .attr('fill', '#1abc9c').attr('fill-opacity', 0.85).attr('rx', 2);
        g.append('text').attr('class', 'dim-label').attr('x', 18).attr('y', descY + 10)
            .attr('fill', '#aaa').text('Attend (i \u2265 j)');
        g.append('rect').attr('x', 140).attr('y', descY).attr('width', 12).attr('height', 12)
            .attr('fill', '#2c3e50').attr('fill-opacity', 0.5).attr('rx', 2);
        g.append('text').attr('class', 'dim-label').attr('x', 158).attr('y', descY + 10)
            .attr('fill', '#aaa').text('Masked (i < j) \u2192 -\u221e');

        g.append('text').attr('class', 'dim-label')
            .attr('x', gridW / 2).attr('y', descY + 28)
            .attr('text-anchor', 'middle').attr('fill', '#555')
            .attr('font-size', '9px').text('Schematic \u2014 reduce S to see individual cells');

        svg.attr('height', descY + 46);
        return;
    }

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

    drawAxisLabels(g, S_q, S, cellSize, labelEvery, 10, 0);

    if (!showCellText) {
        const topAttendX = S_q === S ? 0 : ((queryOffset + 1) / S) * gridW;
        const attendCx = (topAttendX + gridW) / 4;
        drawPillLabel(g, attendCx, 10 + gridH * 2 / 3, '1', '#fff');
        if (gridW - topAttendX > 30) {
            const maskedCx = (topAttendX + 2 * gridW) / 3;
            drawPillLabel(g, maskedCx, 10 + gridH / 3, '\u2212\u221e', '#aaa');
        }
    }

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

// --- Multi-request block-diagonal mask detail ---

export function drawPagedMaskTensorDetail(svg, _tensor, params) {
    const { w: svgW } = detailMetrics();
    const sLens = params.seqLens.slice(0, params.B);               // S per request (columns)
    const sqLens = params.queryLens.slice(0, params.B).map(q => q || 1); // S_q per request (rows)

    const dispCols = sLens.reduce((a, b) => a + b, 0);
    const dispRows = sqLens.reduce((a, b) => a + b, 0);
    const { cellSize, schematic } = maskLayout(svgW, dispRows, dispCols);
    const gridW = dispCols * cellSize;
    const gridH = dispRows * cellSize;
    const pad = Math.max(56, (svgW - gridW - 60) / 2);

    const g = svg.append('g').attr('transform', `translate(${pad}, 38)`);

    g.append('text').attr('class', 'tensor-label')
        .attr('x', gridW / 2).attr('y', -14)
        .text(`Block-Diagonal Causal Mask (B=${params.B})`);

    if (schematic) {
        // Background (cross-sequence blocked)
        g.append('rect').attr('x', 0).attr('y', 10)
            .attr('width', gridW).attr('height', gridH)
            .attr('fill', '#1a1520').attr('fill-opacity', 0.8).attr('rx', 2);

        // Draw each sequence's causal block on the diagonal
        let rowOff = 0, colOff = 0;
        for (let si = 0; si < sqLens.length; si++) {
            const nq = sqLens[si], ns = sLens[si];
            const bx = (colOff / dispCols) * gridW;
            const by = 10 + (rowOff / dispRows) * gridH;
            const bw = (ns / dispCols) * gridW;
            const bh = (nq / dispRows) * gridH;
            const qOff = ns - nq;
            const topX = bx + ((qOff + 1) / ns) * bw;

            g.append('rect').attr('x', bx).attr('y', by)
                .attr('width', bw).attr('height', bh)
                .attr('fill', '#2c3e50').attr('fill-opacity', 0.5);
            g.append('polygon')
                .attr('points', `${bx},${by} ${topX},${by} ${bx + bw},${by + bh} ${bx},${by + bh}`)
                .attr('fill', '#1abc9c').attr('fill-opacity', 0.7);
            g.append('text').attr('class', 'dim-label')
                .attr('x', bx + bw / 2).attr('y', by + bh / 2 + 3)
                .attr('text-anchor', 'middle').attr('font-size', '8px')
                .attr('fill', '#fff').attr('fill-opacity', 0.8)
                .text(`S${si}`);

            if (si < sqLens.length - 1) {
                g.append('line')
                    .attr('x1', 0).attr('y1', by + bh)
                    .attr('x2', gridW).attr('y2', by + bh)
                    .attr('stroke', '#e74c3c').attr('stroke-width', 1).attr('stroke-opacity', 0.6);
            }
            rowOff += nq; colOff += ns;
        }

        // Vertical boundary lines
        let vOff = 0;
        for (let sj = 0; sj < sLens.length - 1; sj++) {
            vOff += sLens[sj];
            const vx = (vOff / dispCols) * gridW;
            g.append('line')
                .attr('x1', vx).attr('y1', 10)
                .attr('x2', vx).attr('y2', gridH + 10)
                .attr('stroke', '#e74c3c').attr('stroke-width', 1).attr('stroke-opacity', 0.6);
        }

        const descY = gridH + 30;
        g.append('rect').attr('x', 0).attr('y', descY).attr('width', 12).attr('height', 12)
            .attr('fill', '#1abc9c').attr('fill-opacity', 0.85).attr('rx', 2);
        g.append('text').attr('class', 'dim-label').attr('x', 18).attr('y', descY + 10)
            .attr('fill', '#aaa').text('Causal attend');
        g.append('rect').attr('x', 0).attr('y', descY + 18).attr('width', 12).attr('height', 12)
            .attr('fill', '#1a1520').attr('fill-opacity', 0.8).attr('rx', 2);
        g.append('text').attr('class', 'dim-label').attr('x', 18).attr('y', descY + 28)
            .attr('fill', '#aaa').text('Cross-sequence (blocked)');

        g.append('text').attr('class', 'dim-label')
            .attr('x', gridW / 2).attr('y', descY + 48)
            .attr('text-anchor', 'middle').attr('fill', '#555')
            .attr('font-size', '9px').text('Schematic \u2014 reduce S to see individual cells');

        g.append('text').attr('class', 'dim-label')
            .attr('x', 0).attr('y', descY + 66)
            .attr('fill', '#777')
            .text(`Per-request: [${sqLens.map((q, i) => `S_q=${q} \u00d7 S=${sLens[i]}`).join(', ')}]`);

        svg.attr('height', descY + 86);
        return;
    }

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
                        fill = '#1a1520'; opacity = 0.8;
                    } else {
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
