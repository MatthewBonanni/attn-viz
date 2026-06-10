// softmax.js — Softmax bar chart section + standalone softmax op detail
import { detailMetrics, maskLayout } from './shared.js';

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

// Standalone softmax op detail (DSA's sparse attention): scores [S_q, k] →
// softmax → weights [S_q, k], shown as matrices like the dense mask detail.
// The crucial difference from dense attention: there is NO mask stage — the k
// columns are each query's selected tokens, so every cell is live.
export function drawSoftmaxOpDetail(svg, op, tensorMap, params) {
    const out = tensorMap[op.output];
    const k = out ? out.shape[out.shape.length - 1] : 32;
    const S_q = params.queryLens?.[0] ?? params.S_q ?? 16;
    const { w: svgW } = detailMetrics();
    const { cellSize, schematic } = maskLayout(svgW, S_q, k);

    const gridW = schematic ? 340 : k * cellSize;
    const gridH = schematic ? 120 : S_q * cellSize;
    const pad = Math.max(30, (svgW - gridW - 60) / 2);
    const g = svg.append('g').attr('transform', `translate(${pad}, 38)`);
    // Long captions center on the panel, not the (possibly narrow) grid
    const capX = svgW / 2 - pad;

    // --- Part 0: scores over selected tokens ---
    g.append('text').attr('class', 'tensor-label')
        .attr('x', gridW / 2).attr('y', -14)
        .text(`Scores over selected tokens (${S_q}×${k})`);
    g.append('text').attr('class', 'dim-label')
        .attr('x', capX).attr('y', 0)
        .attr('text-anchor', 'middle').attr('font-size', '8px').attr('fill', '#777')
        .text('column j = the j-th token query i selected — positions differ per row (topk_idx)');

    let attnWeights = null;
    if (schematic) {
        g.append('rect').attr('x', 0).attr('y', 10)
            .attr('width', gridW).attr('height', gridH)
            .attr('fill', '#3498db').attr('fill-opacity', 0.5).attr('rx', 2);
        drawPillLabel(g, gridW / 2, 10 + gridH / 2, "q'·c_kvⱼ + q_r'·k_rⱼ", '#fff');
    } else {
        // Exact grids: pseudo-scores, softmax per row — all cells live
        const showCellText = cellSize >= 16;
        const rawScoreGrid = [];
        attnWeights = [];
        for (let i = 0; i < S_q; i++) {
            const row = [];
            for (let j = 0; j < k; j++) row.push(pseudoScore(i, j));
            rawScoreGrid.push(row);
            const exps = row.map(Math.exp);
            const sumExp = exps.reduce((a, b) => a + b, 0);
            attnWeights.push(exps.map(e => e / sumExp));
        }
        const absMax = Math.max(...rawScoreGrid.flat().map(Math.abs)) || 1;
        for (let i = 0; i < S_q; i++) {
            for (let j = 0; j < k; j++) {
                const s = rawScoreGrid[i][j];
                g.append('rect')
                    .attr('x', j * cellSize).attr('y', i * cellSize + 10)
                    .attr('width', cellSize - (cellSize > 3 ? 1 : 0))
                    .attr('height', cellSize - (cellSize > 3 ? 1 : 0))
                    .attr('rx', cellSize >= 6 ? 2 : 0)
                    .attr('fill', s >= 0 ? '#3498db' : '#e74c3c')
                    .attr('fill-opacity', 0.15 + Math.abs(s) / absMax * 0.7)
                    .attr('stroke', cellSize >= 4 ? '#1a1d2a' : 'none').attr('stroke-width', 0.5);
                if (showCellText) {
                    g.append('text')
                        .attr('x', j * cellSize + cellSize / 2)
                        .attr('y', i * cellSize + cellSize / 2 + 13)
                        .attr('text-anchor', 'middle').attr('font-size', '7px')
                        .attr('fill', Math.abs(s) / absMax > 0.5 ? '#fff' : '#ccc')
                        .text(s.toFixed(2));
                }
            }
        }
    }

    // --- Divider: softmax, explicitly no mask stage ---
    const divY = gridH + 38;
    g.append('circle')
        .attr('cx', gridW / 2).attr('cy', divY).attr('r', 11)
        .attr('fill', '#1e2030').attr('stroke', '#f39c12').attr('stroke-width', 2);
    g.append('text')
        .attr('x', gridW / 2 + 1).attr('y', divY + 5)
        .attr('text-anchor', 'middle').attr('font-size', '14px').attr('fill', '#f39c12')
        .text('σ');
    g.append('text').attr('class', 'dim-label')
        .attr('x', capX).attr('y', divY + 24)
        .attr('text-anchor', 'middle').attr('font-size', '9px').attr('fill', '#aaa')
        .text('row-wise softmax — no mask: every column is live (top-k was already causal)');

    // --- Part 1: attention weights ---
    const y2 = divY + 44;
    g.append('text').attr('class', 'tensor-label')
        .attr('x', gridW / 2).attr('y', y2 - 6)
        .text(`Attention Weights (${S_q}×${k})`);
    if (schematic) {
        g.append('rect').attr('x', 0).attr('y', y2 + 4)
            .attr('width', gridW).attr('height', gridH)
            .attr('fill', '#f39c12').attr('fill-opacity', 0.5).attr('rx', 2);
        drawPillLabel(g, gridW / 2, y2 + 4 + gridH / 2, 'all cells > 0', '#fff');
    } else {
        const showCellText = cellSize >= 16;
        const maxWeight = Math.max(...attnWeights.flat());
        for (let i = 0; i < S_q; i++) {
            for (let j = 0; j < k; j++) {
                const w = attnWeights[i][j];
                const intensity = maxWeight > 0 ? w / maxWeight : 0;
                g.append('rect')
                    .attr('x', j * cellSize).attr('y', y2 + 4 + i * cellSize)
                    .attr('width', cellSize - (cellSize > 3 ? 1 : 0))
                    .attr('height', cellSize - (cellSize > 3 ? 1 : 0))
                    .attr('rx', cellSize >= 6 ? 2 : 0)
                    .attr('fill', '#f39c12')
                    .attr('fill-opacity', 0.15 + intensity * 0.75)
                    .attr('stroke', cellSize >= 4 ? '#1a1d2a' : 'none').attr('stroke-width', 0.5);
                if (showCellText && w > 0.01) {
                    g.append('text')
                        .attr('x', j * cellSize + cellSize / 2)
                        .attr('y', y2 + 4 + i * cellSize + cellSize / 2 + 3)
                        .attr('text-anchor', 'middle').attr('font-size', '7px')
                        .attr('fill', intensity > 0.5 ? '#fff' : '#ccc')
                        .text(w.toFixed(2));
                }
            }
        }
    }
    g.append('text').attr('class', 'dim-label')
        .attr('x', gridW / 2).attr('y', y2 + 4 + gridH + 14)
        .attr('text-anchor', 'middle').attr('fill', '#f39c12')
        .attr('font-size', '10px')
        .text('each row sums to 1');

    let y3 = y2 + 4 + gridH + 32;
    if (schematic) {
        g.append('text').attr('class', 'dim-label')
            .attr('x', gridW / 2).attr('y', y3 - 6)
            .attr('text-anchor', 'middle').attr('fill', '#555').attr('font-size', '9px')
            .text('Schematic — reduce S_q and k to see individual cells');
        y3 += 8;
    }

    // --- Part 2: softmax bar chart for an example row ---
    let weights = attnWeights;
    let dispS = k;
    if (!weights) {
        // Schematic: generate example rows over a capped column count, all live
        dispS = Math.min(k, 64);
        weights = [];
        for (let i = 0; i <= 4; i++) {
            const raw = [];
            for (let j = 0; j < dispS; j++) raw.push(pseudoScore(i, j));
            const exps = raw.map(Math.exp);
            const sum = exps.reduce((a, b) => a + b, 0);
            weights.push(exps.map(e => e / sum));
        }
    }
    const endY = drawSoftmaxSection(g, 0, y3 + 12, dispS, 0, weights);

    svg.attr('height', 38 + endY + 24);
}

function pseudoScore(i, j) {
    let h = (i * 374761 + j * 668265) ^ 0x5bd1e995;
    h = Math.imul(h ^ (h >>> 13), 0x5bd1e995);
    h = h ^ (h >>> 15);
    return -0.8 + ((h >>> 0) / 0xffffffff) * 2.4;  // range [-0.8, 1.6]
}

export function drawSoftmaxSection(g, x, y, dispS, _cellSize, precomputedWeights) {
    // Pick an example row that exists in precomputed weights
    const maxRow = precomputedWeights ? precomputedWeights.length - 1 : dispS - 1;
    const exampleRow = Math.min(4, maxRow);

    // Use pre-computed weights if available, otherwise generate
    let probs;
    if (precomputedWeights && precomputedWeights[exampleRow]) {
        probs = precomputedWeights[exampleRow];
    } else {
        const rawScores = [];
        for (let j = 0; j < dispS; j++) {
            if (j <= exampleRow) {
                rawScores.push(pseudoScore(exampleRow, j));
            } else {
                rawScores.push(-Infinity);
            }
        }
        const exps = rawScores.map(s => s === -Infinity ? 0 : Math.exp(s));
        const sumExp = exps.reduce((a, b) => a + b, 0);
        probs = exps.map(e => e / sumExp);
    }
    const maxProb = Math.max(...probs);

    // Compute bar width from available chart width, not grid cellSize.
    // This ensures the chart is always visible regardless of S.
    const chartW = 420;
    const barW = Math.max(1, Math.min(28, chartW / dispS));
    const actualChartW = barW * dispS;
    const barMaxH = 50;
    const barBaseY = y + 16;

    g.append('text').attr('class', 'tensor-label')
        .attr('x', x + actualChartW / 2).attr('y', y)
        .text(`Softmax (row ${exampleRow})`);

    // Row label
    g.append('text').attr('class', 'dim-label')
        .attr('x', x - 6).attr('y', barBaseY + barMaxH / 2 + 3)
        .attr('text-anchor', 'end').attr('fill', '#aaa')
        .attr('font-size', '9px')
        .text(`row ${exampleRow}`);

    // When S is large, bin probabilities into at most MAX_BARS buckets
    // to avoid creating thousands of sub-pixel SVG elements.
    const MAX_BARS = 200;
    const binned = dispS > MAX_BARS;
    const numBars = binned ? MAX_BARS : dispS;
    const renderBarW = binned ? chartW / numBars : barW;
    const renderChartW = binned ? chartW : actualChartW;

    // Decide label density based on bar width
    const showBarText = renderBarW >= 14;

    for (let b = 0; b < numBars; b++) {
        let prob, allowed;
        if (binned) {
            // Each bin takes the max probability in its range
            const jStart = Math.floor(b * dispS / numBars);
            const jEnd = Math.floor((b + 1) * dispS / numBars);
            let maxP = 0;
            allowed = false;
            for (let j = jStart; j < jEnd; j++) {
                if (probs[j] > maxP) maxP = probs[j];
                if (probs[j] > 0) allowed = true;
            }
            prob = maxP;
        } else {
            prob = probs[b];
            allowed = prob > 0;
        }
        const barH = allowed ? (prob / maxProb) * barMaxH : 0;

        g.append('rect')
            .attr('x', x + b * renderBarW + (renderBarW > 3 ? 1 : 0))
            .attr('y', barBaseY + barMaxH - barH)
            .attr('width', Math.max(renderBarW - (renderBarW > 3 ? 2 : 0), 0.5))
            .attr('height', Math.max(barH, renderBarW >= 2 ? 1 : 0.5))
            .attr('fill', allowed ? '#f39c12' : '#2c3e50')
            .attr('fill-opacity', allowed ? 0.85 : 0.4)
            .attr('rx', renderBarW >= 4 ? 1 : 0);

        if (showBarText) {
            g.append('text')
                .attr('x', x + b * renderBarW + renderBarW / 2)
                .attr('y', barBaseY + barMaxH + 12)
                .attr('text-anchor', 'middle')
                .attr('font-size', '8px')
                .attr('fill', allowed ? '#ddd' : '#555')
                .text(allowed ? prob.toFixed(2) : '0');
        }
    }

    // Axis ticks for large S
    if (!showBarText && dispS > 20) {
        const ticks = [0, exampleRow, dispS - 1];
        for (const t of ticks) {
            const tickX = binned ? (t / dispS) * renderChartW : t * barW + barW / 2;
            g.append('text')
                .attr('x', x + tickX)
                .attr('y', barBaseY + barMaxH + 10)
                .attr('text-anchor', 'middle')
                .attr('font-size', '7px')
                .attr('fill', '#666')
                .text(t);
        }
        // Causal boundary marker — only when positions beyond the row are actually
        // masked (not for DSA's sparse softmax, where every selected column is live)
        const hasCutoff = probs.slice(exampleRow + 1).every(p => p === 0);
        if (exampleRow < dispS - 1 && hasCutoff) {
            const bx = binned
                ? x + ((exampleRow + 0.5) / dispS) * renderChartW
                : x + (exampleRow + 0.5) * barW;
            g.append('line')
                .attr('x1', bx).attr('y1', barBaseY)
                .attr('x2', bx).attr('y2', barBaseY + barMaxH)
                .attr('stroke', '#e74c3c').attr('stroke-width', 0.5)
                .attr('stroke-dasharray', '2,2').attr('stroke-opacity', 0.6);
            g.append('text')
                .attr('x', bx + 3).attr('y', barBaseY + 8)
                .attr('font-size', '7px').attr('fill', '#e74c3c').attr('fill-opacity', 0.7)
                .text('causal cutoff');
        }
    }

    // "sum = 1" annotation below bars
    const sumLabelY = barBaseY + barMaxH + (showBarText ? 30 : 20);
    g.append('text').attr('class', 'dim-label')
        .attr('x', x + actualChartW / 2).attr('y', sumLabelY)
        .attr('text-anchor', 'middle').attr('fill', '#f39c12')
        .attr('font-size', '10px')
        .text('each row sums to 1');

    return sumLabelY + 16;
}
