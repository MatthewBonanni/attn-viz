// mask.js — Standard mask, mask tensor, and paged mask tensor detail visualizations
import { detailMetrics } from './shared.js';
import { drawSoftmaxSection } from './softmax.js';

export function drawMaskDetail(svg, _op, _tensorMap, params) {
    const { w: svgW } = detailMetrics();
    const S = params.S;
    const dispS = Math.min(S, 12);
    const cellSize = Math.min(28, Math.max(18, (svgW - 140) / dispS));
    const gridW = dispS * cellSize;
    const pad = Math.max(30, (svgW - gridW - 60) / 2);

    const g = svg.append('g').attr('transform', `translate(${pad}, 38)`);

    // --- Part 1: Causal mask ---
    g.append('text').attr('class', 'tensor-label')
        .attr('x', gridW / 2).attr('y', -14)
        .text(`Causal Mask (${S}\u00d7${S})`);

    for (let i = 0; i < dispS; i++) {
        for (let j = 0; j < dispS; j++) {
            const allowed = i >= j;
            g.append('rect')
                .attr('x', j * cellSize).attr('y', i * cellSize + 10)
                .attr('width', cellSize - 1).attr('height', cellSize - 1)
                .attr('rx', 2)
                .attr('fill', allowed ? '#1abc9c' : '#2c3e50')
                .attr('fill-opacity', allowed ? 0.85 : 0.5)
                .attr('stroke', '#1a1d2a').attr('stroke-width', 0.5);

            if (cellSize >= 16) {
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

    for (let i = 0; i < dispS; i++) {
        g.append('text').attr('class', 'dim-label')
            .attr('x', -6).attr('y', i * cellSize + cellSize / 2 + 13)
            .attr('text-anchor', 'end').attr('font-size', '8px').text(i);
        g.append('text').attr('class', 'dim-label')
            .attr('x', i * cellSize + cellSize / 2).attr('y', 6)
            .attr('text-anchor', 'middle').attr('font-size', '8px').text(i);
    }

    if (S > dispS) {
        g.append('text').attr('class', 'dim-label')
            .attr('x', dispS * cellSize / 2).attr('y', dispS * cellSize + 26)
            .attr('text-anchor', 'middle').attr('fill', '#666')
            .text(`(showing ${dispS}\u00d7${dispS} of ${S}\u00d7${S})`);
    }

    const maskLegendY = dispS * cellSize + 40;
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

    // Compute softmax attention weights for each row
    const attnWeights = [];
    for (let i = 0; i < dispS; i++) {
        // Generate varied raw scores for visual interest
        const rawScores = [];
        for (let j = 0; j < dispS; j++) {
            if (j <= i) {
                rawScores.push(1.0 + Math.sin(j * 1.7 + i * 0.3) * 0.7);
            } else {
                rawScores.push(-Infinity);
            }
        }
        const exps = rawScores.map(s => s === -Infinity ? 0 : Math.exp(s));
        const sumExp = exps.reduce((a, b) => a + b, 0);
        attnWeights.push(exps.map(e => e / sumExp));
    }

    // Find max weight for color scaling
    const maxWeight = Math.max(...attnWeights.flat());

    // Draw heatmap
    for (let i = 0; i < dispS; i++) {
        for (let j = 0; j < dispS; j++) {
            const w = attnWeights[i][j];
            const intensity = w / maxWeight;
            g.append('rect')
                .attr('x', j * cellSize).attr('y', y2 + i * cellSize)
                .attr('width', cellSize - 1).attr('height', cellSize - 1)
                .attr('rx', 2)
                .attr('fill', w > 0 ? '#f39c12' : '#1a1d2a')
                .attr('fill-opacity', w > 0 ? 0.15 + intensity * 0.75 : 0.3)
                .attr('stroke', '#1a1d2a').attr('stroke-width', 0.5);

            if (cellSize >= 20 && w > 0.01) {
                g.append('text')
                    .attr('x', j * cellSize + cellSize / 2)
                    .attr('y', y2 + i * cellSize + cellSize / 2 + 3)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '7px')
                    .attr('fill', intensity > 0.5 ? '#fff' : '#ccc')
                    .text(w.toFixed(2));
            }
        }

        // Row sum annotation
        g.append('text').attr('class', 'dim-label')
            .attr('x', dispS * cellSize + 6).attr('y', y2 + i * cellSize + cellSize / 2 + 3)
            .attr('font-size', '8px').attr('fill', '#f39c12')
            .text('= 1');
    }

    // Row/col indices
    for (let i = 0; i < dispS; i++) {
        g.append('text').attr('class', 'dim-label')
            .attr('x', -6).attr('y', y2 + i * cellSize + cellSize / 2 + 3)
            .attr('text-anchor', 'end').attr('font-size', '8px').text(i);
        g.append('text').attr('class', 'dim-label')
            .attr('x', i * cellSize + cellSize / 2).attr('y', y2 - 4)
            .attr('text-anchor', 'middle').attr('font-size', '8px').text(i);
    }

    // "each row sums to 1" label — placed below grid, centered
    g.append('text').attr('class', 'dim-label')
        .attr('x', dispS * cellSize / 2).attr('y', y2 + dispS * cellSize + 14)
        .attr('text-anchor', 'middle').attr('fill', '#f39c12')
        .attr('font-size', '10px')
        .text('each row sums to 1');

    // Color scale legend
    const heatLegendY = y2 + dispS * cellSize + 30;
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
    const softmaxBottom = drawSoftmaxSection(g, 0, gradY + 52, dispS, cellSize, attnWeights);

    svg.attr('height', softmaxBottom + 10);
}

// --- Mask tensor detail (when clicking the mask tensor directly) ---

export function drawMaskTensorDetail(svg, _tensor, params) {
    const { w: svgW } = detailMetrics();
    const S = params.S;
    const dispS = Math.min(S, 12);
    const cellSize = Math.min(28, Math.max(18, (svgW - 140) / dispS));
    const gridW = dispS * cellSize;
    const pad = Math.max(30, (svgW - gridW - 60) / 2);

    const g = svg.append('g').attr('transform', `translate(${pad}, 38)`);

    g.append('text').attr('class', 'tensor-label')
        .attr('x', gridW / 2).attr('y', -14)
        .text(`Causal Mask (${S}\u00d7${S})`);

    for (let i = 0; i < dispS; i++) {
        for (let j = 0; j < dispS; j++) {
            const allowed = i >= j;
            g.append('rect')
                .attr('x', j * cellSize).attr('y', i * cellSize + 10)
                .attr('width', cellSize - 1).attr('height', cellSize - 1)
                .attr('rx', 2)
                .attr('fill', allowed ? '#1abc9c' : '#2c3e50')
                .attr('fill-opacity', allowed ? 0.85 : 0.5)
                .attr('stroke', '#1a1d2a').attr('stroke-width', 0.5);

            if (cellSize >= 16) {
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

    for (let i = 0; i < dispS; i++) {
        g.append('text').attr('class', 'dim-label')
            .attr('x', -6).attr('y', i * cellSize + cellSize / 2 + 13)
            .attr('text-anchor', 'end').attr('font-size', '8px').text(i);
        g.append('text').attr('class', 'dim-label')
            .attr('x', i * cellSize + cellSize / 2).attr('y', 6)
            .attr('text-anchor', 'middle').attr('font-size', '8px').text(i);
    }

    if (S > dispS) {
        g.append('text').attr('class', 'dim-label')
            .attr('x', dispS * cellSize / 2).attr('y', dispS * cellSize + 26)
            .attr('text-anchor', 'middle').attr('fill', '#666')
            .text(`(showing ${dispS}\u00d7${dispS} of ${S}\u00d7${S})`);
    }

    const descY = dispS * cellSize + 40;
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
    const ctxLens = params.seqLens.slice(0, params.B);
    const queryLens = params.queryLens.slice(0, params.B);
    const seqLens = ctxLens.map((c, i) => c + (queryLens[i] || 1));
    const totalS = seqLens.reduce((a, b) => a + b, 0);
    // Cap tokens per sequence so all sequences are represented
    const maxTokens = 24;
    const cappedLens = totalS > maxTokens
        ? seqLens.map(s => Math.max(2, Math.round(s * maxTokens / totalS)))
        : seqLens;
    const dispS = cappedLens.reduce((a, b) => a + b, 0);
    const cellSize = Math.min(22, Math.max(10, (svgW - 100) / dispS));
    const gridW = dispS * cellSize;
    const pad = Math.max(56, (svgW - gridW - 60) / 2);

    const g = svg.append('g').attr('transform', `translate(${pad}, 38)`);

    g.append('text').attr('class', 'tensor-label')
        .attr('x', gridW / 2).attr('y', -14)
        .text('Variable-Length Causal Mask');

    // Draw block-diagonal mask grid using cappedLens
    let rowOff = 0;
    for (let si = 0; si < cappedLens.length; si++) {
        const sLen = cappedLens[si];
        let colOff = 0;

        for (let sj = 0; sj < cappedLens.length; sj++) {
            const sLenJ = cappedLens[sj];

            for (let i = 0; i < sLen; i++) {
                for (let j = 0; j < sLenJ; j++) {
                    const gi = rowOff + i;
                    const gj = colOff + j;

                    let fill, opacity;
                    if (si !== sj) {
                        fill = '#1a1520'; opacity = 0.8;
                    } else if (i >= j) {
                        fill = '#1abc9c'; opacity = 0.85;
                    } else {
                        fill = '#2c3e50'; opacity = 0.5;
                    }

                    g.append('rect')
                        .attr('x', gj * cellSize).attr('y', gi * cellSize + 10)
                        .attr('width', cellSize - 1).attr('height', cellSize - 1)
                        .attr('rx', 1)
                        .attr('fill', fill).attr('fill-opacity', opacity)
                        .attr('stroke', '#1a1d2a').attr('stroke-width', 0.3);
                }
            }
            colOff += cappedLens[sj];
        }

        // Sequence boundary lines
        if (si < cappedLens.length - 1) {
            g.append('line')
                .attr('x1', 0).attr('y1', (rowOff + sLen) * cellSize + 10)
                .attr('x2', dispS * cellSize).attr('y2', (rowOff + sLen) * cellSize + 10)
                .attr('stroke', '#e74c3c').attr('stroke-width', 1).attr('stroke-opacity', 0.6);
            g.append('line')
                .attr('x1', (rowOff + sLen) * cellSize).attr('y1', 10)
                .attr('x2', (rowOff + sLen) * cellSize).attr('y2', dispS * cellSize + 10)
                .attr('stroke', '#e74c3c').attr('stroke-width', 1).attr('stroke-opacity', 0.6);
        }

        rowOff += sLen;
    }

    // Sequence labels
    let labelOff = 0;
    for (let si = 0; si < cappedLens.length; si++) {
        const sLen = cappedLens[si];
        const midPos = labelOff + sLen / 2;
        g.append('text').attr('class', 'dim-label')
            .attr('x', -8).attr('y', midPos * cellSize + 14)
            .attr('text-anchor', 'end').attr('font-size', '8px').attr('fill', '#aaa')
            .text(`S${si}(${seqLens[si]})`);
        labelOff += sLen;
    }

    const descY = dispS * cellSize + 40;

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
        .text(`Per-request total: [${seqLens.join(', ')}] (cached+new), total: ${totalS}`);

    svg.attr('height', descY + 86);
}
