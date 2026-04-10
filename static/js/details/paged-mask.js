// paged-mask.js — Paged / variable-length mask detail visualization
import { detailMetrics } from './shared.js';
import { drawSoftmaxSection } from './softmax.js';

export function drawPagedMaskDetail(svg, _op, _tensorMap, params) {
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
    const pad = Math.max(56, (svgW - dispS * cellSize - 60) / 2);

    const g = svg.append('g').attr('transform', `translate(${pad}, 20)`);

    g.append('text').attr('class', 'tensor-label')
        .attr('x', dispS * cellSize / 2).attr('y', -6)
        .text(`Variable-Length Causal Mask`);

    // Draw the block-diagonal mask using cappedLens for display
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

    // Sequence lengths
    g.append('text').attr('class', 'dim-label')
        .attr('x', 0).attr('y', descY + 66)
        .attr('fill', '#777')
        .text(`Per-request total: [${seqLens.join(', ')}] (cached+new), total: ${totalS}`);

    // --- Part 2: Attention weights heatmap (after softmax) ---
    let y2 = descY + 92;

    g.append('text').attr('class', 'tensor-label')
        .attr('x', dispS * cellSize / 2).attr('y', y2)
        .text('Attention Weights (after softmax)');
    y2 += 26;

    // Compute softmax attention weights for each row (block-diagonal)
    const attnWeights = [];
    let rOff = 0;
    for (let si = 0; si < cappedLens.length; si++) {
        const sLen = cappedLens[si];
        for (let i = 0; i < sLen; i++) {
            const row = new Array(dispS).fill(0);
            // Within this sequence's block: causal softmax
            const cOff = rOff; // column offset for this sequence
            const rawScores = [];
            for (let j = 0; j < sLen; j++) {
                if (j <= i) {
                    rawScores.push(1.0 + Math.sin(j * 1.7 + i * 0.3) * 0.7);
                } else {
                    rawScores.push(-Infinity);
                }
            }
            const exps = rawScores.map(s => s === -Infinity ? 0 : Math.exp(s));
            const sumExp = exps.reduce((a, b) => a + b, 0);
            const probs = exps.map(e => e / sumExp);
            for (let j = 0; j < sLen; j++) {
                row[cOff + j] = probs[j];
            }
            attnWeights.push(row);
        }
        rOff += sLen;
    }

    const maxWeight = Math.max(...attnWeights.flat());

    // Draw heatmap
    for (let i = 0; i < dispS; i++) {
        for (let j = 0; j < dispS; j++) {
            const w = attnWeights[i][j];
            const intensity = maxWeight > 0 ? w / maxWeight : 0;
            g.append('rect')
                .attr('x', j * cellSize).attr('y', y2 + i * cellSize)
                .attr('width', cellSize - 1).attr('height', cellSize - 1)
                .attr('rx', 1)
                .attr('fill', w > 0 ? '#f39c12' : '#1a1d2a')
                .attr('fill-opacity', w > 0 ? 0.15 + intensity * 0.75 : 0.3)
                .attr('stroke', '#1a1d2a').attr('stroke-width', 0.3);

            if (cellSize >= 16 && w > 0.01) {
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

    // Sequence boundary lines on the heatmap
    let bOff = 0;
    for (let si = 0; si < cappedLens.length - 1; si++) {
        bOff += cappedLens[si];
        g.append('line')
            .attr('x1', 0).attr('y1', y2 + bOff * cellSize)
            .attr('x2', dispS * cellSize).attr('y2', y2 + bOff * cellSize)
            .attr('stroke', '#e74c3c').attr('stroke-width', 1).attr('stroke-opacity', 0.4);
        g.append('line')
            .attr('x1', bOff * cellSize).attr('y1', y2)
            .attr('x2', bOff * cellSize).attr('y2', y2 + dispS * cellSize)
            .attr('stroke', '#e74c3c').attr('stroke-width', 1).attr('stroke-opacity', 0.4);
    }

    // Row labels
    let lOff = 0;
    for (let si = 0; si < cappedLens.length; si++) {
        const sLen = cappedLens[si];
        const midPos = lOff + sLen / 2;
        g.append('text').attr('class', 'dim-label')
            .attr('x', -8).attr('y', y2 + midPos * cellSize + cellSize / 2)
            .attr('text-anchor', 'end').attr('font-size', '8px').attr('fill', '#aaa')
            .text(`S${si}`);
        lOff += sLen;
    }

    // "each row sums to 1" label
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

    // --- Part 3: Softmax bar chart ---
    const softmaxBottom = drawSoftmaxSection(g, 0, gradY + 52, dispS, cellSize, attnWeights);

    svg.attr('height', softmaxBottom + 10);
}
