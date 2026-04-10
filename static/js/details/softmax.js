// softmax.js — Softmax bar chart section

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

    // Decide label density based on bar width
    const showBarText = barW >= 14;

    for (let j = 0; j < dispS; j++) {
        const allowed = probs[j] > 0;
        const barH = allowed ? (probs[j] / maxProb) * barMaxH : 0;

        g.append('rect')
            .attr('x', x + j * barW + (barW > 3 ? 1 : 0))
            .attr('y', barBaseY + barMaxH - barH)
            .attr('width', Math.max(barW - (barW > 3 ? 2 : 0), 0.5))
            .attr('height', Math.max(barH, barW >= 2 ? 1 : 0.5))
            .attr('fill', allowed ? '#f39c12' : '#2c3e50')
            .attr('fill-opacity', allowed ? 0.85 : 0.4)
            .attr('rx', barW >= 4 ? 1 : 0);

        if (showBarText) {
            g.append('text')
                .attr('x', x + j * barW + barW / 2)
                .attr('y', barBaseY + barMaxH + 12)
                .attr('text-anchor', 'middle')
                .attr('font-size', '8px')
                .attr('fill', allowed ? '#ddd' : '#555')
                .text(allowed ? probs[j].toFixed(2) : '0');
        }
    }

    // Axis ticks for large S
    if (!showBarText && dispS > 20) {
        const ticks = [0, exampleRow, dispS - 1];
        for (const t of ticks) {
            g.append('text')
                .attr('x', x + t * barW + barW / 2)
                .attr('y', barBaseY + barMaxH + 10)
                .attr('text-anchor', 'middle')
                .attr('font-size', '7px')
                .attr('fill', '#666')
                .text(t);
        }
        // Causal boundary marker
        if (exampleRow < dispS - 1) {
            const bx = x + (exampleRow + 0.5) * barW;
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
