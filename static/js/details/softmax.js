// softmax.js — Softmax bar chart section

export function drawSoftmaxSection(g, x, y, dispS, cellSize, precomputedWeights) {
    const exampleRow = Math.min(4, dispS - 1);

    g.append('text').attr('class', 'tensor-label')
        .attr('x', x + dispS * cellSize / 2).attr('y', y)
        .text(`Softmax (row ${exampleRow})`);

    // Use pre-computed weights if available, otherwise generate
    let probs;
    if (precomputedWeights && precomputedWeights[exampleRow]) {
        probs = precomputedWeights[exampleRow];
    } else {
        const rawScores = [];
        for (let j = 0; j < dispS; j++) {
            if (j <= exampleRow) {
                rawScores.push(1.0 + Math.sin(j * 1.7 + exampleRow * 0.3) * 0.7);
            } else {
                rawScores.push(-Infinity);
            }
        }
        const exps = rawScores.map(s => s === -Infinity ? 0 : Math.exp(s));
        const sumExp = exps.reduce((a, b) => a + b, 0);
        probs = exps.map(e => e / sumExp);
    }
    const maxProb = Math.max(...probs);

    const barW = Math.min(cellSize, 28);
    const barMaxH = 50;
    const barBaseY = y + 16;

    // Row label
    g.append('text').attr('class', 'dim-label')
        .attr('x', x - 6).attr('y', barBaseY + barMaxH / 2 + 3)
        .attr('text-anchor', 'end').attr('fill', '#aaa')
        .attr('font-size', '9px')
        .text(`row ${exampleRow}`);

    for (let j = 0; j < dispS; j++) {
        const allowed = j <= exampleRow;
        const barH = allowed ? (probs[j] / maxProb) * barMaxH : 0;

        g.append('rect')
            .attr('x', x + j * barW + 1)
            .attr('y', barBaseY + barMaxH - barH)
            .attr('width', barW - 2)
            .attr('height', Math.max(barH, 1))
            .attr('fill', allowed ? '#f39c12' : '#2c3e50')
            .attr('fill-opacity', allowed ? 0.85 : 0.4)
            .attr('rx', 1);

        if (barW >= 14) {
            g.append('text')
                .attr('x', x + j * barW + barW / 2)
                .attr('y', barBaseY + barMaxH + 12)
                .attr('text-anchor', 'middle')
                .attr('font-size', '8px')
                .attr('fill', allowed ? '#ddd' : '#555')
                .text(allowed ? probs[j].toFixed(2) : '0');
        }
    }

    // "sum = 1" annotation below bars
    g.append('text').attr('class', 'dim-label')
        .attr('x', x + dispS * barW / 2).attr('y', barBaseY + barMaxH + 30)
        .attr('text-anchor', 'middle').attr('fill', '#f39c12')
        .attr('font-size', '10px')
        .text('each row sums to 1');

    return barBaseY + barMaxH + 46;
}
