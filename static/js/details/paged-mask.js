// paged-mask.js — Paged / variable-length mask detail visualization
import { detailMetrics, maskLayout } from './shared.js';

export function drawPagedMaskDetail(svg, _op, _tensorMap, params) {
    const { w: svgW } = detailMetrics();
    const sLens = params.seqLens.slice(0, params.B);               // S per request (columns)
    const sqLens = params.queryLens.slice(0, params.B).map(q => q || 1); // S_q per request (rows)
    const dispCols = sLens.reduce((a, b) => a + b, 0);
    const dispRows = sqLens.reduce((a, b) => a + b, 0);
    const { cellSize, schematic } = maskLayout(svgW, dispRows, dispCols);
    const gridW = dispCols * cellSize;
    const gridH = dispRows * cellSize;
    const pad = Math.max(56, (svgW - gridW - 60) / 2);

    const g = svg.append('g').attr('transform', `translate(${pad}, 20)`);

    g.append('text').attr('class', 'tensor-label')
        .attr('x', gridW / 2).attr('y', -6)
        .text(`Variable-Length Causal Mask`);

    if (schematic) {
        // Background (cross-sequence blocked)
        g.append('rect').attr('x', 0).attr('y', 10)
            .attr('width', gridW).attr('height', gridH)
            .attr('fill', '#1a1520').attr('fill-opacity', 0.8).attr('rx', 2);

        // Draw each sequence's causal block on the diagonal
        let sRowOff = 0, sColOff = 0;
        for (let si = 0; si < sqLens.length; si++) {
            const nq = sqLens[si], ns = sLens[si];
            const bx = (sColOff / dispCols) * gridW;
            const by = 10 + (sRowOff / dispRows) * gridH;
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
            sRowOff += nq; sColOff += ns;
        }

        // Vertical boundary lines
        let svOff = 0;
        for (let sj = 0; sj < sLens.length - 1; sj++) {
            svOff += sLens[sj];
            const vx = (svOff / dispCols) * gridW;
            g.append('line')
                .attr('x1', vx).attr('y1', 10)
                .attr('x2', vx).attr('y2', gridH + 10)
                .attr('stroke', '#e74c3c').attr('stroke-width', 1).attr('stroke-opacity', 0.6);
        }

        // Heatmap schematic
        let y2 = gridH + 46;
        g.append('text').attr('class', 'tensor-label')
            .attr('x', gridW / 2).attr('y', y2)
            .text('Attention Weights (after softmax)');
        y2 += 16;

        g.append('rect').attr('x', 0).attr('y', y2)
            .attr('width', gridW).attr('height', gridH)
            .attr('fill', '#1a1d2a').attr('fill-opacity', 0.5).attr('rx', 2);

        let hRowOff = 0, hColOff = 0;
        for (let si = 0; si < sqLens.length; si++) {
            const nq = sqLens[si], ns = sLens[si];
            const bx = (hColOff / dispCols) * gridW;
            const by = y2 + (hRowOff / dispRows) * gridH;
            const bw = (ns / dispCols) * gridW;
            const bh = (nq / dispRows) * gridH;
            const qOff = ns - nq;
            const topX = bx + ((qOff + 1) / ns) * bw;

            g.append('polygon')
                .attr('points', `${bx},${by} ${topX},${by} ${bx + bw},${by + bh} ${bx},${by + bh}`)
                .attr('fill', '#f39c12').attr('fill-opacity', 0.5);
            hRowOff += nq; hColOff += ns;
        }

        g.append('text').attr('class', 'dim-label')
            .attr('x', gridW / 2).attr('y', y2 + gridH + 14)
            .attr('text-anchor', 'middle').attr('fill', '#f39c12')
            .attr('font-size', '10px').text('each row sums to 1');

        // Legend
        const descY = y2 + gridH + 30;
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
            .text(`Per-request: [${sqLens.map((q, i) => `S_q=${q}, S=${sLens[i]}`).join('; ')}]`);

        svg.attr('height', descY + 86);
        return;
    }

    // Draw block-diagonal mask: rows = S_q, cols = S
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

        // Horizontal boundary line
        if (si < sqLens.length - 1) {
            g.append('line')
                .attr('x1', 0).attr('y1', (rowOff + nq) * cellSize + 10)
                .attr('x2', gridW).attr('y2', (rowOff + nq) * cellSize + 10)
                .attr('stroke', '#e74c3c').attr('stroke-width', 1).attr('stroke-opacity', 0.6);
        }

        rowOff += nq;
    }

    // Vertical boundary lines
    let vOff = 0;
    for (let sj = 0; sj < sLens.length - 1; sj++) {
        vOff += sLens[sj];
        g.append('line')
            .attr('x1', vOff * cellSize).attr('y1', 10)
            .attr('x2', vOff * cellSize).attr('y2', gridH + 10)
            .attr('stroke', '#e74c3c').attr('stroke-width', 1).attr('stroke-opacity', 0.6);
    }

    // Row labels (S_q)
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

    // Column labels (S)
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
        .text(`Per-request: [${sqLens.map((q, i) => `S_q=${q}, S=${sLens[i]}`).join('; ')}]`);

    // --- Part 2: Attention weights heatmap (after softmax) ---
    let y2 = descY + 92;

    g.append('text').attr('class', 'tensor-label')
        .attr('x', gridW / 2).attr('y', y2)
        .text('Attention Weights (after softmax)');
    y2 += 26;

    // Compute softmax attention weights: S_q rows, S columns per sequence
    const attnWeights = [];
    let rOff = 0;
    for (let si = 0; si < sqLens.length; si++) {
        const nq = sqLens[si];
        const ns = sLens[si];
        let cOff = 0;
        for (let k = 0; k < si; k++) cOff += sLens[k];

        for (let i = 0; i < nq; i++) {
            const row = new Array(dispCols).fill(0);
            const rawScores = [];
            for (let j = 0; j < ns; j++) {
                const allowed = j <= (ns - nq + i);
                if (allowed) {
                    rawScores.push(1.0 + Math.sin(j * 1.7 + i * 0.3) * 0.7);
                } else {
                    rawScores.push(-Infinity);
                }
            }
            const exps = rawScores.map(s => s === -Infinity ? 0 : Math.exp(s));
            const sumExp = exps.reduce((a, b) => a + b, 0);
            const probs = exps.map(e => e / sumExp);
            for (let j = 0; j < ns; j++) {
                row[cOff + j] = probs[j];
            }
            attnWeights.push(row);
        }
        rOff += nq;
    }

    const maxWeight = Math.max(...attnWeights.flat());

    // Draw heatmap: S_q rows × S cols
    for (let i = 0; i < dispRows; i++) {
        for (let j = 0; j < dispCols; j++) {
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
    let hRowOff = 0;
    for (let si = 0; si < sqLens.length - 1; si++) {
        hRowOff += sqLens[si];
        g.append('line')
            .attr('x1', 0).attr('y1', y2 + hRowOff * cellSize)
            .attr('x2', gridW).attr('y2', y2 + hRowOff * cellSize)
            .attr('stroke', '#e74c3c').attr('stroke-width', 1).attr('stroke-opacity', 0.4);
    }
    let hColOff = 0;
    for (let sj = 0; sj < sLens.length - 1; sj++) {
        hColOff += sLens[sj];
        g.append('line')
            .attr('x1', hColOff * cellSize).attr('y1', y2)
            .attr('x2', hColOff * cellSize).attr('y2', y2 + dispRows * cellSize)
            .attr('stroke', '#e74c3c').attr('stroke-width', 1).attr('stroke-opacity', 0.4);
    }

    // Row labels on heatmap
    let lOff = 0;
    for (let si = 0; si < sqLens.length; si++) {
        const nq = sqLens[si];
        const midPos = lOff + nq / 2;
        g.append('text').attr('class', 'dim-label')
            .attr('x', -8).attr('y', y2 + midPos * cellSize + cellSize / 2)
            .attr('text-anchor', 'end').attr('font-size', '8px').attr('fill', '#aaa')
            .text(`S${si}`);
        lOff += nq;
    }

    // "each row sums to 1" label
    g.append('text').attr('class', 'dim-label')
        .attr('x', gridW / 2).attr('y', y2 + dispRows * cellSize + 14)
        .attr('text-anchor', 'middle').attr('fill', '#f39c12')
        .attr('font-size', '10px')
        .text('each row sums to 1');

    // Color scale legend
    const heatLegendY = y2 + dispRows * cellSize + 30;
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
