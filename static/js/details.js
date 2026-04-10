// details.js — Op detail panel visualizations

export function showDetail(op, graph, params) {
    const panel = d3.select('#detail-panel');
    panel.classed('visible', true);
    d3.select('#detail-title').text(op.label);
    d3.select('#detail-desc').html(op.desc || '');

    const svg = d3.select('#detail-svg');
    svg.selectAll('*').remove();
    svg.attr('height', 350);

    const tensorMap = {};
    for (const t of graph.tensors) tensorMap[t.id] = t;

    switch (op.type) {
        case 'matmul':
        case 'compress':
        case 'decompress':
            drawMatmulDetail(svg, op, tensorMap);
            break;
        case 'mask':
            if (params.pagedAttn) {
                drawPagedMaskDetail(svg, op, tensorMap, params);
            } else {
                drawMaskDetail(svg, op, tensorMap, params);
            }
            break;
        case 'broadcast':
            drawBroadcastDetail(svg, op, tensorMap);
            break;
        default:
            drawGenericDetail(svg, op, tensorMap);
    }
}

export function hideDetail() {
    d3.select('#detail-panel').classed('visible', false);
}

// --- Matmul: L-shaped diagram ---

function drawMatmulDetail(svg, op, tensorMap) {
    const A = tensorMap[op.inputs[0]];
    const B_tensor = op.inputs.length > 1 ? tensorMap[op.inputs[1]] : null;
    const C = tensorMap[op.output];
    if (!A || !C) return;

    const shA = A.shape;
    const shC = C.shape;
    const rows_a = shA.length >= 2 ? shA[shA.length - 2] : shA[0];
    const inner = shA[shA.length - 1];
    const cols_b = shC[shC.length - 1];

    const maxDim = 140;
    const scale = (v) => Math.max(20, Math.min(maxDim, Math.sqrt(v) * 10));

    const wA = scale(inner);
    const hA = scale(rows_a);
    const wC = scale(cols_b);
    const hC = hA;
    const hB = scale(inner);

    const pad = 30;
    const originX = wA + pad + 20;
    const originY = hB + pad + 10;

    const g = svg.append('g').attr('transform', 'translate(10, 5)');

    // Matrix A (left of result)
    drawDetailBlock(g, originX - wA - pad, originY, wA, hA, A.color, A.label);
    g.append('text').attr('class', 'dim-label')
        .attr('x', originX - wA - pad + wA / 2).attr('y', originY + hA + 14)
        .attr('text-anchor', 'middle').text(A.dimNames ? A.dimNames[A.dimNames.length - 1] + '=' + inner : inner);
    g.append('text').attr('class', 'dim-label')
        .attr('x', originX - wA - pad - 8).attr('y', originY + hA / 2 + 3)
        .attr('text-anchor', 'end').text(A.dimNames ? A.dimNames[A.dimNames.length - 2] + '=' + rows_a : rows_a);

    // Matrix B (above result)
    if (B_tensor) {
        drawDetailBlock(g, originX, originY - hB - pad, wC, hB, B_tensor.color, B_tensor.label);
        g.append('text').attr('class', 'dim-label')
            .attr('x', originX + wC / 2).attr('y', originY - hB - pad - 6)
            .attr('text-anchor', 'middle').text(B_tensor.dimNames ? B_tensor.dimNames[B_tensor.dimNames.length - 1] + '=' + cols_b : cols_b);
        g.append('text').attr('class', 'dim-label')
            .attr('x', originX - 8).attr('y', originY - hB - pad + hB / 2 + 3)
            .attr('text-anchor', 'end').text(B_tensor.dimNames ? B_tensor.dimNames[B_tensor.dimNames.length - 2] + '=' + inner : inner);
    }

    // Result C (center)
    drawDetailBlock(g, originX, originY, wC, hC, C.color, C.label);
    g.append('text').attr('class', 'dim-label')
        .attr('x', originX + wC / 2).attr('y', originY + hC + 14)
        .attr('text-anchor', 'middle').text(C.dimNames ? C.dimNames[C.dimNames.length - 1] + '=' + cols_b : cols_b);

    // Highlight row/col
    const highlightRow = Math.min(2, rows_a - 1);
    const highlightCol = Math.min(2, cols_b - 1);
    const rowH = hA / rows_a;
    const colW = wC / cols_b;

    g.append('rect')
        .attr('x', originX - wA - pad).attr('y', originY + highlightRow * rowH)
        .attr('width', wA).attr('height', Math.max(rowH, 2))
        .attr('fill', '#fff').attr('fill-opacity', 0.25)
        .attr('stroke', '#fff').attr('stroke-width', 1);

    if (B_tensor) {
        g.append('rect')
            .attr('x', originX + highlightCol * colW).attr('y', originY - hB - pad)
            .attr('width', Math.max(colW, 2)).attr('height', hB)
            .attr('fill', '#fff').attr('fill-opacity', 0.25)
            .attr('stroke', '#fff').attr('stroke-width', 1);
    }

    g.append('rect')
        .attr('x', originX + highlightCol * colW).attr('y', originY + highlightRow * rowH)
        .attr('width', Math.max(colW, 3)).attr('height', Math.max(rowH, 3))
        .attr('fill', '#fff').attr('fill-opacity', 0.4)
        .attr('stroke', '#fff').attr('stroke-width', 2);

    // Dotted lines
    g.append('line')
        .attr('x1', originX - pad / 2).attr('y1', originY + highlightRow * rowH + rowH / 2)
        .attr('x2', originX).attr('y2', originY + highlightRow * rowH + rowH / 2)
        .attr('stroke', '#fff').attr('stroke-width', 1).attr('stroke-dasharray', '3,3').attr('stroke-opacity', 0.5);

    if (B_tensor) {
        g.append('line')
            .attr('x1', originX + highlightCol * colW + colW / 2).attr('y1', originY - pad / 2)
            .attr('x2', originX + highlightCol * colW + colW / 2).attr('y2', originY)
            .attr('stroke', '#fff').attr('stroke-width', 1).attr('stroke-dasharray', '3,3').attr('stroke-opacity', 0.5);
    }

    g.append('text').attr('class', 'dim-label')
        .attr('x', originX + wC + 10).attr('y', originY + highlightRow * rowH + rowH / 2 + 3)
        .attr('text-anchor', 'start').attr('fill', '#aaa')
        .text('← dot product');

    // Batch dims note
    if (shA.length > 2) {
        const batchDims = shA.slice(0, -2).join(' × ');
        const batchNames = A.dimNames ? A.dimNames.slice(0, -2).join(' × ') : '';
        g.append('text').attr('class', 'dim-label')
            .attr('x', originX + wC / 2).attr('y', originY + hC + 30)
            .attr('text-anchor', 'middle').attr('fill', '#666')
            .text(`repeated across ${batchNames} = ${batchDims}`);
    }

    svg.attr('height', originY + hC + 50);
}

// --- Standard mask detail ---

function drawMaskDetail(svg, op, tensorMap, params) {
    const S = params.S;
    const dispS = Math.min(S, 12);
    const cellSize = Math.min(24, 280 / dispS);
    const pad = 40;

    const g = svg.append('g').attr('transform', `translate(${pad}, 20)`);

    g.append('text').attr('class', 'tensor-label')
        .attr('x', dispS * cellSize / 2).attr('y', -6)
        .text(`Causal Mask (${S}×${S})`);

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
                    .text(allowed ? '1' : '-∞');
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
            .text(`(showing ${dispS}×${dispS} of ${S}×${S})`);
    }

    const descY = dispS * cellSize + 50;
    g.append('rect').attr('x', 0).attr('y', descY).attr('width', 12).attr('height', 12)
        .attr('fill', '#1abc9c').attr('fill-opacity', 0.85).attr('rx', 2);
    g.append('text').attr('class', 'dim-label').attr('x', 18).attr('y', descY + 10)
        .attr('fill', '#aaa').text('Attend (i ≥ j): token can see this position');

    g.append('rect').attr('x', 0).attr('y', descY + 20).attr('width', 12).attr('height', 12)
        .attr('fill', '#2c3e50').attr('fill-opacity', 0.5).attr('rx', 2);
    g.append('text').attr('class', 'dim-label').attr('x', 18).attr('y', descY + 30)
        .attr('fill', '#aaa').text('Masked (i < j): future position, set to -∞');

    svg.attr('height', descY + 60);
}

// --- Paged / variable-length mask detail ---

function drawPagedMaskDetail(svg, op, tensorMap, params) {
    const seqLens = params.seqLens.slice(0, params.B);
    const totalS = seqLens.reduce((a, b) => a + b, 0);
    const dispS = Math.min(totalS, 16);
    const cellSize = Math.min(22, 280 / dispS);
    const pad = 40;

    const g = svg.append('g').attr('transform', `translate(${pad}, 20)`);

    g.append('text').attr('class', 'tensor-label')
        .attr('x', dispS * cellSize / 2).attr('y', -6)
        .text(`Variable-Length Causal Mask`);

    // Draw the block-diagonal mask
    let rowOff = 0;
    for (let si = 0; si < seqLens.length; si++) {
        const sLen = Math.min(seqLens[si], dispS - rowOff);
        if (sLen <= 0) break;
        let colOff = 0;

        for (let sj = 0; sj < seqLens.length; sj++) {
            const sLenJ = Math.min(seqLens[sj], dispS - colOff);
            if (sLenJ <= 0) break;

            for (let i = 0; i < sLen; i++) {
                for (let j = 0; j < sLenJ; j++) {
                    const gi = rowOff + i;
                    const gj = colOff + j;
                    if (gi >= dispS || gj >= dispS) continue;

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
            colOff += seqLens[sj];
        }

        // Sequence boundary lines
        if (si < seqLens.length - 1 && rowOff + sLen < dispS) {
            g.append('line')
                .attr('x1', 0).attr('y1', (rowOff + sLen) * cellSize + 10)
                .attr('x2', dispS * cellSize).attr('y2', (rowOff + sLen) * cellSize + 10)
                .attr('stroke', '#e74c3c').attr('stroke-width', 1).attr('stroke-opacity', 0.6);
            g.append('line')
                .attr('x1', (rowOff + sLen) * cellSize).attr('y1', 10)
                .attr('x2', (rowOff + sLen) * cellSize).attr('y2', dispS * cellSize + 10)
                .attr('stroke', '#e74c3c').attr('stroke-width', 1).attr('stroke-opacity', 0.6);
        }

        rowOff += seqLens[si];
    }

    // Labels
    let labelOff = 0;
    for (let si = 0; si < seqLens.length; si++) {
        const sLen = seqLens[si];
        const midPos = Math.min(labelOff + sLen / 2, dispS);
        if (midPos > dispS) break;
        g.append('text').attr('class', 'dim-label')
            .attr('x', -8).attr('y', midPos * cellSize + 14)
            .attr('text-anchor', 'end').attr('font-size', '8px').attr('fill', '#aaa')
            .text(`S${si}`);
        labelOff += sLen;
    }

    const descY = Math.min(dispS, totalS) * cellSize + 40;

    // Legend
    g.append('rect').attr('x', 0).attr('y', descY).attr('width', 12).attr('height', 12)
        .attr('fill', '#1abc9c').attr('fill-opacity', 0.85).attr('rx', 2);
    g.append('text').attr('class', 'dim-label').attr('x', 18).attr('y', descY + 10)
        .attr('fill', '#aaa').text('Causal attend (same seq, i ≥ j)');

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
        .text(`Seq lengths: [${seqLens.join(', ')}], total: ${totalS}`);

    svg.attr('height', descY + 90);
}

// --- Broadcast detail ---

function drawBroadcastDetail(svg, op, tensorMap) {
    const input = tensorMap[op.inputs[0]];
    const output = tensorMap[op.output];
    if (!input || !output) return;

    const g = svg.append('g').attr('transform', 'translate(20, 20)');
    const w = 60, h = 40;

    const inDepth = 20;
    drawDetailBlock3D(g, 20, 60, w, h, inDepth, input.color, input.label);
    g.append('text').attr('class', 'dim-label')
        .attr('x', 20 + w / 2).attr('y', 60 + h + 14)
        .attr('text-anchor', 'middle')
        .text(`[${input.shape.join(', ')}]`);

    g.append('text').attr('x', 130).attr('y', 85).attr('fill', '#888').attr('font-size', '24px')
        .attr('text-anchor', 'middle').text('→');

    const outDepth = 50;
    drawDetailBlock3D(g, 170, 60, w, h, outDepth, output.color, output.label);
    g.append('text').attr('class', 'dim-label')
        .attr('x', 170 + w / 2).attr('y', 60 + h + 14)
        .attr('text-anchor', 'middle')
        .text(`[${output.shape.join(', ')}]`);

    const inShape = input.shape;
    const outShape = output.shape;
    let broadcastDim = '';
    for (let i = 0; i < inShape.length; i++) {
        if (inShape[i] !== outShape[i]) {
            const name = (output.dimNames && output.dimNames[i]) || (input.dimNames && input.dimNames[i]) || `dim${i}`;
            broadcastDim = `${name}: ${inShape[i]} → ${outShape[i]}`;
            break;
        }
    }
    if (broadcastDim) {
        g.append('text').attr('class', 'dim-label')
            .attr('x', 150).attr('y', 160)
            .attr('text-anchor', 'middle').attr('fill', '#7c8cf8')
            .text(broadcastDim);
    }

    svg.attr('height', 200);
}

// --- Generic detail ---

function drawGenericDetail(svg, op) {
    const g = svg.append('g').attr('transform', 'translate(20, 30)');
    let y = 0;

    g.append('text').attr('class', 'tensor-label')
        .attr('x', 0).attr('y', y).text(`Op: ${op.label}`);
    y += 25;

    g.append('text').attr('class', 'dim-label').attr('fill', '#aaa')
        .attr('x', 0).attr('y', y).text(`Type: ${op.type}`);
    y += 20;

    g.append('text').attr('class', 'dim-label').attr('fill', '#aaa')
        .attr('x', 0).attr('y', y).text(`Inputs: ${op.inputs.join(', ')}`);
    y += 20;

    g.append('text').attr('class', 'dim-label').attr('fill', '#aaa')
        .attr('x', 0).attr('y', y).text(`Output: ${op.output}`);

    svg.attr('height', y + 30);
}

// --- Helpers ---

function drawDetailBlock(g, x, y, w, h, color, label) {
    g.append('rect')
        .attr('x', x).attr('y', y).attr('width', w).attr('height', h)
        .attr('fill', color).attr('fill-opacity', 0.8)
        .attr('stroke', d3.color(color).darker(0.3)).attr('stroke-width', 1)
        .attr('rx', 3);
    g.append('text')
        .attr('class', 'tensor-label')
        .attr('x', x + w / 2).attr('y', y + h / 2 + 4)
        .text(label);
}

function drawDetailBlock3D(g, x, y, w, h, d, color, label) {
    const dx = d * 0.7;
    const dy = -d * 0.4;

    g.append('polygon')
        .attr('points', `${x},${y} ${x+dx},${y+dy} ${x+w+dx},${y+dy} ${x+w},${y}`)
        .attr('fill', d3.color(color).darker(0.4)).attr('stroke', 'none');
    g.append('polygon')
        .attr('points', `${x+w},${y} ${x+w+dx},${y+dy} ${x+w+dx},${y+h+dy} ${x+w},${y+h}`)
        .attr('fill', d3.color(color).darker(0.8)).attr('stroke', 'none');
    g.append('rect')
        .attr('x', x).attr('y', y).attr('width', w).attr('height', h)
        .attr('fill', color).attr('fill-opacity', 0.85)
        .attr('stroke', d3.color(color).darker(0.3)).attr('stroke-width', 1);
    g.append('text')
        .attr('class', 'tensor-label')
        .attr('x', x + w / 2).attr('y', y + h / 2 + 4)
        .text(label);
}
