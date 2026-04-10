// details.js — Op detail panel visualizations

// Track currently displayed detail for live refresh
let _currentDetail = null;  // { type: 'op'|'tensor', id, graphId }

// Get available detail SVG width and center x, accounting for padding/margins
function detailMetrics(svg) {
    const w = svg.node().clientWidth || svg.node().getBoundingClientRect().width || 460;
    return { w, cx: w / 2, pad: 20 };
}

export function showDetail(op, graph, params) {
    _currentDetail = { type: 'op', id: op.id, graphId: graph.id };
    _renderOpDetail(op, graph, params);
}

function _renderOpDetail(op, graph, params) {
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
        case 'rope':
            drawRopeDetail(svg, op, tensorMap);
            break;
        default:
            drawGenericDetail(svg, op, tensorMap);
    }
}

export function showTensorDetail(tensor, params) {
    _currentDetail = { type: 'tensor', id: tensor.id };
    _renderTensorDetail(tensor, params);
}

function _renderTensorDetail(tensor, params) {
    const panel = d3.select('#detail-panel');
    panel.classed('visible', true);
    d3.select('#detail-title').text(tensor.label);
    d3.select('#detail-desc').html(tensor.desc || '');

    const svg = d3.select('#detail-svg');
    svg.selectAll('*').remove();

    if (tensor.badge === 'PAGED' && params.pagedAttn) {
        svg.attr('height', 100);
        drawPagedCacheDetail(svg, tensor, params);
    } else if (tensor.type === 'mask' && tensor.pagedMask && params.pagedAttn) {
        svg.attr('height', 350);
        drawPagedMaskTensorDetail(svg, tensor, params);
    } else if (tensor.type === 'mask') {
        svg.attr('height', 350);
        drawMaskTensorDetail(svg, tensor, params);
    } else {
        drawTensorShapeDetail(svg, tensor);
    }
}

export function refreshDetail(graphs, params) {
    if (!_currentDetail) return;
    const panel = d3.select('#detail-panel');
    if (!panel.classed('visible')) return;

    // Find the matching op or tensor in the fresh graphs
    for (const graph of graphs) {
        if (_currentDetail.type === 'op') {
            const op = graph.ops.find(o => o.id === _currentDetail.id);
            if (op && (!_currentDetail.graphId || graph.id === _currentDetail.graphId)) {
                _renderOpDetail(op, graph, params);
                return;
            }
        } else {
            const tensor = graph.tensors.find(t => t.id === _currentDetail.id);
            if (tensor) {
                _renderTensorDetail(tensor, params);
                return;
            }
        }
    }
}

export function hideDetail() {
    _currentDetail = null;
    d3.select('#detail-panel').classed('visible', false);
}

// --- Matmul: L-shaped diagram ---

function drawMatmulDetail(svg, op, tensorMap) {
    const { w: svgW } = detailMetrics(svg);
    const A = tensorMap[op.inputs[0]];
    const B_tensor = op.inputs.length > 1 ? tensorMap[op.inputs[1]] : null;
    const C = tensorMap[op.output];
    if (!A || !C) return;

    const shA = A.shape;
    const shC = C.shape;
    const rows_a = shA.length >= 2 ? shA[shA.length - 2] : shA[0];
    const inner = shA[shA.length - 1];
    // Determine cols from B tensor, accounting for possible transpose (e.g. Q @ K^T).
    // If B's last dim == A's inner dim AND output's last dim == B's second-to-last,
    // then B is transposed and cols come from B's second-to-last dim.
    let cols_b;
    if (B_tensor) {
        const bShape = B_tensor.shape;
        const bLast = bShape[bShape.length - 1];
        const bSecondLast = bShape.length >= 2 ? bShape[bShape.length - 2] : bLast;
        const isTransposed = bLast === inner && shC[shC.length - 1] === bSecondLast && bLast !== bSecondLast;
        cols_b = isTransposed ? bSecondLast : bLast;
    } else {
        cols_b = shC[shC.length - 1];
    }

    const maxDim = Math.min(180, svgW * 0.35);
    const scale = (v) => Math.max(24, Math.min(maxDim, Math.sqrt(v) * 12));

    const wA = scale(inner);
    const hA = scale(rows_a);
    const wC = scale(cols_b);
    const hC = hA;
    const hB = scale(inner);

    const pad = 30;
    // Center the L-shaped diagram: result block center should be near SVG center
    const totalW = wA + pad + wC + 40;
    const leftMargin = Math.max(10, (svgW - totalW) / 2);
    const originX = leftMargin + wA + pad;
    const originY = hB + pad + 10;

    const g = svg.append('g').attr('transform', 'translate(0, 5)');

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
        const shB = B_tensor.shape;
        drawDetailBlock(g, originX, originY - hB - pad, wC, hB, B_tensor.color, B_tensor.label);
        const bColName = B_tensor.dimNames ? B_tensor.dimNames[B_tensor.dimNames.length - 1] : '';
        g.append('text').attr('class', 'dim-label')
            .attr('x', originX + wC / 2).attr('y', originY - hB - pad - 6)
            .attr('text-anchor', 'middle').text(bColName ? bColName + '=' + cols_b : cols_b);
        const bRowName = B_tensor.dimNames ? B_tensor.dimNames[B_tensor.dimNames.length - 2] : '';
        g.append('text').attr('class', 'dim-label')
            .attr('x', originX - 8).attr('y', originY - hB - pad + hB / 2 + 3)
            .attr('text-anchor', 'end').text(bRowName ? bRowName + '=' + shB[shB.length - 2] : inner);
    }

    // Result C (center)
    drawDetailBlock(g, originX, originY, wC, hC, C.color, C.label);
    const resultColName = B_tensor && B_tensor.dimNames
        ? B_tensor.dimNames[B_tensor.dimNames.length - 1]
        : (C.dimNames ? C.dimNames[C.dimNames.length - 1] : '');
    g.append('text').attr('class', 'dim-label')
        .attr('x', originX + wC / 2).attr('y', originY + hC + 14)
        .attr('text-anchor', 'middle').text(resultColName ? resultColName + '=' + cols_b : cols_b);

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

    let noteY = originY + hC + 30;

    // Batch dims note
    if (shA.length > 2) {
        const batchDims = shA.slice(0, -2).join(' \u00d7 ');
        const batchNames = A.dimNames ? A.dimNames.slice(0, -2).join(' \u00d7 ') : '';
        g.append('text').attr('class', 'dim-label')
            .attr('x', originX + wC / 2).attr('y', noteY)
            .attr('text-anchor', 'middle').attr('fill', '#666')
            .text(`repeated across ${batchNames} = ${batchDims}`);
        noteY += 18;
    }

    // Reshape visualization if output shape differs from matmul result
    const matmulResultShape = B_tensor
        ? [...shA.slice(0, -1), cols_b]
        : shC;
    const needsReshape = cols_b !== shC[shC.length - 1] ||
        matmulResultShape.join(',') !== shC.join(',');

    if (needsReshape) {
        noteY += 14;
        g.append('text').attr('class', 'tensor-label')
            .attr('x', svgW / 2).attr('y', noteY)
            .attr('text-anchor', 'middle')
            .attr('font-size', '14px')
            .text('Reshape / Concat Heads');
        noteY += 24;

        // Scale for reshape blocks — large enough to read clearly
        const reshScale = (v) => Math.max(30, Math.min(120, Math.sqrt(v) * 8));

        // Before block dimensions (matmul result shape)
        const beforeW = reshScale(matmulResultShape[matmulResultShape.length - 1]);
        const beforeH = reshScale(matmulResultShape[matmulResultShape.length - 2]);
        const beforeDepthVal = matmulResultShape.length >= 3
            ? matmulResultShape.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
        const beforeD = Math.max(12, Math.min(40, Math.sqrt(beforeDepthVal) * 5));

        // After block dimensions (final output shape)
        const afterW = reshScale(shC[shC.length - 1]);
        const afterH = reshScale(shC[shC.length - 2]);
        const afterDepthVal = shC.length >= 3
            ? shC.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
        const afterD = Math.max(12, Math.min(40, Math.sqrt(afterDepthVal) * 5));

        // Layout: center both blocks + arrow in available width
        const arrowGap = 100;
        const beforeTotalW = beforeW + beforeD * 0.7;
        const afterTotalW = afterW + afterD * 0.7;
        const totalReshW = beforeTotalW + arrowGap + afterTotalW;
        const reshLeftMargin = Math.max(40, (svgW - totalReshW) / 2);

        const beforeX = reshLeftMargin;
        const beforeTopPad = Math.max(beforeD, 12) * 0.4;
        const afterTopPad = Math.max(afterD, 12) * 0.4;
        const maxTopPad = Math.max(beforeTopPad, afterTopPad);

        // Draw "before" block (matmul result shape)
        const beforeGrp = matmulResultShape.length === 4
            ? { outer: matmulResultShape[0], inner: matmulResultShape[1] } : null;
        drawDetailBlock3D(g, beforeX, noteY + maxTopPad, beforeW, beforeH, beforeD, C.color, '', beforeGrp);

        // Edge dimension labels on before block
        // The matmul result shape takes A's batch/row dims + B's col dim
        const aNames = A.dimNames || [];
        const bNames = B_tensor && B_tensor.dimNames ? B_tensor.dimNames : [];
        const beforeNames = [...aNames.slice(0, -1), bNames[bNames.length - 1] || aNames[aNames.length - 1] || ''];
        const lastBeforeDim = beforeNames[beforeNames.length - 1] || '';
        const secLastBeforeDim = beforeNames[beforeNames.length - 2] || '';
        // Show compound dim breakdown if reshape will split D → n_h × d_h
        let beforeBottomLabel = `${lastBeforeDim}=${matmulResultShape[matmulResultShape.length - 1]}`;
        if (matmulResultShape.length === 3 && shC.length === 4) {
            const n_h_val = shC[1];
            const d_h_val = shC[3];
            beforeBottomLabel = `${lastBeforeDim} = ${n_h_val}\u00d7${d_h_val} = ${matmulResultShape[matmulResultShape.length - 1]}`;
        }
        g.append('text').attr('class', 'dim-label')
            .attr('x', beforeX + beforeW / 2).attr('y', noteY + maxTopPad + beforeH + 16)
            .attr('text-anchor', 'middle')
            .text(beforeBottomLabel);
        g.append('text').attr('class', 'dim-label')
            .attr('x', beforeX - 8).attr('y', noteY + maxTopPad + beforeH / 2 + 4)
            .attr('text-anchor', 'end')
            .text(`${secLastBeforeDim}=${matmulResultShape[matmulResultShape.length - 2]}`);
        if (matmulResultShape.length >= 3) {
            const depthLabel = matmulResultShape.length === 4
                ? `${beforeNames[0] || 'B'}\u00b7${beforeNames[1] || 'n_h'}=${beforeDepthVal}`
                : `${beforeNames[0] || 'B'}=${matmulResultShape[0]}`;
            g.append('text').attr('class', 'dim-label')
                .attr('x', beforeX + beforeW / 2 + beforeD * 0.35)
                .attr('y', noteY + maxTopPad - beforeD * 0.4 - 12)
                .attr('text-anchor', 'middle').attr('font-size', '9px')
                .text(depthLabel);
        }

        // Arrow between blocks
        const arrowMidX = reshLeftMargin + beforeTotalW + arrowGap / 2;
        const arrowMidY = noteY + maxTopPad + Math.max(beforeH, afterH) / 2;
        g.append('text')
            .attr('x', arrowMidX).attr('y', arrowMidY + 5)
            .attr('text-anchor', 'middle')
            .attr('fill', '#7c8cf8').attr('font-size', '22px')
            .text('\u2192');
        g.append('text').attr('class', 'dim-label')
            .attr('x', arrowMidX).attr('y', arrowMidY - 16)
            .attr('text-anchor', 'middle')
            .attr('fill', '#7c8cf8').attr('font-size', '11px')
            .text('reshape');

        // Draw "after" block (final output shape)
        const afterX = reshLeftMargin + beforeTotalW + arrowGap;
        const afterGrp = shC.length === 4 ? { outer: shC[0], inner: shC[1] } : null;
        drawDetailBlock3D(g, afterX, noteY + maxTopPad, afterW, afterH, afterD, C.color, C.label, afterGrp);

        // Edge dimension labels on after block
        const outNames = C.dimNames || [];
        const lastOutDim = outNames[outNames.length - 1] || '';
        const secLastOutDim = outNames[outNames.length - 2] || '';
        // Show compound dim breakdown if reshape collapsed n_h × d_h → D
        let afterBottomLabel = `${lastOutDim}=${shC[shC.length - 1]}`;
        if (matmulResultShape.length === 4 && shC.length === 3) {
            const n_h_val = matmulResultShape[1];
            const d_h_val = matmulResultShape[3];
            afterBottomLabel = `${lastOutDim} = ${n_h_val}\u00d7${d_h_val} = ${shC[shC.length - 1]}`;
        }
        g.append('text').attr('class', 'dim-label')
            .attr('x', afterX + afterW / 2).attr('y', noteY + maxTopPad + afterH + 16)
            .attr('text-anchor', 'middle')
            .text(afterBottomLabel);
        g.append('text').attr('class', 'dim-label')
            .attr('x', afterX - 8).attr('y', noteY + maxTopPad + afterH / 2 + 4)
            .attr('text-anchor', 'end')
            .text(`${secLastOutDim}=${shC[shC.length - 2]}`);
        if (shC.length >= 3) {
            const depthLabel = shC.length === 4
                ? `${outNames[0] || 'B'}\u00b7${outNames[1] || 'n_h'}=${afterDepthVal}`
                : `${outNames[0] || 'B'}=${shC[0]}`;
            g.append('text').attr('class', 'dim-label')
                .attr('x', afterX + afterW / 2 + afterD * 0.35)
                .attr('y', noteY + maxTopPad - afterD * 0.4 - 12)
                .attr('text-anchor', 'middle').attr('font-size', '9px')
                .text(depthLabel);
        }

        // Advance past blocks
        const maxBlockH = maxTopPad + Math.max(beforeH, afterH);
        noteY += maxBlockH + 36;

        // Show the concat explanation if n_h dimension is being collapsed
        if (matmulResultShape.length === 4 && shC.length === 3) {
            const n_h_val = matmulResultShape[1];
            const d_h_val = matmulResultShape[3];
            const D_val = shC[shC.length - 1];
            g.append('text').attr('class', 'dim-label')
                .attr('x', svgW / 2).attr('y', noteY)
                .attr('text-anchor', 'middle').attr('fill', '#7c8cf8').attr('font-size', '11px')
                .text(`Concat ${n_h_val} heads: n_h \u00d7 d_h = ${n_h_val} \u00d7 ${d_h_val} = ${D_val} = D`);
            noteY += 22;
        }
    }

    // TP all-reduce visualization
    if (op.tpAllReduce && op.tpSize > 1) {
        noteY += 10;
        noteY = drawAllReduceSection(g, originX - 20, noteY, op.tpSize, C);
    }

    svg.attr('height', noteY + 20);
}

// --- Standard mask detail ---

function drawMaskDetail(svg, _op, _tensorMap, params) {
    const { w: svgW } = detailMetrics(svg);
    const S = params.S;
    const dispS = Math.min(S, 12);
    const cellSize = Math.min(28, Math.max(18, (svgW - 140) / dispS));
    const gridW = dispS * cellSize;
    const pad = Math.max(30, (svgW - gridW - 60) / 2);

    const g = svg.append('g').attr('transform', `translate(${pad}, 38)`);

    // --- Part 1: Causal mask ---
    g.append('text').attr('class', 'tensor-label')
        .attr('x', gridW / 2).attr('y', -14)
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

function drawMaskTensorDetail(svg, _tensor, params) {
    const { w: svgW } = detailMetrics(svg);
    const S = params.S;
    const dispS = Math.min(S, 12);
    const cellSize = Math.min(28, Math.max(18, (svgW - 140) / dispS));
    const gridW = dispS * cellSize;
    const pad = Math.max(30, (svgW - gridW - 60) / 2);

    const g = svg.append('g').attr('transform', `translate(${pad}, 38)`);

    g.append('text').attr('class', 'tensor-label')
        .attr('x', gridW / 2).attr('y', -14)
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

function drawPagedMaskTensorDetail(svg, _tensor, params) {
    const { w: svgW } = detailMetrics(svg);
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

// --- Paged / variable-length mask detail ---

function drawPagedMaskDetail(svg, _op, _tensorMap, params) {
    const { w: svgW } = detailMetrics(svg);
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

// --- Broadcast detail ---

function drawBroadcastDetail(svg, op, tensorMap) {
    const { cx: svgCx } = detailMetrics(svg);
    const input = tensorMap[op.inputs[0]];
    const output = tensorMap[op.output];
    if (!input || !output) return;

    const gPad = 10;
    const g = svg.append('g').attr('transform', `translate(${gPad}, 20)`);
    const mid = svgCx - gPad;
    const inShape = input.shape;
    const outShape = output.shape;
    const inNames = input.dimNames || [];
    const outNames = output.dimNames || [];

    const scale = (v) => Math.max(24, Math.min(90, Math.sqrt(v) * 7));

    // Input block dimensions
    const inW = scale(inShape[inShape.length - 1]);
    const inH = scale(inShape[inShape.length - 2]);
    const inDepthVal = inShape.length >= 3 ? inShape[0] * (inShape.length >= 4 ? inShape[1] : 1) : 1;
    const inD = Math.max(10, Math.min(35, Math.sqrt(inDepthVal) * 5));

    // Output block dimensions
    const outW = scale(outShape[outShape.length - 1]);
    const outH = scale(outShape[outShape.length - 2]);
    const outDepthVal = outShape.length >= 3 ? outShape[0] * (outShape.length >= 4 ? outShape[1] : 1) : 1;
    const outD = Math.max(10, Math.min(60, Math.sqrt(outDepthVal) * 5));

    // Center the two blocks with arrow in available width
    const arrowGap = 50;
    const totalBlockW = inW + inD * 0.7 + arrowGap + outW + outD * 0.7;
    const inX = Math.max(30, (mid * 2 - totalBlockW) / 2);
    const topPad = Math.max(inD, outD) * 0.4 + 10;
    const blockY = topPad;

    // Draw input block
    const inGrp = inShape.length === 4 ? { outer: inShape[0], inner: inShape[1] } : null;
    drawDetailBlock3D(g, inX, blockY, inW, inH, inD, input.color, input.label, inGrp);

    // Input edge labels
    const lastIn = inNames[inNames.length - 1] || '';
    const secLastIn = inNames[inNames.length - 2] || '';
    g.append('text').attr('class', 'dim-label')
        .attr('x', inX + inW / 2).attr('y', blockY + inH + 14)
        .attr('text-anchor', 'middle')
        .text(`${lastIn}=${inShape[inShape.length - 1]}`);
    g.append('text').attr('class', 'dim-label')
        .attr('x', inX - 6).attr('y', blockY + inH / 2 + 3)
        .attr('text-anchor', 'end')
        .text(`${secLastIn}=${inShape[inShape.length - 2]}`);
    if (inShape.length >= 3) {
        const depthLabel = inShape.length === 4
            ? `${inNames[0] || ''}·${inNames[1] || ''}=${inDepthVal}`
            : `${inNames[0] || ''}=${inShape[0]}`;
        g.append('text').attr('class', 'dim-label')
            .attr('x', inX + inW + inD * 0.7 / 2 + 6)
            .attr('y', blockY - inD * 0.4 / 2 - 2)
            .text(depthLabel);
    }

    // Arrow
    const arrowX = inX + inW + inD * 0.7 + 25;
    g.append('text').attr('x', arrowX).attr('y', blockY + inH / 2 + 4)
        .attr('fill', '#888').attr('font-size', '24px')
        .attr('text-anchor', 'middle').text('\u2192');

    // Draw output block
    const outX = arrowX + 25;
    const outGrp = outShape.length === 4 ? { outer: outShape[0], inner: outShape[1] } : null;
    drawDetailBlock3D(g, outX, blockY, outW, outH, outD, output.color, output.label, outGrp);

    // Output edge labels
    const lastOut = outNames[outNames.length - 1] || '';
    const secLastOut = outNames[outNames.length - 2] || '';
    g.append('text').attr('class', 'dim-label')
        .attr('x', outX + outW / 2).attr('y', blockY + outH + 14)
        .attr('text-anchor', 'middle')
        .text(`${lastOut}=${outShape[outShape.length - 1]}`);
    g.append('text').attr('class', 'dim-label')
        .attr('x', outX - 6).attr('y', blockY + outH / 2 + 3)
        .attr('text-anchor', 'end')
        .text(`${secLastOut}=${outShape[outShape.length - 2]}`);
    if (outShape.length >= 3) {
        const depthLabel = outShape.length === 4
            ? `${outNames[0] || ''}·${outNames[1] || ''}=${outDepthVal}`
            : `${outNames[0] || ''}=${outShape[0]}`;
        g.append('text').attr('class', 'dim-label')
            .attr('x', outX + outW + outD * 0.7 / 2 + 6)
            .attr('y', blockY - outD * 0.4 / 2 - 2)
            .text(depthLabel);
    }

    // Broadcast dimension annotation
    let noteY = blockY + Math.max(inH, outH) + 32;
    for (let i = 0; i < inShape.length; i++) {
        if (inShape[i] !== outShape[i]) {
            const name = outNames[i] || inNames[i] || `dim${i}`;
            g.append('text').attr('class', 'dim-label')
                .attr('x', mid).attr('y', noteY)
                .attr('text-anchor', 'middle').attr('fill', '#7c8cf8')
                .text(`${name}: ${inShape[i]} \u2192 ${outShape[i]} (repeat ${outShape[i] / inShape[i]}\u00d7)`);
            noteY += 18;
        }
    }

    svg.attr('height', noteY + 10);
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

// --- Paged KV cache detail ---

function drawPagedCacheDetail(svg, _tensor, params) {
    const bs = params.block_size;
    const ctxLens = params.seqLens.slice(0, params.B);
    const queryLens = params.queryLens.slice(0, params.B);
    const totalLens = ctxLens.map((c, i) => c + queryLens[i]);
    const blocksPerSeq = totalLens.map(s => Math.ceil(s / bs));
    const totalBlocks = blocksPerSeq.reduce((a, b) => a + b, 0);
    const pad = 20;

    const g = svg.append('g').attr('transform', `translate(${pad}, 10)`);

    g.append('text').attr('class', 'tensor-label')
        .attr('x', 130).attr('y', 0)
        .text('Paged KV Cache Layout');

    // Block table section
    let y = 20;
    g.append('text').attr('class', 'dim-label')
        .attr('x', 0).attr('y', y).attr('fill', '#aaa').attr('font-size', '10px')
        .text('Block Table (logical \u2192 physical mapping):');
    y += 16;

    const seqColors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];
    const blockW = Math.min(36, 280 / Math.max(totalBlocks, 4));
    const blockH = 24;

    // Draw block table per sequence
    for (let si = 0; si < totalLens.length; si++) {
        const sLen = totalLens[si];
        const nBlocks = blocksPerSeq[si];
        const color = seqColors[si % seqColors.length];
        const typeStr = queryLens[si] > 1 ? 'prefill' : 'decode';

        g.append('text').attr('class', 'dim-label')
            .attr('x', 0).attr('y', y + blockH / 2 + 3)
            .attr('fill', '#aaa').attr('font-size', '9px')
            .text(`Req ${si} (${ctxLens[si]}+${queryLens[si]}, ${typeStr}):`);

        const tableX = 120;
        for (let bi = 0; bi < nBlocks; bi++) {
            const bx = tableX + bi * (blockW + 3);
            const tokensInBlock = Math.min(bs, sLen - bi * bs);
            const fillRatio = tokensInBlock / bs;

            // Block background (full)
            g.append('rect')
                .attr('x', bx).attr('y', y)
                .attr('width', blockW).attr('height', blockH)
                .attr('fill', '#1e2030')
                .attr('stroke', color).attr('stroke-width', 1.5)
                .attr('rx', 3);

            // Filled portion
            g.append('rect')
                .attr('x', bx + 1).attr('y', y + 1)
                .attr('width', (blockW - 2) * fillRatio).attr('height', blockH - 2)
                .attr('fill', color).attr('fill-opacity', 0.4)
                .attr('rx', 2);

            // Block label
            if (blockW >= 20) {
                g.append('text')
                    .attr('x', bx + blockW / 2).attr('y', y + blockH / 2 + 4)
                    .attr('text-anchor', 'middle')
                    .attr('fill', '#ddd').attr('font-size', '8px')
                    .text(`${tokensInBlock}/${bs}`);
            }
        }
        y += blockH + 8;
    }

    // Physical memory layout
    y += 8;
    g.append('text').attr('class', 'dim-label')
        .attr('x', 0).attr('y', y).attr('fill', '#aaa').attr('font-size', '10px')
        .text('Physical blocks (non-contiguous in memory):');
    y += 16;

    // Draw physical blocks interleaved
    const physicalBlocks = [];
    for (let si = 0; si < totalLens.length; si++) {
        for (let bi = 0; bi < blocksPerSeq[si]; bi++) {
            physicalBlocks.push({ seq: si, block: bi, tokens: Math.min(bs, totalLens[si] - bi * bs) });
        }
    }
    // Shuffle to show non-contiguous nature
    const shuffled = [...physicalBlocks];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = (i * 7 + 3) % (i + 1); // deterministic shuffle
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }

    const physBlockW = Math.min(36, 280 / Math.max(shuffled.length, 4));
    for (let i = 0; i < shuffled.length; i++) {
        const pb = shuffled[i];
        const color = seqColors[pb.seq % seqColors.length];
        const bx = i * (physBlockW + 3);

        g.append('rect')
            .attr('x', bx).attr('y', y)
            .attr('width', physBlockW).attr('height', blockH)
            .attr('fill', color).attr('fill-opacity', 0.35)
            .attr('stroke', color).attr('stroke-width', 1)
            .attr('rx', 3);

        if (physBlockW >= 20) {
            g.append('text')
                .attr('x', bx + physBlockW / 2).attr('y', y + blockH / 2 + 4)
                .attr('text-anchor', 'middle')
                .attr('fill', '#ddd').attr('font-size', '8px')
                .text(`S${pb.seq}`);
        }
    }
    y += blockH + 8;

    // Block shape info
    y += 4;
    g.append('text').attr('class', 'dim-label')
        .attr('x', 0).attr('y', y).attr('fill', '#777')
        .text(`Each block: [${bs}, n_heads, d_h]`);
    y += 14;
    g.append('text').attr('class', 'dim-label')
        .attr('x', 0).attr('y', y).attr('fill', '#777')
        .text(`Total blocks: ${totalBlocks} (${blocksPerSeq.join(' + ')})`);
    y += 14;
    g.append('text').attr('class', 'dim-label')
        .attr('x', 0).attr('y', y).attr('fill', '#777')
        .text(`Block size: ${bs} tokens/block`);

    svg.attr('height', y + 20);
}

// --- All-Reduce detail ---

const TP_COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#c0392b'];

function drawAllReduceSection(g, x, y, tpSize, outputTensor) {
    g.append('text').attr('class', 'tensor-label')
        .attr('x', x + 130).attr('y', y)
        .text('All-Reduce (sum across TP ranks)');

    const rankH = 22;
    const rankW = 80;
    const rankGap = 4;
    const rankX = x;
    const startY = y + 14;
    const displayRanks = Math.min(tpSize, 8);

    for (let r = 0; r < displayRanks; r++) {
        const ry = startY + r * (rankH + rankGap);
        g.append('rect')
            .attr('x', rankX).attr('y', ry)
            .attr('width', rankW).attr('height', rankH)
            .attr('fill', TP_COLORS[r % TP_COLORS.length])
            .attr('fill-opacity', 0.6)
            .attr('rx', 3);
        g.append('text')
            .attr('x', rankX + rankW / 2).attr('y', ry + rankH / 2 + 4)
            .attr('text-anchor', 'middle')
            .attr('fill', '#fff')
            .attr('font-size', '9px')
            .attr('font-weight', '500')
            .text(`Rank ${r}`);
    }

    // Sum symbol
    const totalH = displayRanks * (rankH + rankGap) - rankGap;
    const sumX = rankX + rankW + 30;
    const sumY = startY + totalH / 2 - 14;

    // Lines from ranks to sum
    for (let r = 0; r < displayRanks; r++) {
        const ry = startY + r * (rankH + rankGap) + rankH / 2;
        g.append('line')
            .attr('x1', rankX + rankW + 2).attr('y1', ry)
            .attr('x2', sumX).attr('y2', sumY + 14)
            .attr('stroke', '#555').attr('stroke-width', 1);
    }

    g.append('rect')
        .attr('x', sumX).attr('y', sumY)
        .attr('width', 28).attr('height', 28)
        .attr('fill', '#1e2030')
        .attr('stroke', '#3498db')
        .attr('stroke-width', 2)
        .attr('rx', 4);
    g.append('text')
        .attr('x', sumX + 14).attr('y', sumY + 19)
        .attr('text-anchor', 'middle')
        .attr('fill', '#3498db')
        .attr('font-size', '16px')
        .attr('font-weight', '600')
        .text('\u03a3');

    // Arrow from sum to output
    const outX = sumX + 44;
    g.append('line')
        .attr('x1', sumX + 30).attr('y1', sumY + 14)
        .attr('x2', outX - 2).attr('y2', sumY + 14)
        .attr('stroke', '#555').attr('stroke-width', 1.5);

    // Output block
    const outW = 70;
    g.append('rect')
        .attr('x', outX).attr('y', sumY - 2)
        .attr('width', outW).attr('height', 32)
        .attr('fill', outputTensor.color)
        .attr('fill-opacity', 0.7)
        .attr('rx', 3);
    g.append('text')
        .attr('x', outX + outW / 2).attr('y', sumY + 17)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff')
        .attr('font-size', '10px')
        .attr('font-weight', '600')
        .text(outputTensor.label);

    return startY + totalH + 20;
}

// --- Softmax bar chart ---

function drawSoftmaxSection(g, x, y, dispS, cellSize, precomputedWeights) {
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

// --- Tensor shape detail ---

function drawTensorShapeDetail(svg, tensor) {
    const { cx } = detailMetrics(svg);
    const shape = tensor.shape;
    const dimNames = tensor.dimNames || [];
    const gPad = 20;
    const g = svg.append('g').attr('transform', `translate(${gPad}, 24)`);
    const mid = cx - gPad;

    // Shape title
    g.append('text')
        .attr('x', mid).attr('y', 0)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff').attr('font-size', '15px').attr('font-weight', '600')
        .attr('font-family', 'Inter, system-ui, sans-serif')
        .text(tensor.label);

    // Shape subtitle
    const shapeStr = shape.map((d, i) =>
        `${dimNames[i] || ''}${dimNames[i] ? '=' : ''}${d}`
    ).join(', ');
    g.append('text').attr('class', 'dim-label')
        .attr('x', mid).attr('y', 20)
        .attr('text-anchor', 'middle').attr('fill', '#7c8cf8')
        .text(`[${shapeStr}]`);

    // Draw a visual representation — fixed size to fill available space
    // (relative sizes are shown in the graph and op details)
    const blockW = Math.min(200, mid * 0.9);
    const blockH = 120;

    if (shape.length === 2) {
        const w = blockW, h = blockH;
        const x = mid - w / 2, y = 36;
        drawDetailBlock(g, x, y, w, h, tensor.color, tensor.label);

        g.append('text').attr('class', 'dim-label')
            .attr('x', x + w / 2).attr('y', y + h + 18)
            .attr('text-anchor', 'middle')
            .text(`${dimNames[1] || 'cols'} = ${shape[1]}`);
        g.append('text').attr('class', 'dim-label')
            .attr('x', x - 10).attr('y', y + h / 2 + 4)
            .attr('text-anchor', 'end')
            .text(`${dimNames[0] || 'rows'} = ${shape[0]}`);

        svg.attr('height', 24 + y + h + 56);
    } else if (shape.length >= 3) {
        const w = blockW;
        const h = blockH;
        const d = 45;
        const depth = shape.length === 4 ? shape[0] * shape[1] : shape[0];
        const x = mid - w / 2, y = d * 0.4 + 36;

        const grp = shape.length === 4 ? { outer: shape[0], inner: shape[1] } : null;
        drawDetailBlock3D(g, x, y, w, h, d, tensor.color, tensor.label, grp);

        const lastDim = dimNames[dimNames.length - 1] || 'cols';
        const secondLast = dimNames[dimNames.length - 2] || 'rows';
        g.append('text').attr('class', 'dim-label')
            .attr('x', x + w / 2).attr('y', y + h + 18)
            .attr('text-anchor', 'middle')
            .text(`${lastDim} = ${shape[shape.length - 1]}`);
        g.append('text').attr('class', 'dim-label')
            .attr('x', x - 10).attr('y', y + h / 2 + 4)
            .attr('text-anchor', 'end')
            .text(`${secondLast} = ${shape[shape.length - 2]}`);

        if (shape.length === 3) {
            g.append('text').attr('class', 'dim-label')
                .attr('x', x + w + d * 0.7 / 2 + 10)
                .attr('y', y - d * 0.4 / 2 - 4)
                .text(`${dimNames[0] || 'depth'} = ${shape[0]}`);
        } else if (shape.length === 4) {
            g.append('text').attr('class', 'dim-label')
                .attr('x', x + w + d * 0.7 / 2 + 10)
                .attr('y', y - d * 0.4 / 2 - 4)
                .text(`${dimNames[0] || 'B'} \u00d7 ${dimNames[1] || 'n_h'} = ${depth}`);
        }

        let noteY = y + h + 40;
        if (tensor.type === 'weight') {
            g.append('text').attr('class', 'dim-label')
                .attr('x', mid).attr('y', noteY)
                .attr('text-anchor', 'middle').attr('fill', '#7c8cf8')
                .text('Learned weight matrix (dashed border)');
            noteY += 20;
        }
        if (tensor.badge) {
            g.append('text').attr('class', 'dim-label')
                .attr('x', mid).attr('y', noteY)
                .attr('text-anchor', 'middle').attr('fill', '#e67e22')
                .text(`Badge: ${tensor.badge}`);
            noteY += 20;
        }

        const total = shape.reduce((a, b) => a * b, 1);
        g.append('text').attr('class', 'dim-label')
            .attr('x', mid).attr('y', noteY)
            .attr('text-anchor', 'middle').attr('fill', '#666')
            .text(`Total elements: ${total.toLocaleString()}`);
        noteY += 20;

        svg.attr('height', noteY + 34);
    } else {
        svg.attr('height', 50);
    }
}

// --- RoPE detail ---

function drawRopeDetail(svg, op, tensorMap) {
    const { w: svgW, cx: svgCx } = detailMetrics(svg);
    const input = tensorMap[op.inputs[0]];
    const output = tensorMap[op.output];
    if (!input || !output) return;

    const gPad = 16;
    const g = svg.append('g').attr('transform', `translate(${gPad}, 20)`);
    const mid = svgCx - gPad;
    const d_h = input.shape[input.shape.length - 1];
    const numPairs = Math.floor(d_h / 2);
    const dispDims = Math.min(d_h, 12);
    const dispPairs = Math.floor(dispDims / 2);

    // --- Part 0: Show tensor extraction (stacked vertically) ---
    g.append('text').attr('class', 'tensor-label')
        .attr('x', mid).attr('y', 0)
        .attr('font-size', '14px')
        .text('RoPE: Dimension Pairing');

    // Draw isometric tensor block showing the full shape
    const inShape = input.shape;
    const inNames = input.dimNames || [];
    const blockScale = (v) => Math.max(30, Math.min(90, Math.sqrt(v) * 8));
    const bw = blockScale(inShape[inShape.length - 1]);
    const bh = blockScale(inShape[inShape.length - 2]);
    const bDepthVal = inShape.length >= 3 ? inShape[0] * (inShape.length >= 4 ? inShape[1] : 1) : 1;
    const bd = Math.max(16, Math.min(40, Math.sqrt(bDepthVal) * 5));

    const tensorBlockX = mid - bw / 2;
    const tensorBlockY = 20 + bd * 0.4;

    const ropeGrp = inShape.length === 4 ? { outer: inShape[0], inner: inShape[1] } : null;
    drawDetailBlock3D(g, tensorBlockX, tensorBlockY, bw, bh, bd, input.color, input.label, ropeGrp);

    // Dim labels on the tensor
    g.append('text').attr('class', 'dim-label')
        .attr('x', tensorBlockX + bw / 2).attr('y', tensorBlockY + bh + 16)
        .attr('text-anchor', 'middle')
        .text(`${inNames[inNames.length - 1] || 'd_h'}=${inShape[inShape.length - 1]}`);
    g.append('text').attr('class', 'dim-label')
        .attr('x', tensorBlockX - 8).attr('y', tensorBlockY + bh / 2 + 4)
        .attr('text-anchor', 'end')
        .text(`${inNames[inNames.length - 2] || 'S'}=${inShape[inShape.length - 2]}`);
    if (inShape.length >= 3) {
        const depthStr = inShape.length === 4
            ? `${inNames[0] || 'B'}\u00d7${inNames[1] || 'n_h'}`
            : `${inNames[0] || 'B'}`;
        g.append('text').attr('class', 'dim-label')
            .attr('x', tensorBlockX + bw + bd * 0.7 / 2 + 8)
            .attr('y', tensorBlockY - bd * 0.4 / 2 - 4)
            .text(`${depthStr}=${bDepthVal}`);
    }

    // Highlight a single row (one token) on the front face
    const sliceY = tensorBlockY + bh * 0.4;
    const sliceH = Math.max(3, bh / inShape[inShape.length - 2]);
    g.append('rect')
        .attr('x', tensorBlockX).attr('y', sliceY)
        .attr('width', bw).attr('height', sliceH)
        .attr('fill', '#fff').attr('fill-opacity', 0.35)
        .attr('stroke', '#fff').attr('stroke-width', 1);

    // Arrow pointing down from the highlighted slice
    const arrowX = mid;
    const arrowStartY = tensorBlockY + bh + 22;
    const arrowEndY = arrowStartY + 24;
    g.append('line')
        .attr('x1', arrowX).attr('y1', arrowStartY)
        .attr('x2', arrowX).attr('y2', arrowEndY)
        .attr('stroke', '#888').attr('stroke-width', 1.5)
        .attr('marker-end', 'url(#arrowhead)');

    // Label the extraction
    g.append('text').attr('class', 'dim-label')
        .attr('x', arrowX + 10).attr('y', arrowStartY + 12)
        .attr('fill', '#aaa')
        .text('one token, one head');

    // "One token's head vector" label
    const vecLabelY = arrowEndY + 16;
    g.append('text').attr('class', 'dim-label')
        .attr('x', mid).attr('y', vecLabelY)
        .attr('text-anchor', 'middle')
        .attr('fill', '#aaa').attr('font-size', '12px')
        .text(`d_h = ${d_h} elements:`);

    const cellW = Math.min(28, (svgW - gPad * 2 - 20) / dispDims);
    const cellH = 26;
    const gridX = mid - (dispDims * cellW) / 2;
    const gridY = vecLabelY + 10;

    // Draw input vector cells with pair coloring
    const pairColors = ['#ff7043', '#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c', '#e67e22'];
    for (let i = 0; i < dispDims; i++) {
        const pairIdx = Math.floor(i / 2);
        const color = pairColors[pairIdx % pairColors.length];
        const cx = gridX + i * cellW;

        g.append('rect')
            .attr('x', cx).attr('y', gridY)
            .attr('width', cellW - 1).attr('height', cellH)
            .attr('rx', 2)
            .attr('fill', color).attr('fill-opacity', 0.5)
            .attr('stroke', color).attr('stroke-width', 1);

        g.append('text')
            .attr('x', cx + cellW / 2).attr('y', gridY + cellH / 2 + 4)
            .attr('text-anchor', 'middle')
            .attr('font-size', cellW >= 22 ? '10px' : '8px')
            .attr('fill', '#fff')
            .text(`x${i}`);
    }

    if (d_h > dispDims) {
        g.append('text').attr('class', 'dim-label')
            .attr('x', gridX + dispDims * cellW + 6)
            .attr('y', gridY + cellH / 2 + 4)
            .attr('fill', '#666')
            .text(`…${d_h - dispDims}`);
    }

    // Draw pair brackets underneath spanning both elements
    const bracketY = gridY + cellH + 6;
    for (let i = 0; i < dispPairs; i++) {
        const color = pairColors[i % pairColors.length];
        const x1 = gridX + i * 2 * cellW + 2;
        const x2 = gridX + (i * 2 + 2) * cellW - 3;
        const midBracketX = (x1 + x2) / 2;
        const bH = 10;

        // U-shaped bracket spanning both cells of the pair
        g.append('path')
            .attr('d', `M${x1},${bracketY} L${x1},${bracketY + bH} L${x2},${bracketY + bH} L${x2},${bracketY}`)
            .attr('fill', 'none')
            .attr('stroke', color).attr('stroke-width', 1.5);

        g.append('text')
            .attr('x', midBracketX).attr('y', bracketY + bH + 12)
            .attr('text-anchor', 'middle')
            .attr('font-size', '9px').attr('fill', color)
            .text(`θ${i}`);
    }

    // --- Part 2: Rotation formula for one pair ---
    let y = bracketY + 50;

    g.append('text').attr('class', 'tensor-label')
        .attr('x', mid).attr('y', y)
        .attr('font-size', '13px')
        .text('Rotation per pair (at position m):');
    y += 22;

    // Draw the 2D rotation formula — center it
    const formulaX = Math.max(20, mid - 120);

    // Output pair
    g.append('text')
        .attr('x', formulaX).attr('y', y + 12)
        .attr('fill', '#aaa').attr('font-size', '13px')
        .text('[');
    g.append('text').attr('x', formulaX + 10).attr('y', y + 5)
        .attr('font-size', '12px').attr('fill', '#ff7043').text('x₂ᵢ\u2032');
    g.append('text').attr('x', formulaX + 10).attr('y', y + 20)
        .attr('font-size', '12px').attr('fill', '#ff7043').text('x₂ᵢ₊₁\u2032');
    g.append('text')
        .attr('x', formulaX + 44).attr('y', y + 12)
        .attr('fill', '#aaa').attr('font-size', '13px')
        .text('] = [');

    // Rotation matrix
    const matX = formulaX + 72;
    g.append('text').attr('x', matX).attr('y', y + 5)
        .attr('font-size', '11px').attr('fill', '#888').text('cos mθᵢ');
    g.append('text').attr('x', matX + 62).attr('y', y + 5)
        .attr('font-size', '11px').attr('fill', '#888').text('-sin mθᵢ');
    g.append('text').attr('x', matX).attr('y', y + 20)
        .attr('font-size', '11px').attr('fill', '#888').text('sin mθᵢ');
    g.append('text').attr('x', matX + 66).attr('y', y + 20)
        .attr('font-size', '11px').attr('fill', '#888').text('cos mθᵢ');

    g.append('text')
        .attr('x', matX + 122).attr('y', y + 12)
        .attr('fill', '#aaa').attr('font-size', '13px')
        .text('][');

    // Input vector
    const vecX = matX + 134;
    g.append('text').attr('x', vecX).attr('y', y + 5)
        .attr('font-size', '12px').attr('fill', '#fff').text('x₂ᵢ');
    g.append('text').attr('x', vecX).attr('y', y + 20)
        .attr('font-size', '12px').attr('fill', '#fff').text('x₂ᵢ₊₁');
    g.append('text')
        .attr('x', vecX + 32).attr('y', y + 12)
        .attr('fill', '#aaa').attr('font-size', '13px')
        .text(']');

    y += 55;

    // --- Part 3: Rotation circle for a single pair ---
    g.append('text').attr('class', 'tensor-label')
        .attr('x', mid).attr('y', y)
        .attr('font-size', '13px')
        .text('Example: pair (x₀, x₁) at different positions');
    y += 30;

    const circR = 65;
    const circCx = mid;
    const circCy = y + circR + 4;

    // Circle
    g.append('circle')
        .attr('cx', circCx).attr('cy', circCy).attr('r', circR)
        .attr('fill', 'none')
        .attr('stroke', '#444').attr('stroke-width', 1.5);

    // Draw vectors at different positions
    const posAngles = [0, 0.4, 0.8, 1.2, 1.6];
    const posColors = ['#666', '#ff7043', '#e74c3c', '#f39c12', '#e67e22'];
    for (let p = 0; p < posAngles.length; p++) {
        const angle = posAngles[p];
        const endX = circCx + circR * 0.85 * Math.cos(angle);
        const endY = circCy - circR * 0.85 * Math.sin(angle);

        g.append('line')
            .attr('x1', circCx).attr('y1', circCy)
            .attr('x2', endX).attr('y2', endY)
            .attr('stroke', posColors[p]).attr('stroke-width', 2.5)
            .attr('stroke-linecap', 'round');

        // Position label at arrow tip
        const labelX = circCx + (circR + 16) * Math.cos(angle);
        const labelY = circCy - (circR + 16) * Math.sin(angle);
        g.append('text')
            .attr('x', labelX).attr('y', labelY + 3)
            .attr('text-anchor', 'middle')
            .attr('font-size', '10px').attr('fill', posColors[p])
            .text(p === 0 ? 'orig' : `m=${p}`);
    }

    // Arc showing θ₀ increment
    const arcR2 = circR * 0.4;
    g.append('text').attr('class', 'dim-label')
        .attr('x', circCx + arcR2 + 16).attr('y', circCy - 8)
        .attr('fill', '#ff7043').attr('font-size', '10px')
        .text('mθ₀');

    y = circCy + circR + 40;

    // --- Part 4: Frequency spectrum ---
    g.append('text').attr('class', 'tensor-label')
        .attr('x', mid).attr('y', y)
        .attr('font-size', '13px')
        .text('Rotation frequency per pair');
    y += 18;

    g.append('text').attr('class', 'dim-label')
        .attr('x', mid).attr('y', y)
        .attr('text-anchor', 'middle').attr('fill', '#888')
        .attr('font-size', '11px')
        .text('θᵢ = 10000⁻²ⁱ/ᵈ — low dims rotate fast, high dims slow');
    y += 18;

    const barW = Math.min(svgW - gPad * 2 - 20, 420);
    const barH = 28;
    const barX = mid - barW / 2;
    const freqPairs = Math.min(numPairs, 24);
    const segW = barW / freqPairs;

    for (let i = 0; i < freqPairs; i++) {
        const freq = Math.pow(10000, -2 * i / d_h);
        const intensity = Math.max(0.1, freq);
        const color = pairColors[i % pairColors.length];
        g.append('rect')
            .attr('x', barX + i * segW).attr('y', y)
            .attr('width', segW - 1).attr('height', barH)
            .attr('fill', color)
            .attr('fill-opacity', intensity * 0.8)
            .attr('rx', 1);
    }

    g.append('text').attr('class', 'dim-label')
        .attr('x', barX).attr('y', y + barH + 14)
        .attr('fill', '#aaa').attr('font-size', '10px')
        .text('pair 0: fast (local pos)');
    g.append('text').attr('class', 'dim-label')
        .attr('x', barX + barW).attr('y', y + barH + 14)
        .attr('text-anchor', 'end').attr('fill', '#aaa').attr('font-size', '10px')
        .text(`pair ${numPairs-1}: slow (global pos)`);
    y += barH + 30;

    // Key property
    g.append('text').attr('class', 'dim-label')
        .attr('x', mid).attr('y', y)
        .attr('text-anchor', 'middle').attr('fill', '#7c8cf8')
        .attr('font-size', '11px')
        .text('Key property: \u27e8RoPE(q, m), RoPE(k, n)\u27e9 = f(q, k, m\u2212n)');
    y += 18;
    g.append('text').attr('class', 'dim-label')
        .attr('x', mid).attr('y', y)
        .attr('text-anchor', 'middle').attr('fill', '#666')
        .attr('font-size', '11px')
        .text('Dot product depends only on relative position m\u2212n');
    y += 20;

    svg.attr('height', y + 10);
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

function drawDetailBlock3D(g, x, y, w, h, d, color, label, grouping) {
    const dx = d * 0.7;
    const dy = -d * 0.4;

    g.append('polygon')
        .attr('points', `${x},${y} ${x+dx},${y+dy} ${x+w+dx},${y+dy} ${x+w},${y}`)
        .attr('fill', d3.color(color).darker(0.4)).attr('stroke', 'none');
    g.append('polygon')
        .attr('points', `${x+w},${y} ${x+w+dx},${y+dy} ${x+w+dx},${y+h+dy} ${x+w},${y+h}`)
        .attr('fill', d3.color(color).darker(0.8)).attr('stroke', 'none');

    // 4D depth grouping lines
    if (grouping) {
        const { outer, inner } = grouping;
        const total = outer * inner;
        const showInnerLines = total <= 16;
        for (let i = 1; i < total; i++) {
            const isBatch = (i % inner === 0);
            if (!showInnerLines && !isBatch) continue;
            const frac = i / total;
            const lx = dx * frac;
            const ly = dy * frac;
            const opacity = isBatch ? 0.6 : 0.25;
            const strokeW = isBatch ? 1.5 : 0.75;
            // Top face line
            g.append('line')
                .attr('x1', x + lx).attr('y1', y + ly)
                .attr('x2', x + w + lx).attr('y2', y + ly)
                .attr('stroke', '#fff').attr('stroke-opacity', opacity)
                .attr('stroke-width', strokeW);
            // Right face line
            g.append('line')
                .attr('x1', x + w + lx).attr('y1', y + ly)
                .attr('x2', x + w + lx).attr('y2', y + h + ly)
                .attr('stroke', '#fff').attr('stroke-opacity', opacity)
                .attr('stroke-width', strokeW);
        }
    }

    g.append('rect')
        .attr('x', x).attr('y', y).attr('width', w).attr('height', h)
        .attr('fill', color).attr('fill-opacity', 0.85)
        .attr('stroke', d3.color(color).darker(0.3)).attr('stroke-width', 1);
    g.append('text')
        .attr('class', 'tensor-label')
        .attr('x', x + w / 2).attr('y', y + h / 2 + 4)
        .text(label);
}
