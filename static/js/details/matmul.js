// matmul.js — Matmul (L-shaped diagram) and All-Reduce detail visualizations
import { detailMetrics, drawDetailBlock, drawDetailBlock3D, TP_COLORS } from './shared.js';

export function drawMatmulDetail(svg, op, tensorMap) {
    const { w: svgW } = detailMetrics();
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
    let bTransposed = false;
    if (B_tensor) {
        const bShape = B_tensor.shape;
        const bLast = bShape[bShape.length - 1];
        const bSecondLast = bShape.length >= 2 ? bShape[bShape.length - 2] : bLast;
        bTransposed = bLast === inner && shC[shC.length - 1] === bSecondLast && bLast !== bSecondLast;
        cols_b = bTransposed ? bSecondLast : bLast;
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
        // When B is transposed (e.g. K^T), swap col/row dim names and values
        const bDimNames = B_tensor.dimNames || [];
        const bColName = bTransposed ? bDimNames[bDimNames.length - 2] || '' : bDimNames[bDimNames.length - 1] || '';
        const bRowName = bTransposed ? bDimNames[bDimNames.length - 1] || '' : bDimNames[bDimNames.length - 2] || '';
        const bRowVal = bTransposed ? shB[shB.length - 1] : shB[shB.length - 2];
        g.append('text').attr('class', 'dim-label')
            .attr('x', originX + wC / 2).attr('y', originY - hB - pad - 6)
            .attr('text-anchor', 'middle').text(bColName ? bColName + '=' + cols_b : cols_b);
        g.append('text').attr('class', 'dim-label')
            .attr('x', originX - 8).attr('y', originY - hB - pad + hB / 2 + 3)
            .attr('text-anchor', 'end').text(bRowName ? bRowName + '=' + bRowVal : inner);
    }

    // Result C (center)
    drawDetailBlock(g, originX, originY, wC, hC, C.color, C.label);
    const resultColName = B_tensor && B_tensor.dimNames
        ? (bTransposed ? B_tensor.dimNames[B_tensor.dimNames.length - 2] : B_tensor.dimNames[B_tensor.dimNames.length - 1])
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
        .text('\u2190 dot product');

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
        noteY = drawAllReduceSection(g, originX - 20, noteY, op.tpSize, C, svgW);
    }

    svg.attr('height', noteY + 20);
}

function drawAllReduceSection(g, x, y, tpSize, outputTensor, svgW) {
    const rankH = 22;
    const rankW = 80;
    const rankGap = 4;
    const displayRanks = Math.min(tpSize, 8);
    const outW = 70;
    const sumBoxW = 28;
    const gap1 = 30; // ranks to sum
    const gap2 = 16; // sum to output

    // Total width of the diagram: ranks + gap + sum + gap + output
    const totalW = rankW + gap1 + sumBoxW + gap2 + outW;
    const rankX = svgW ? (svgW - totalW) / 2 : x;

    g.append('text').attr('class', 'tensor-label')
        .attr('x', rankX + totalW / 2).attr('y', y)
        .text('All-Reduce (sum across TP ranks)');

    const startY = y + 14;

    // Build per-rank shape annotation
    const shape = outputTensor.shape;
    const dimNames = outputTensor.dimNames || [];
    let rankAnnotation = '';
    if (shape && shape.length >= 2) {
        const parts = [];
        for (let di = 0; di < shape.length; di++) {
            const dn = dimNames[di] || '';
            const val = (di === shape.length - 1) ? Math.round(shape[di] / tpSize) : shape[di];
            parts.push(dn ? `${dn}=${val}` : `${val}`);
        }
        rankAnnotation = `[${parts.join(', ')}]`;
    }

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

    // Per-rank dimension annotation (to the left of rank blocks)
    if (rankAnnotation) {
        const midRankY = startY + (displayRanks * (rankH + rankGap) - rankGap) / 2;
        g.append('text').attr('class', 'dim-label')
            .attr('x', rankX - 6).attr('y', midRankY + 4)
            .attr('text-anchor', 'end').attr('font-size', '8px').attr('fill', '#aaa')
            .text(rankAnnotation);
    }

    // Sum symbol
    const totalH = displayRanks * (rankH + rankGap) - rankGap;
    const sumX = rankX + rankW + gap1;
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
        .attr('width', sumBoxW).attr('height', 28)
        .attr('fill', '#1e2030')
        .attr('stroke', '#3498db')
        .attr('stroke-width', 2)
        .attr('rx', 4);
    g.append('text')
        .attr('x', sumX + sumBoxW / 2).attr('y', sumY + 19)
        .attr('text-anchor', 'middle')
        .attr('fill', '#3498db')
        .attr('font-size', '16px')
        .attr('font-weight', '600')
        .text('\u03a3');

    // Arrow from sum to output
    const outX = sumX + sumBoxW + gap2;
    g.append('line')
        .attr('x1', sumX + sumBoxW + 2).attr('y1', sumY + 14)
        .attr('x2', outX - 2).attr('y2', sumY + 14)
        .attr('stroke', '#555').attr('stroke-width', 1.5);
    // Arrowhead
    g.append('path')
        .attr('d', `M${outX - 2},${sumY + 14} l-5,-3.5 l0,7 z`)
        .attr('fill', '#555');

    // Output block
    const outBlockY = sumY - 2;
    const outBlockH = 32;
    g.append('rect')
        .attr('x', outX).attr('y', outBlockY)
        .attr('width', outW).attr('height', outBlockH)
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

    // Dimension annotations on the output block
    if (shape && shape.length >= 2) {
        const lastDim = shape[shape.length - 1];
        const lastDimName = dimNames[dimNames.length - 1] || '';
        const secDim = shape[shape.length - 2];
        const secDimName = dimNames.length >= 2 ? dimNames[dimNames.length - 2] : '';

        // Bottom label (columns / last dim)
        const colLabel = lastDimName ? `${lastDimName}=${lastDim}` : `${lastDim}`;
        g.append('text').attr('class', 'dim-label')
            .attr('x', outX + outW / 2).attr('y', outBlockY + outBlockH + 12)
            .attr('text-anchor', 'middle').attr('font-size', '8px').attr('fill', '#aaa')
            .text(colLabel);

        // Right label (rows / second-to-last dim) — placed on right to avoid overlap with Σ
        const rowLabel = secDimName ? `${secDimName}=${secDim}` : `${secDim}`;
        g.append('text').attr('class', 'dim-label')
            .attr('x', outX + outW + 6).attr('y', outBlockY + outBlockH / 2 + 3)
            .attr('font-size', '8px').attr('fill', '#aaa')
            .text(rowLabel);

        // Batch dims (if 3D or 4D)
        if (shape.length >= 3) {
            const batchParts = [];
            for (let di = 0; di < shape.length - 2; di++) {
                const dn = dimNames[di] || '';
                batchParts.push(dn ? `${dn}=${shape[di]}` : `${shape[di]}`);
            }
            g.append('text').attr('class', 'dim-label')
                .attr('x', outX + outW / 2).attr('y', outBlockY - 6)
                .attr('text-anchor', 'middle').attr('font-size', '8px').attr('fill', '#aaa')
                .text(`[${batchParts.join(', ')}]`);
        }
    }

    return startY + totalH + 20;
}
