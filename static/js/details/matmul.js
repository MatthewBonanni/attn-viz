// matmul.js — Matmul (L-shaped diagram) and All-Reduce detail visualizations
import { detailMetrics, drawDetailBlock, drawDetailBlock3D, TP_COLORS, RANK_COLORS } from './shared.js';

export function drawMatmulDetail(svg, op, tensorMap, params) {
    const { w: svgW } = detailMetrics();
    const A = tensorMap[op.inputs[0]];
    const B_tensor = op.inputs.length > 1 ? tensorMap[op.inputs[1]] : null;
    const C = tensorMap[op.output];
    if (!A || !C) return;

    const shA = A.shape;
    const shC = C.shape;
    const rows_a = shA.length >= 2 ? shA[shA.length - 2] : shA[0];
    const inner = shA[shA.length - 1];
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
    const allDims = [rows_a, inner, cols_b];
    const maxSqrt = Math.max(...allDims.map(v => Math.sqrt(v)));
    const scaleFactor = maxSqrt > 0 ? maxDim / maxSqrt : 12;
    const scale = (v) => Math.max(24, Math.min(maxDim, Math.sqrt(v) * scaleFactor));

    const wA = scale(inner);
    const hA = scale(rows_a);
    const wC = scale(cols_b);
    const hC = hA;
    const hB = scale(inner);

    const pad = 30;
    const totalW = wA + pad + wC + 40;
    const leftMargin = Math.max(10, (svgW - totalW) / 2);
    const originX = leftMargin + wA + pad;
    const originY = hB + pad + 10;

    const g = svg.append('g').attr('transform', 'translate(0, 5)');

    // --- DP state ---
    const dpSize = (params && params.dp_size) || 1;
    const tpSize = (params && params.tp_size) || 1;
    const B_count = (params && params.B) || 1;
    const seqDims = new Set(['S_q', '\u03a3S_q', 'S', '\u03a3S']);
    const uniformFracs = (n) => Array.from({length: n + 1}, (_, i) => i / n);

    const aDimNames = A.dimNames || [];
    const cDimNames = C.dimNames || [];
    const aRowDim = aDimNames[aDimNames.length - 2];
    const cColDim = cDimNames[cDimNames.length - 1];
    const aRowIsDp = A.dpSharded && dpSize > 1 && seqDims.has(aRowDim);
    const cColIsDp = dpSize > 1 && seqDims.has(cColDim);
    const isBlockDiag = aRowIsDp && cColIsDp;
    const hasDp = dpSize > 1 && (aRowIsDp || cColIsDp);

    const rowFracs = (aRowIsDp && A.dpBoundaryFracs && A.dpBoundaryFracs.length === dpSize + 1)
        ? A.dpBoundaryFracs : uniformFracs(dpSize);
    let colFracs = null;
    if (isBlockDiag && B_tensor) {
        colFracs = (B_tensor.dpBoundaryFracs && B_tensor.dpBoundaryFracs.length === dpSize + 1)
            ? B_tensor.dpBoundaryFracs : uniformFracs(dpSize);
    }

    // --- Draw base blocks ---

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

    // --- TP stripes on 2D weight (B_tensor) ---
    if (B_tensor && B_tensor.tpSharded && B_tensor.tpSize > 1 && B_tensor.tpDim != null) {
        const bTpSize = B_tensor.tpSize;
        const bTpDim = B_tensor.tpDim;
        const bX = originX, bY = originY - hB - pad;
        // tpDim=1 means column-parallel: shard cols (visual width after transpose consideration)
        // tpDim=0 means row-parallel: shard rows
        // In the matmul diagram, B is displayed as [inner × cols_b] (possibly transposed)
        // tpDim refers to the original tensor dimension
        const shardVisualCols = bTransposed ? (bTpDim === 0) : (bTpDim === 1);

        for (let tp = 0; tp < bTpSize; tp++) {
            const tpColor = TP_COLORS[tp % TP_COLORS.length];
            if (shardVisualCols) {
                const sx = bX + (tp / bTpSize) * wC;
                const sw = wC / bTpSize;
                g.append('rect')
                    .attr('x', sx).attr('y', bY).attr('width', sw).attr('height', hB)
                    .attr('fill', tpColor).attr('fill-opacity', 0.3).attr('stroke', 'none');
            } else {
                const sy = bY + (tp / bTpSize) * hB;
                const sh = hB / bTpSize;
                g.append('rect')
                    .attr('x', bX).attr('y', sy).attr('width', wC).attr('height', sh)
                    .attr('fill', tpColor).attr('fill-opacity', 0.3).attr('stroke', 'none');
            }
        }
        for (let tp = 1; tp < bTpSize; tp++) {
            if (shardVisualCols) {
                const lx = bX + (tp / bTpSize) * wC;
                g.append('line')
                    .attr('x1', lx).attr('y1', bY).attr('x2', lx).attr('y2', bY + hB)
                    .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
            } else {
                const ly = bY + (tp / bTpSize) * hB;
                g.append('line')
                    .attr('x1', bX).attr('y1', ly).attr('x2', bX + wC).attr('y2', ly)
                    .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
            }
        }
    }

    // --- TP stripes on 2D activation A (e.g. ctx before row-parallel W_O) ---
    if (A.tpSharded && A.tpSize > 1 && A.tpDim != null && A.type !== 'weight') {
        const aTpSize = A.tpSize;
        const aX = originX - wA - pad, aY = originY;
        // tpDim=1 means last dim sharded → visual width for A
        for (let tp = 0; tp < aTpSize; tp++) {
            const tpColor = TP_COLORS[tp % TP_COLORS.length];
            if (A.tpDim === 1) {
                const sx = aX + (tp / aTpSize) * wA;
                const sw = wA / aTpSize;
                g.append('rect')
                    .attr('x', sx).attr('y', aY).attr('width', sw).attr('height', hA)
                    .attr('fill', tpColor).attr('fill-opacity', 0.3).attr('stroke', 'none');
            } else {
                const sy = aY + (tp / aTpSize) * hA;
                const sh = hA / aTpSize;
                g.append('rect')
                    .attr('x', aX).attr('y', sy).attr('width', wA).attr('height', sh)
                    .attr('fill', tpColor).attr('fill-opacity', 0.3).attr('stroke', 'none');
            }
        }
        for (let tp = 1; tp < aTpSize; tp++) {
            if (A.tpDim === 1) {
                const lx = aX + (tp / aTpSize) * wA;
                g.append('line')
                    .attr('x1', lx).attr('y1', aY).attr('x2', lx).attr('y2', aY + hA)
                    .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
            } else {
                const ly = aY + (tp / aTpSize) * hA;
                g.append('line')
                    .attr('x1', aX).attr('y1', ly).attr('x2', aX + wA).attr('y2', ly)
                    .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
            }
        }
    }

    // --- TP stripes on 2D result C (e.g. Q_flat output of column-parallel) ---
    if (C.tpSharded && C.tpSize > 1 && C.tpDim != null && C.shape.length === 2) {
        const cTpSize = C.tpSize;
        for (let tp = 0; tp < cTpSize; tp++) {
            const tpColor = TP_COLORS[tp % TP_COLORS.length];
            if (C.tpDim === 1) {
                const sx = originX + (tp / cTpSize) * wC;
                const sw = wC / cTpSize;
                g.append('rect')
                    .attr('x', sx).attr('y', originY).attr('width', sw).attr('height', hC)
                    .attr('fill', tpColor).attr('fill-opacity', 0.3).attr('stroke', 'none');
            } else {
                const sy = originY + (tp / cTpSize) * hC;
                const sh = hC / cTpSize;
                g.append('rect')
                    .attr('x', originX).attr('y', sy).attr('width', wC).attr('height', sh)
                    .attr('fill', tpColor).attr('fill-opacity', 0.3).attr('stroke', 'none');
            }
        }
        for (let tp = 1; tp < cTpSize; tp++) {
            if (C.tpDim === 1) {
                const lx = originX + (tp / cTpSize) * wC;
                g.append('line')
                    .attr('x1', lx).attr('y1', originY).attr('x2', lx).attr('y2', originY + hC)
                    .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
            } else {
                const ly = originY + (tp / cTpSize) * hC;
                g.append('line')
                    .attr('x1', originX).attr('y1', ly).attr('x2', originX + wC).attr('y2', ly)
                    .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
            }
        }
    }

    // --- DP overlays on A, B, C ---
    let cellInfoG = null;
    let cellInfoY = 0;

    function showCellInfo(dp, isSkipped) {
        if (cellInfoG) cellInfoG.remove();
        cellInfoG = g.append('g');
        if (isSkipped) {
            cellInfoG.append('text').attr('class', 'dim-label')
                .attr('x', svgW / 2).attr('y', cellInfoY)
                .attr('text-anchor', 'middle').attr('fill', '#888').attr('font-size', '11px')
                .text('Skipped \u2014 sequences don\'t attend across DP boundaries');
            return;
        }
        const cellColor = RANK_COLORS[dp % RANK_COLORS.length];
        cellInfoG.append('circle')
            .attr('cx', svgW / 2 - 120).attr('cy', cellInfoY - 3)
            .attr('r', 6).attr('fill', cellColor).attr('fill-opacity', 0.8);
        const parts = [];
        if (B_count > 1) {
            const reqs = [];
            for (let r = 0; r < B_count; r++) {
                if (Math.floor(r * dpSize / B_count) === dp) reqs.push(r);
            }
            parts.push(`Req ${reqs.join(', ')}`);
        }
        parts.push(`DP Rank ${dp}`);
        cellInfoG.append('text').attr('class', 'dim-label')
            .attr('x', svgW / 2 - 108).attr('y', cellInfoY)
            .attr('text-anchor', 'start').attr('fill', cellColor).attr('font-size', '11px')
            .attr('font-weight', '600')
            .text(parts.join('  \u00b7  '));
    }

    if (hasDp) {
        // DP stripes on A (rows)
        if (aRowIsDp) {
            for (let dp = 0; dp < dpSize; dp++) {
                const rankColor = RANK_COLORS[dp % RANK_COLORS.length];
                const bandY = originY + rowFracs[dp] * hA;
                const bandH = (rowFracs[dp + 1] - rowFracs[dp]) * hA;
                g.append('rect')
                    .attr('x', originX - wA - pad).attr('y', bandY)
                    .attr('width', wA).attr('height', bandH)
                    .attr('fill', rankColor).attr('fill-opacity', 0.25)
                    .attr('stroke', 'none');
            }
            for (let dp = 1; dp < dpSize; dp++) {
                const ly = originY + rowFracs[dp] * hA;
                g.append('line')
                    .attr('x1', originX - wA - pad).attr('y1', ly)
                    .attr('x2', originX - pad).attr('y2', ly)
                    .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
            }
        }

        // DP stripes on B
        if (B_tensor && B_tensor.dpSharded && dpSize > 1) {
            const bDimNames = B_tensor.dimNames || [];
            const bDpIdx = bDimNames.lastIndexOf(B_tensor.dpSharded);
            const bDpIsSecondLast = bDpIdx === bDimNames.length - 2;
            const bStripesVertical = bDpIsSecondLast ? bTransposed : !bTransposed;
            const bFracs = (B_tensor.dpBoundaryFracs && B_tensor.dpBoundaryFracs.length === dpSize + 1)
                ? B_tensor.dpBoundaryFracs : uniformFracs(dpSize);

            for (let dp = 0; dp < dpSize; dp++) {
                const rankColor = RANK_COLORS[dp % RANK_COLORS.length];
                if (bStripesVertical) {
                    const bandX = originX + bFracs[dp] * wC;
                    const bandW = (bFracs[dp + 1] - bFracs[dp]) * wC;
                    g.append('rect')
                        .attr('x', bandX).attr('y', originY - hB - pad)
                        .attr('width', bandW).attr('height', hB)
                        .attr('fill', rankColor).attr('fill-opacity', 0.25)
                        .attr('stroke', 'none');
                } else {
                    const bandY = originY - hB - pad + bFracs[dp] * hB;
                    const bandH = (bFracs[dp + 1] - bFracs[dp]) * hB;
                    g.append('rect')
                        .attr('x', originX).attr('y', bandY)
                        .attr('width', wC).attr('height', bandH)
                        .attr('fill', rankColor).attr('fill-opacity', 0.25)
                        .attr('stroke', 'none');
                }
            }
            for (let dp = 1; dp < dpSize; dp++) {
                if (bStripesVertical) {
                    const lx = originX + bFracs[dp] * wC;
                    g.append('line')
                        .attr('x1', lx).attr('y1', originY - hB - pad)
                        .attr('x2', lx).attr('y2', originY - pad)
                        .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
                } else {
                    const ly = originY - hB - pad + bFracs[dp] * hB;
                    g.append('line')
                        .attr('x1', originX).attr('y1', ly)
                        .attr('x2', originX + wC).attr('y2', ly)
                        .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
                }
            }
        }

        // DP on result C
        if (isBlockDiag && colFracs) {
            // Block-diagonal grid
            for (let dr = 0; dr < dpSize; dr++) {
                for (let dc = 0; dc < dpSize; dc++) {
                    const cellX = originX + colFracs[dc] * wC;
                    const cellW = (colFracs[dc + 1] - colFracs[dc]) * wC;
                    const cellY = originY + rowFracs[dr] * hC;
                    const cellH = (rowFracs[dr + 1] - rowFracs[dr]) * hC;

                    if (dr === dc) {
                        const rankColor = RANK_COLORS[dr % RANK_COLORS.length];
                        const cell = g.append('rect')
                            .attr('x', cellX).attr('y', cellY)
                            .attr('width', cellW).attr('height', cellH)
                            .attr('fill', rankColor).attr('fill-opacity', 0.35)
                            .attr('stroke', rankColor).attr('stroke-width', 1)
                            .attr('stroke-opacity', 0.6)
                            .style('cursor', 'pointer');

                        cell.on('mouseenter', function() {
                            d3.select(this).attr('fill-opacity', 0.6).attr('stroke-width', 2);
                        });
                        cell.on('mouseleave', function() {
                            d3.select(this).attr('fill-opacity', 0.35).attr('stroke-width', 1);
                        });
                        cell.on('click', (event) => {
                            event.stopPropagation();
                            showCellInfo(dr, false);
                        });
                    } else {
                        const cell = g.append('rect')
                            .attr('x', cellX).attr('y', cellY)
                            .attr('width', cellW).attr('height', cellH)
                            .attr('fill', '#000').attr('fill-opacity', 0.35)
                            .attr('stroke', '#333').attr('stroke-width', 0.5)
                            .style('cursor', 'pointer');

                        g.append('line')
                            .attr('x1', cellX + 2).attr('y1', cellY + 2)
                            .attr('x2', cellX + cellW - 2).attr('y2', cellY + cellH - 2)
                            .attr('stroke', '#444').attr('stroke-width', 0.75)
                            .attr('pointer-events', 'none');
                        g.append('line')
                            .attr('x1', cellX + cellW - 2).attr('y1', cellY + 2)
                            .attr('x2', cellX + 2).attr('y2', cellY + cellH - 2)
                            .attr('stroke', '#444').attr('stroke-width', 0.75)
                            .attr('pointer-events', 'none');

                        cell.on('click', (event) => {
                            event.stopPropagation();
                            showCellInfo(null, true);
                        });
                    }
                }
            }
        } else if (aRowIsDp) {
            // Horizontal stripes on C
            for (let dp = 0; dp < dpSize; dp++) {
                const rankColor = RANK_COLORS[dp % RANK_COLORS.length];
                const bandY = originY + rowFracs[dp] * hC;
                const bandH = (rowFracs[dp + 1] - rowFracs[dp]) * hC;
                const cell = g.append('rect')
                    .attr('x', originX).attr('y', bandY)
                    .attr('width', wC).attr('height', bandH)
                    .attr('fill', rankColor).attr('fill-opacity', 0.3)
                    .attr('stroke', 'none')
                    .style('cursor', 'pointer');

                cell.on('mouseenter', function() {
                    d3.select(this).attr('fill-opacity', 0.55).attr('stroke', '#fff').attr('stroke-width', 1.5);
                });
                cell.on('mouseleave', function() {
                    d3.select(this).attr('fill-opacity', 0.3).attr('stroke', 'none');
                });
                cell.on('click', (event) => {
                    event.stopPropagation();
                    showCellInfo(dp, false);
                });
            }
            for (let dp = 1; dp < dpSize; dp++) {
                const ly = originY + rowFracs[dp] * hC;
                g.append('line')
                    .attr('x1', originX).attr('y1', ly)
                    .attr('x2', originX + wC).attr('y2', ly)
                    .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
            }
        }
    }

    // --- Highlight row/col (on top of DP overlays) ---
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

    // --- DP info section ---
    if (hasDp) {
        noteY += 4;
        if (isBlockDiag) {
            g.append('text').attr('class', 'dim-label')
                .attr('x', svgW / 2).attr('y', noteY)
                .attr('text-anchor', 'middle').attr('fill', '#e67e22').attr('font-size', '10px')
                .text('Block-diagonal: cross-rank computation skipped');
            noteY += 14;
            g.append('text').attr('class', 'dim-label')
                .attr('x', svgW / 2).attr('y', noteY)
                .attr('text-anchor', 'middle').attr('fill', '#888').attr('font-size', '9px')
                .text('(sequences don\'t attend across DP boundaries)');
            noteY += 18;
        } else if (aRowIsDp && B_tensor && B_tensor.dpSharded) {
            g.append('text').attr('class', 'dim-label')
                .attr('x', svgW / 2).attr('y', noteY)
                .attr('text-anchor', 'middle').attr('fill', '#e67e22').attr('font-size', '10px')
                .text('Each DP rank computes independently on its portion');
            noteY += 18;
        } else if (aRowIsDp) {
            g.append('text').attr('class', 'dim-label')
                .attr('x', svgW / 2).attr('y', noteY)
                .attr('text-anchor', 'middle').attr('fill', '#e67e22').attr('font-size', '10px')
                .text('Each DP rank processes its batch independently (weight shared)');
            noteY += 18;
        }

        // Legend + click hint
        g.append('text').attr('class', 'dim-label')
            .attr('x', svgW / 2).attr('y', noteY)
            .attr('text-anchor', 'middle').attr('fill', '#666').attr('font-size', '9px')
            .text('Click a colored region for details');
        noteY += 14;

        const legendCols = Math.min(dpSize, 4);
        const legendRows = Math.ceil(dpSize / legendCols);
        const colWLeg = svgW / legendCols;

        for (let ri = 0; ri < dpSize; ri++) {
            const col = ri % legendCols;
            const row = Math.floor(ri / legendCols);
            const rankColor = RANK_COLORS[ri % RANK_COLORS.length];
            const lx = col * colWLeg + 12;
            const ly = noteY + row * 18;

            const legendItem = g.append('g').style('cursor', 'pointer');
            legendItem.append('circle')
                .attr('cx', lx + 4).attr('cy', ly - 3)
                .attr('r', 5).attr('fill', rankColor).attr('fill-opacity', 0.7);

            let label = `DP Rank ${ri}`;
            if (B_count > 1) {
                const reqs = [];
                for (let r = 0; r < B_count; r++) {
                    if (Math.floor(r * dpSize / B_count) === ri) reqs.push(r);
                }
                label += ` (Req ${reqs.join(', ')})`;
            }

            legendItem.append('text').attr('class', 'dim-label')
                .attr('x', lx + 12).attr('y', ly)
                .attr('text-anchor', 'start').attr('fill', rankColor).attr('font-size', '10px')
                .text(label);

            legendItem.on('click', (event) => {
                event.stopPropagation();
                showCellInfo(ri, false);
            });
        }
        noteY += legendRows * 18 + 8;

        cellInfoY = noteY;
        noteY += 24;
    }

    // --- TP info on weight ---
    if (B_tensor && B_tensor.tpSharded && B_tensor.tpSize > 1 && B_tensor.tpDim != null) {
        const bTpSize = B_tensor.tpSize;
        const parallel = B_tensor.tpDim === 1 ? 'Column-parallel' : 'Row-parallel';
        const bDimNames = B_tensor.dimNames || [];
        const dimLabel = B_tensor.tpDim === 0 ? (bDimNames[0] || 'rows') : (bDimNames[1] || 'cols');
        const perRank = Math.round(B_tensor.shape[B_tensor.tpDim] / bTpSize);

        g.append('text').attr('class', 'dim-label')
            .attr('x', svgW / 2).attr('y', noteY)
            .attr('text-anchor', 'middle').attr('fill', '#7c8cf8').attr('font-size', '10px')
            .text(`${parallel} (${B_tensor.label}): each TP rank holds ${dimLabel}=${perRank}`);
        noteY += 16;

        const tpLegCols = Math.min(bTpSize, 4);
        const tpLegColW = svgW / tpLegCols;
        for (let tp = 0; tp < bTpSize; tp++) {
            const tpColor = TP_COLORS[tp % TP_COLORS.length];
            const col = tp % tpLegCols;
            const row = Math.floor(tp / tpLegCols);
            const lx = col * tpLegColW + 12;
            const ly = noteY + row * 16;
            g.append('circle')
                .attr('cx', lx + 4).attr('cy', ly - 3)
                .attr('r', 4).attr('fill', tpColor).attr('fill-opacity', 0.7);
            g.append('text').attr('class', 'dim-label')
                .attr('x', lx + 12).attr('y', ly)
                .attr('text-anchor', 'start').attr('fill', tpColor).attr('font-size', '9px')
                .text(`TP${tp}: ${dimLabel} [${tp * perRank}\u2013${(tp + 1) * perRank - 1}]`);
        }
        const tpLegRows = Math.ceil(bTpSize / tpLegCols);
        noteY += tpLegRows * 16 + 8;
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
            .text('Reshape / View');
        noteY += 24;

        const reshMaxPx = 120;
        const reshFaceDims = [
            matmulResultShape[matmulResultShape.length - 1],
            matmulResultShape[matmulResultShape.length - 2],
            shC[shC.length - 1], shC[shC.length - 2],
        ];
        const reshMaxSqrt = Math.max(...reshFaceDims.map(v => Math.sqrt(v)));
        const reshFactor = reshMaxSqrt > 0 ? reshMaxPx / reshMaxSqrt : 8;
        const reshScale = (v) => Math.max(30, Math.min(reshMaxPx, Math.sqrt(v) * reshFactor));

        const beforeW = reshScale(matmulResultShape[matmulResultShape.length - 1]);
        const beforeH = reshScale(matmulResultShape[matmulResultShape.length - 2]);
        const beforeDepthVal = matmulResultShape.length >= 3
            ? matmulResultShape.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
        const beforeD = Math.max(4, Math.min(100, Math.sqrt(beforeDepthVal) * 8));

        const afterW = reshScale(shC[shC.length - 1]);
        const afterH = reshScale(shC[shC.length - 2]);
        const afterDepthVal = shC.length >= 3
            ? shC.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
        const afterD = Math.max(4, Math.min(100, Math.sqrt(afterDepthVal) * 8));

        const arrowGap = 100;
        const beforeTotalW = beforeW + beforeD * 0.7;
        const afterTotalW = afterW + afterD * 0.7;
        const totalReshW = beforeTotalW + arrowGap + afterTotalW;
        const reshLeftMargin = Math.max(40, (svgW - totalReshW) / 2);

        const beforeX = reshLeftMargin;
        const beforeTopPad = Math.max(beforeD, 12) * 0.4;
        const afterTopPad = Math.max(afterD, 12) * 0.4;
        const maxTopPad = Math.max(beforeTopPad, afterTopPad);

        // DP/TP info for reshape blocks
        const cTpInfo = (C.tpSharded && tpSize > 1) ? { tpSize } : null;
        const cDpInfo = (C.dpSharded && dpSize > 1) ? { dpSize, dpFracs: C.dpBoundaryFracs } : null;

        const beforeGrp = matmulResultShape.length === 4
            ? { outer: matmulResultShape[0], inner: matmulResultShape[1] } : null;
        drawDetailBlock3D(g, beforeX, noteY + maxTopPad, beforeW, beforeH, beforeD, C.color, '',
            beforeGrp, cTpInfo, cDpInfo);

        const aNames = A.dimNames || [];
        const bNames = B_tensor && B_tensor.dimNames ? B_tensor.dimNames : [];
        const beforeNames = [...aNames.slice(0, -1), bNames[bNames.length - 1] || aNames[aNames.length - 1] || ''];
        const lastBeforeDim = beforeNames[beforeNames.length - 1] || '';
        const secLastBeforeDim = beforeNames[beforeNames.length - 2] || '';
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

        const afterX = reshLeftMargin + beforeTotalW + arrowGap;
        const afterGrp = shC.length === 4 ? { outer: shC[0], inner: shC[1] } : null;
        drawDetailBlock3D(g, afterX, noteY + maxTopPad, afterW, afterH, afterD, C.color, C.label,
            afterGrp, cTpInfo, cDpInfo);

        const outNames = C.dimNames || [];
        const lastOutDim = outNames[outNames.length - 1] || '';
        const secLastOutDim = outNames[outNames.length - 2] || '';
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

        const maxBlockH = maxTopPad + Math.max(beforeH, afterH);
        noteY += maxBlockH + 36;

        if (matmulResultShape.length === 4 && shC.length === 3) {
            const n_h_val = matmulResultShape[1];
            const d_h_val = matmulResultShape[3];
            const D_val = shC[shC.length - 1];
            g.append('text').attr('class', 'dim-label')
                .attr('x', svgW / 2).attr('y', noteY)
                .attr('text-anchor', 'middle').attr('fill', '#7c8cf8').attr('font-size', '11px')
                .text(`View ${n_h_val} heads: n_h \u00d7 d_h = ${n_h_val} \u00d7 ${d_h_val} = ${D_val} = D`);
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
    const gap1 = 30;
    const gap2 = 16;

    const totalW = rankW + gap1 + sumBoxW + gap2 + outW;
    const rankX = svgW ? (svgW - totalW) / 2 : x;

    g.append('text').attr('class', 'tensor-label')
        .attr('x', rankX + totalW / 2).attr('y', y)
        .text('All-Reduce (sum across TP ranks)');

    const startY = y + 14;

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

    if (rankAnnotation) {
        const midRankY = startY + (displayRanks * (rankH + rankGap) - rankGap) / 2;
        g.append('text').attr('class', 'dim-label')
            .attr('x', rankX - 6).attr('y', midRankY + 4)
            .attr('text-anchor', 'end').attr('font-size', '8px').attr('fill', '#aaa')
            .text(rankAnnotation);
    }

    const totalH = displayRanks * (rankH + rankGap) - rankGap;
    const sumX = rankX + rankW + gap1;
    const sumY = startY + totalH / 2 - 14;

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

    const outX = sumX + sumBoxW + gap2;
    g.append('line')
        .attr('x1', sumX + sumBoxW + 2).attr('y1', sumY + 14)
        .attr('x2', outX - 2).attr('y2', sumY + 14)
        .attr('stroke', '#555').attr('stroke-width', 1.5);
    g.append('path')
        .attr('d', `M${outX - 2},${sumY + 14} l-5,-3.5 l0,7 z`)
        .attr('fill', '#555');

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

    if (shape && shape.length >= 2) {
        const lastDim = shape[shape.length - 1];
        const lastDimName = dimNames[dimNames.length - 1] || '';
        const secDim = shape[shape.length - 2];
        const secDimName = dimNames.length >= 2 ? dimNames[dimNames.length - 2] : '';

        const colLabel = lastDimName ? `${lastDimName}=${lastDim}` : `${lastDim}`;
        g.append('text').attr('class', 'dim-label')
            .attr('x', outX + outW / 2).attr('y', outBlockY + outBlockH + 12)
            .attr('text-anchor', 'middle').attr('font-size', '8px').attr('fill', '#aaa')
            .text(colLabel);

        const rowLabel = secDimName ? `${secDimName}=${secDim}` : `${secDim}`;
        g.append('text').attr('class', 'dim-label')
            .attr('x', outX + outW + 6).attr('y', outBlockY + outBlockH / 2 + 3)
            .attr('font-size', '8px').attr('fill', '#aaa')
            .text(rowLabel);

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
