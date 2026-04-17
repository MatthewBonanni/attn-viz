// matmul.js — Matmul (L-shaped diagram) and All-Reduce detail visualizations
import { detailMetrics, drawDetailBlock, drawDetailBlock3D, TP_COLORS, RANK_COLORS, SHARD_BASE, SHARD_OPACITY, SHARD_HIGHLIGHT_OPACITY } from './shared.js';

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
    const SHARDED_NEUTRAL = '#f39c12';
    const NEUTRALIZE_COLORS = new Set(['#e74c3c', '#2ecc71', '#f39c12', '#e67e22']);
    const colorFor = (t) => {
        const s = (t.tpSharded && tpSize > 1) || (t.dpSharded && dpSize > 1);
        return (s && t.type !== 'weight' && t.type !== 'mask' && NEUTRALIZE_COLORS.has(t.color)) ? SHARDED_NEUTRAL : t.color;
    };

    // Matrix A (left of result)
    const aX = originX - wA - pad;
    const aSharded = (A.tpSharded && tpSize > 1) || (A.dpSharded && dpSize > 1);
    drawDetailBlock(g, aX, originY, wA, hA, colorFor(A), A.label, aSharded && A.type !== 'weight' && A.type !== 'mask');
    g.append('text').attr('class', 'dim-label')
        .attr('x', aX + wA / 2).attr('y', originY + hA + 14)
        .attr('text-anchor', 'middle').text(A.dimNames ? A.dimNames[A.dimNames.length - 1] + '=' + inner : inner);
    g.append('text').attr('class', 'dim-label')
        .attr('x', aX - 8).attr('y', originY + hA / 2 + 3)
        .attr('text-anchor', 'end').text(A.dimNames ? A.dimNames[A.dimNames.length - 2] + '=' + rows_a : rows_a);

    // Matrix B (above result)
    const bX = originX, bY_pos = originY - hB - pad;
    if (B_tensor) {
        const shB = B_tensor.shape;
        const bSharded = (B_tensor.tpSharded && tpSize > 1) || (B_tensor.dpSharded && dpSize > 1);
        drawDetailBlock(g, bX, bY_pos, wC, hB, colorFor(B_tensor), B_tensor.label, bSharded && B_tensor.type !== 'weight' && B_tensor.type !== 'mask');
        const bDimNames = B_tensor.dimNames || [];
        const bColName = bTransposed ? bDimNames[bDimNames.length - 2] || '' : bDimNames[bDimNames.length - 1] || '';
        const bRowName = bTransposed ? bDimNames[bDimNames.length - 1] || '' : bDimNames[bDimNames.length - 2] || '';
        const bRowVal = bTransposed ? shB[shB.length - 1] : shB[shB.length - 2];
        g.append('text').attr('class', 'dim-label')
            .attr('x', bX + wC / 2).attr('y', bY_pos - 6)
            .attr('text-anchor', 'middle').text(bColName ? bColName + '=' + cols_b : cols_b);
        g.append('text').attr('class', 'dim-label')
            .attr('x', bX - 8).attr('y', bY_pos + hB / 2 + 3)
            .attr('text-anchor', 'end').text(bRowName ? bRowName + '=' + bRowVal : inner);
    }

    // Result C (center)
    const cSharded = (C.tpSharded && tpSize > 1) || (C.dpSharded && dpSize > 1);
    drawDetailBlock(g, originX, originY, wC, hC, colorFor(C), C.label, cSharded && C.type !== 'weight' && C.type !== 'mask');
    const resultColName = B_tensor && B_tensor.dimNames
        ? (bTransposed ? B_tensor.dimNames[B_tensor.dimNames.length - 2] : B_tensor.dimNames[B_tensor.dimNames.length - 1])
        : (C.dimNames ? C.dimNames[C.dimNames.length - 1] : '');
    g.append('text').attr('class', 'dim-label')
        .attr('x', originX + wC / 2).attr('y', originY + hC + 14)
        .attr('text-anchor', 'middle').text(resultColName ? resultColName + '=' + cols_b : cols_b);

    // --- Tooltip system ---
    let tooltipG = null;
    let baseSvgH = 0;
    function formatDimRange(dimName, start, end) {
        const d_h = params && params.d_h;
        const n_h = params && params.n_h;
        const n_kv = params && params.n_kv;
        if (d_h && d_h > 1) {
            if (dimName === 'D' && n_h)
                return `n_h[${Math.floor(start / d_h)}\u2013${Math.floor(end / d_h)}] (${dimName}[${start}\u2013${end}])`;
            if (dimName.includes('\u00b7d_h') && n_kv)
                return `n_kv[${Math.floor(start / d_h)}\u2013${Math.floor(end / d_h)}] (${dimName}[${start}\u2013${end}])`;
        }
        return `${dimName}[${start}\u2013${end}]`;
    }

    function showTooltip(info, tensor, tx, ty) {
        if (tooltipG) tooltipG.remove();
        if (!info) { tooltipG = null; if (baseSvgH) svg.attr('height', baseSvgH); return; }
        tooltipG = g.append('g');

        const shape = tensor.shape;
        const dimNames = tensor.dimNames || [];
        const isTp = tensor.tpSharded && tpSize > 1;
        const isDp = tensor.dpSharded && dpSize > 1;
        const dpFracs = (tensor.dpBoundaryFracs) || uniformFracs(dpSize);
        const parts = [];
        const rankIdx = (info.dp || 0) * Math.max(1, tpSize) + (info.tp || 0);
        const cellColor = RANK_COLORS[rankIdx % RANK_COLORS.length];

        if (info.dp != null) {
            parts.push(`DP Rank ${info.dp}`);
            if (B_count > 1) {
                const reqs = [];
                for (let r = 0; r < B_count; r++) {
                    if (Math.floor(r * dpSize / B_count) === info.dp) reqs.push(r);
                }
                parts.push(`Req ${reqs.join(', ')}`);
            }
            if (isDp) {
                const rowDim = dimNames[dimNames.length - 2] || 'rows';
                const rowSize = shape[shape.length - 2];
                const dpStart = Math.round(dpFracs[info.dp] * rowSize);
                const dpEnd = Math.round(dpFracs[info.dp + 1] * rowSize) - 1;
                parts.push(formatDimRange(rowDim, dpStart, dpEnd));
            }
        }
        if (isTp && info.tp != null) {
            parts.push(`TP Rank ${info.tp}`);
            const tpDim = tensor.tpDim;
            const dimLabel = tpDim === 0 ? (dimNames[0] || 'rows') : (dimNames[1] || 'cols');
            const perRank = Math.round(shape[tpDim] / tpSize);
            parts.push(formatDimRange(dimLabel, info.tp * perRank, (info.tp + 1) * perRank - 1));
        }
        if (isTp && isDp) parts.unshift(`Rank ${rankIdx}`);
        if (info.skipped) {
            parts.push('Not computed \u2014 sequences don\u2019t');
            parts.push('attend across DP boundaries');
        }

        const padX = 10, padY = 8, lineH = 16;
        const boxW = info.skipped ? 260 : 220, boxH = padY * 2 + parts.length * lineH;
        const ttX = svgW / 2 - boxW / 2;
        const ttY = baseSvgH - 5;
        tooltipG.append('rect')
            .attr('x', ttX).attr('y', ttY)
            .attr('width', boxW).attr('height', boxH)
            .attr('rx', 6)
            .attr('fill', '#12141f').attr('fill-opacity', 0.95)
            .attr('stroke', cellColor).attr('stroke-width', 1.5).attr('stroke-opacity', 0.7);
        parts.forEach((text, i) => {
            tooltipG.append('text')
                .attr('x', ttX + padX).attr('y', ttY + padY + (i + 1) * lineH - 3)
                .attr('fill', i === 0 ? cellColor : '#bbb')
                .attr('font-size', '11px').attr('font-weight', i === 0 ? '600' : '400')
                .text(text);
        });

        const tooltipBottom = ttY + boxH + 40;
        if (tooltipBottom > baseSvgH) svg.attr('height', tooltipBottom);
    }

    // --- Unified sharding overlay system ---
    // All shard cells across A, B, C are tracked for cross-highlighting.
    const allShardCells = [];

    function highlightShardCells(matchFn) {
        for (const c of allShardCells) {
            const on = matchFn(c);
            c.el.attr('fill-opacity', on ? SHARD_HIGHLIGHT_OPACITY : SHARD_OPACITY);
            if (on) c.el.attr('stroke', '#fff').attr('stroke-width', 1.5);
            else c.el.attr('stroke', 'none');
        }
    }

    function clearShardHighlight() {
        for (const c of allShardCells) {
            c.el.attr('fill-opacity', SHARD_OPACITY).attr('stroke', 'none');
        }
    }

    // Draw combined TP+DP grid on a 2D block
    function drawShardGrid(tensor, bx, by, bw, bh, key, tooltipX, tooltipY) {
        const isTp = tensor.tpSharded && tpSize > 1 && tensor.tpDim != null;
        const isDp = tensor.dpSharded && dpSize > 1;
        if (!isTp && !isDp) return;

        // Skip TP on weights for A position, and on 3D+ C
        if (key === 'a' && tensor.type === 'weight' && isTp) return;
        if (key === 'c' && tensor.shape.length !== 2 && isTp) return;

        const effTpSize = isTp ? tpSize : 1;
        const effDpSize = isDp ? dpSize : 1;

        // Determine TP visual direction
        let tpVisual = null;
        if (isTp) {
            const tpDim = tensor.tpDim;
            if (key === 'b') {
                tpVisual = bTransposed ? (tpDim === 0 ? 'cols' : 'rows') : (tpDim === 1 ? 'cols' : 'rows');
            } else {
                tpVisual = tpDim === 1 ? 'cols' : 'rows';
            }
        }

        // Determine DP visual direction and fracs
        let dpVisual = 'rows';
        const dpFracs = (tensor.dpBoundaryFracs && tensor.dpBoundaryFracs.length === dpSize + 1)
            ? tensor.dpBoundaryFracs : uniformFracs(dpSize);
        if (isDp && key === 'b') {
            const bDimNames = tensor.dimNames || [];
            const bDpIdx = bDimNames.lastIndexOf(tensor.dpSharded);
            const bDpIsSecondLast = bDpIdx === bDimNames.length - 2;
            dpVisual = (bDpIsSecondLast ? bTransposed : !bTransposed) ? 'cols' : 'rows';
        }

        for (let dp = 0; dp < effDpSize; dp++) {
            for (let tp = 0; tp < effTpSize; tp++) {
                const rankIdx = dp * effTpSize + tp;
                const cellColor = RANK_COLORS[rankIdx % RANK_COLORS.length];

                let sx, sy, sw, sh;
                // Compute position based on DP and TP directions
                const dpF0 = dpFracs[dp], dpF1 = dpFracs[dp + 1];

                if (isDp && isTp) {
                    // Both active — subdivide in both directions
                    if (dpVisual === 'rows' && tpVisual === 'cols') {
                        sx = bx + (tp / effTpSize) * bw; sw = bw / effTpSize;
                        sy = by + dpF0 * bh; sh = (dpF1 - dpF0) * bh;
                    } else if (dpVisual === 'rows' && tpVisual === 'rows') {
                        sx = bx; sw = bw;
                        const dpH = (dpF1 - dpF0) * bh;
                        sy = by + dpF0 * bh + (tp / effTpSize) * dpH;
                        sh = dpH / effTpSize;
                    } else if (dpVisual === 'cols' && tpVisual === 'cols') {
                        sy = by; sh = bh;
                        const dpW = (dpF1 - dpF0) * bw;
                        sx = bx + dpF0 * bw + (tp / effTpSize) * dpW;
                        sw = dpW / effTpSize;
                    } else {
                        sx = bx + dpF0 * bw; sw = (dpF1 - dpF0) * bw;
                        sy = by + (tp / effTpSize) * bh; sh = bh / effTpSize;
                    }
                } else if (isTp) {
                    if (tpVisual === 'cols') {
                        sx = bx + (tp / effTpSize) * bw; sw = bw / effTpSize;
                        sy = by; sh = bh;
                    } else {
                        sx = bx; sw = bw;
                        sy = by + (tp / effTpSize) * bh; sh = bh / effTpSize;
                    }
                } else {
                    if (dpVisual === 'cols') {
                        sx = bx + dpF0 * bw; sw = (dpF1 - dpF0) * bw;
                        sy = by; sh = bh;
                    } else {
                        sx = bx; sw = bw;
                        sy = by + dpF0 * bh; sh = (dpF1 - dpF0) * bh;
                    }
                }

                const el = g.append('rect')
                    .attr('x', sx).attr('y', sy).attr('width', sw).attr('height', sh)
                    .attr('fill', cellColor).attr('fill-opacity', SHARD_OPACITY).attr('stroke', 'none')
                    .style('cursor', 'pointer');

                const cellData = { el, dp, tp, rankIdx, key, isTp, isDp };
                allShardCells.push(cellData);

                el.on('mouseenter', () => {
                    highlightShardCells(c => {
                        const dpMatch = !isDp || !c.isDp || c.dp === dp;
                        const tpMatch = !isTp || !c.isTp || c.tp === tp;
                        return dpMatch && tpMatch;
                    });
                    showTooltip({ dp: isDp ? dp : null, tp: isTp ? tp : null }, tensor, tooltipX, tooltipY);
                });
                el.on('mouseleave', () => {
                    clearShardHighlight();
                    showTooltip(null);
                });
            }
        }

        // Boundary lines
        if (effTpSize > 1) {
            for (let tp = 1; tp < effTpSize; tp++) {
                if (tpVisual === 'cols') {
                    const lx = bx + (tp / effTpSize) * bw;
                    g.append('line')
                        .attr('x1', lx).attr('y1', by).attr('x2', lx).attr('y2', by + bh)
                        .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
                } else {
                    const ly = by + (tp / effTpSize) * bh;
                    g.append('line')
                        .attr('x1', bx).attr('y1', ly).attr('x2', bx + bw).attr('y2', ly)
                        .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
                }
            }
        }
        if (effDpSize > 1) {
            for (let dp = 1; dp < effDpSize; dp++) {
                if (dpVisual === 'cols') {
                    const lx = bx + dpFracs[dp] * bw;
                    g.append('line')
                        .attr('x1', lx).attr('y1', by).attr('x2', lx).attr('y2', by + bh)
                        .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
                } else {
                    const ly = by + dpFracs[dp] * bh;
                    g.append('line')
                        .attr('x1', bx).attr('y1', ly).attr('x2', bx + bw).attr('y2', ly)
                        .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
                }
            }
        }
    }

    // Draw sharding on all three matrices
    const tooltipTpY = originY + hC + 20;
    drawShardGrid(A, aX, originY, wA, hA, 'a', aX, tooltipTpY);
    if (B_tensor) drawShardGrid(B_tensor, bX, bY_pos, wC, hB, 'b', originX, tooltipTpY);
    drawShardGrid(C, originX, originY, wC, hC, 'c', originX, tooltipTpY);

    // --- DP block-diagonal on result C (special case: attention score matmuls) ---
    if (hasDp && isBlockDiag && colFracs) {
        const offDiagCells = [];

        for (let dr = 0; dr < dpSize; dr++) {
            for (let dc = 0; dc < dpSize; dc++) {
                const cellX = originX + colFracs[dc] * wC;
                const cellW2 = (colFracs[dc + 1] - colFracs[dc]) * wC;
                const cellY = originY + rowFracs[dr] * hC;
                const cellH2 = (rowFracs[dr + 1] - rowFracs[dr]) * hC;

                if (dr !== dc) {
                    const bg = g.append('rect')
                        .attr('x', cellX).attr('y', cellY)
                        .attr('width', cellW2).attr('height', cellH2)
                        .attr('fill', '#000').attr('fill-opacity', 0.35)
                        .attr('stroke', '#333').attr('stroke-width', 0.5);

                    g.append('line')
                        .attr('x1', cellX + 2).attr('y1', cellY + 2)
                        .attr('x2', cellX + cellW2 - 2).attr('y2', cellY + cellH2 - 2)
                        .attr('stroke', '#444').attr('stroke-width', 0.75)
                        .attr('pointer-events', 'none');
                    g.append('line')
                        .attr('x1', cellX + cellW2 - 2).attr('y1', cellY + 2)
                        .attr('x2', cellX + 2).attr('y2', cellY + cellH2 - 2)
                        .attr('stroke', '#444').attr('stroke-width', 0.75)
                        .attr('pointer-events', 'none');

                    const hoverRect = g.append('rect')
                        .attr('x', cellX).attr('y', cellY)
                        .attr('width', cellW2).attr('height', cellH2)
                        .attr('fill', 'transparent')
                        .style('cursor', 'pointer');

                    offDiagCells.push({ el: bg, dr, dc, hover: hoverRect });

                    hoverRect.on('mouseenter', () => {
                        highlightShardCells(c => c.dp === dr);
                        for (const oc of offDiagCells) {
                            oc.el.attr('fill-opacity', oc.dr === dr ? 0.5 : 0.35);
                            if (oc.dr === dr) oc.el.attr('stroke', '#888').attr('stroke-width', 1);
                            else oc.el.attr('stroke', '#333').attr('stroke-width', 0.5);
                        }
                        showTooltip({ dp: dr, skipped: true }, C, originX, tooltipTpY);
                    });
                    hoverRect.on('mouseleave', () => {
                        clearShardHighlight();
                        for (const oc of offDiagCells) {
                            oc.el.attr('fill-opacity', 0.35).attr('stroke', '#333').attr('stroke-width', 0.5);
                        }
                        showTooltip(null);
                    });
                }
            }
        }
    }

    // --- Highlight row/col (on top of DP overlays) ---
    const highlightRow = Math.min(2, rows_a - 1);
    const highlightCol = Math.min(2, cols_b - 1);
    const rowH = hA / rows_a;
    const colW = wC / cols_b;

    g.append('rect')
        .attr('x', aX).attr('y', originY + highlightRow * rowH)
        .attr('width', wA).attr('height', Math.max(rowH, 2))
        .attr('fill', '#fff').attr('fill-opacity', 0.25)
        .attr('stroke', '#fff').attr('stroke-width', 1);

    if (B_tensor) {
        g.append('rect')
            .attr('x', originX + highlightCol * colW).attr('y', bY_pos)
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

    // --- Sharding info section ---
    const hasAnySharding = allShardCells.length > 0;
    if (hasAnySharding) {
        noteY += 4;
        if (hasDp && isBlockDiag) {
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
        } else if (hasDp && aRowIsDp && B_tensor && B_tensor.dpSharded) {
            g.append('text').attr('class', 'dim-label')
                .attr('x', svgW / 2).attr('y', noteY)
                .attr('text-anchor', 'middle').attr('fill', '#e67e22').attr('font-size', '10px')
                .text('Each DP rank computes independently on its portion');
            noteY += 18;
        } else if (hasDp && aRowIsDp) {
            g.append('text').attr('class', 'dim-label')
                .attr('x', svgW / 2).attr('y', noteY)
                .attr('text-anchor', 'middle').attr('fill', '#e67e22').attr('font-size', '10px')
                .text('Each DP rank processes its batch independently (weight shared)');
            noteY += 18;
        }

        // Rank legend
        const effDpSize = hasDp ? dpSize : 1;
        const effTpSize = (tpSize > 1 && (A.tpSharded || (B_tensor && B_tensor.tpSharded) || C.tpSharded)) ? tpSize : 1;
        const totalRanks = effDpSize * effTpSize;
        const legendCols = Math.min(totalRanks, 4);
        const legendRows = Math.ceil(totalRanks / legendCols);
        const colWLeg = svgW / legendCols;

        for (let ri = 0; ri < totalRanks; ri++) {
            const col = ri % legendCols;
            const row = Math.floor(ri / legendCols);
            const dpRank = Math.floor(ri / effTpSize);
            const tpRank = ri % effTpSize;
            const rankColor = RANK_COLORS[ri % RANK_COLORS.length];
            const lx = col * colWLeg + 12;
            const ly = noteY + row * 18;

            const legendItem = g.append('g').style('cursor', 'pointer');
            legendItem.append('circle')
                .attr('cx', lx + 4).attr('cy', ly - 3)
                .attr('r', 5).attr('fill', rankColor).attr('fill-opacity', 0.7);

            let label = `R${ri}`;
            if (effDpSize > 1 && effTpSize > 1) label += ` (DP${dpRank},TP${tpRank})`;
            else if (effDpSize > 1) label += ` DP${dpRank}`;
            else if (effTpSize > 1) label += ` TP${tpRank}`;

            legendItem.append('text').attr('class', 'dim-label')
                .attr('x', lx + 12).attr('y', ly)
                .attr('text-anchor', 'start').attr('fill', rankColor).attr('font-size', '10px')
                .text(label);

            legendItem.on('mouseenter', () => {
                highlightShardCells(c => {
                    const dpMatch = effDpSize <= 1 || !c.isDp || c.dp === dpRank;
                    const tpMatch = effTpSize <= 1 || !c.isTp || c.tp === tpRank;
                    return dpMatch && tpMatch;
                });
            });
            legendItem.on('mouseleave', () => {
                clearShardHighlight();
            });
        }
        noteY += legendRows * 18 + 8;
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
        drawDetailBlock3D(g, beforeX, noteY + maxTopPad, beforeW, beforeH, beforeD, colorFor(C), '',
            beforeGrp, cTpInfo, cDpInfo, null, C.color);

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
        drawDetailBlock3D(g, afterX, noteY + maxTopPad, afterW, afterH, afterD, colorFor(C), C.label,
            afterGrp, cTpInfo, cDpInfo, null, C.color);

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
    baseSvgH = noteY + 20;
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
