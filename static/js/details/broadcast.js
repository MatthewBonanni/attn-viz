// broadcast.js — Broadcast detail visualization
import { detailMetrics, drawDetailBlock, drawDetailBlock3D, TP_COLORS, RANK_COLORS, SHARD_BASE, SHARD_OPACITY, SHARD_HIGHLIGHT_OPACITY } from './shared.js';

export function drawBroadcastDetail(svg, op, tensorMap, params) {
    const { cx: svgCx } = detailMetrics();
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

    const tpSize = (params && params.tp_size) || 1;
    const dpSize = (params && params.dp_size) || 1;
    const B = (params && params.B) || 1;
    const SHARDED_NEUTRAL = '#f39c12';
    const NEUTRALIZE_COLORS = new Set(['#e74c3c', '#2ecc71', '#f39c12', '#e67e22']);
    const blockColorFor = (t) => {
        const s = (t.tpSharded && tpSize > 1) || (t.dpSharded && dpSize > 1);
        return (s && t.type !== 'weight' && t.type !== 'mask' && NEUTRALIZE_COLORS.has(t.color)) ? SHARDED_NEUTRAL : t.color;
    };
    const tpInfoFor = (t) => {
        if (!(t.tpSharded && tpSize > 1)) return null;
        const ndim = t.shape.length;
        if (t.tpDim == null) return { tpSize };
        if (ndim === 3 && t.tpDim === 0) return { tpSize };
        if (ndim === 4 && (t.tpDim === 0 || t.tpDim === 1)) return { tpSize };
        return null;
    };
    const dpInfoFor = (t) => (t.dpSharded && dpSize > 1) ? { dpSize, dpFracs: t.dpBoundaryFracs } : null;

    // Render as 2D when it's a 2D tensor or when TP is along a face dim
    const inFaceTp = inShape.length <= 2 || ((input.tpSharded && tpSize > 1) && input.tpDim != null && !tpInfoFor(input));
    const outFaceTp = outShape.length <= 2 || ((output.tpSharded && tpSize > 1) && output.tpDim != null && !tpInfoFor(output));

    // Use a single scale factor across both shapes so aspect ratios are preserved.
    const maxBlockPx = 260;
    const allFaceDims = [
        inShape[inShape.length - 1], inShape[inShape.length - 2],
        outShape[outShape.length - 1], outShape[outShape.length - 2],
    ];
    const maxSqrt = Math.max(...allFaceDims.map(v => Math.sqrt(v)));
    const scaleFactor = maxSqrt > 0 ? maxBlockPx / maxSqrt : 7;
    const scale = (v) => Math.max(24, Math.min(maxBlockPx, Math.sqrt(v) * scaleFactor));

    let inW = scale(inShape[inShape.length - 1]);
    let inH = scale(inShape[inShape.length - 2]);
    const inDepthVal = inShape.length >= 3 ? inShape[0] * (inShape.length >= 4 ? inShape[1] : 1) : 1;
    let inD = inFaceTp ? 0 : Math.max(4, Math.min(200, Math.sqrt(inDepthVal) * 16));

    let outW = scale(outShape[outShape.length - 1]);
    let outH = scale(outShape[outShape.length - 2]);
    const outDepthVal = outShape.length >= 3 ? outShape[0] * (outShape.length >= 4 ? outShape[1] : 1) : 1;
    let outD = outFaceTp ? 0 : Math.max(4, Math.min(200, Math.sqrt(outDepthVal) * 16));

    // Constrain total layout width to panel
    const arrowGap = 100;
    const availW = mid * 2 - 60;
    const rawTotalW = inW + inD * 0.7 + arrowGap + outW + outD * 0.7;
    if (rawTotalW > availW) {
        const shrink = availW / rawTotalW;
        inW *= shrink; outW *= shrink; inH *= shrink; outH *= shrink; inD *= shrink; outD *= shrink;
    }
    const totalBlockW = inW + inD * 0.7 + arrowGap + outW + outD * 0.7;
    const inX = Math.max(30, (mid * 2 - totalBlockW) / 2);
    const topPad = Math.max(inD, outD) * 0.4 + 10;
    const blockY = topPad;

    // --- Tooltip state ---
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
        const dpFracs = (tensor.dpBoundaryFracs) || Array.from({length: dpSize + 1}, (_, i) => i / dpSize);

        const parts = [];
        const rankIdx = (info.dp || 0) * Math.max(1, tpSize) + (info.tp || 0);
        const cellColor = RANK_COLORS[rankIdx % RANK_COLORS.length];

        if (isDp && info.dp != null) {
            parts.push(`DP Rank ${info.dp}`);
            if (B > 1) {
                const reqs = [];
                for (let r = 0; r < B; r++) {
                    if (Math.floor(r * dpSize / B) === info.dp) reqs.push(r);
                }
                parts.push(`Req ${reqs.join(', ')}`);
            }
            const rowDim = shape.length >= 3
                ? (dimNames[dimNames.length - 2] || 'rows')
                : (dimNames[0] || 'rows');
            const rowSize = shape.length >= 3 ? shape[shape.length - 2] : shape[0];
            const dpStart = Math.round(dpFracs[info.dp] * rowSize);
            const dpEnd = Math.round(dpFracs[info.dp + 1] * rowSize) - 1;
            parts.push(formatDimRange(rowDim, dpStart, dpEnd));
        }
        if (isTp && info.tp != null) {
            parts.push(`TP Rank ${info.tp}`);
            const tpDim = tensor.tpDim;
            if (tpDim != null) {
                const dimLabel = dimNames[tpDim] || `dim${tpDim}`;
                const perRank = Math.round(shape[tpDim] / tpSize);
                parts.push(formatDimRange(dimLabel, info.tp * perRank, (info.tp + 1) * perRank - 1));
            } else if (typeof tensor.tpSharded === 'string') {
                const dimIdx = dimNames.indexOf(tensor.tpSharded);
                if (dimIdx >= 0) {
                    const perRank = Math.round(shape[dimIdx] / tpSize);
                    parts.push(formatDimRange(dimNames[dimIdx], info.tp * perRank, (info.tp + 1) * perRank - 1));
                }
            }
        }
        if (isTp && isDp) {
            parts.unshift(`Rank ${rankIdx}`);
        }

        const padX = 10, padY = 8, lineH = 16;
        const boxW = 220, boxH = padY * 2 + parts.length * lineH;
        const ttX = mid - boxW / 2;
        const ttY = baseSvgH - 20;

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

    // --- Draw input block ---
    const inGrp = inShape.length === 4 ? { outer: inShape[0], inner: inShape[1] } : null;
    const inIsTp = input.tpSharded && tpSize > 1;
    const inIsDp = input.dpSharded && dpSize > 1;
    const inHasParallelism = inIsTp || inIsDp;
    const inTooltipX = inFaceTp ? inX : inX + inW + inD * 0.7 + 12;
    const inTooltipY = inFaceTp ? blockY + inH + 20 : blockY + inH * 0.2;

    let inApi;

    if (inFaceTp) {
        // 2D rendering for 2D tensors or tensors with face-dim TP
        drawDetailBlock(g, inX, blockY, inW, inH, blockColorFor(input), input.label, inHasParallelism);

        const inGridCells = [];
        inApi = null;

        if (inHasParallelism) {
            const tpDim = input.tpDim;
            const faceDim = inShape.length <= 2 ? tpDim : tpDim - (inShape.length - 2);
            const dpFracs = (input.dpBoundaryFracs) || Array.from({length: dpSize + 1}, (_, i) => i / dpSize);
            const effDpSize = inIsDp ? dpSize : 1;
            const effTpSize = inIsTp ? tpSize : 1;

            for (let dp = 0; dp < effDpSize; dp++) {
                for (let tp = 0; tp < effTpSize; tp++) {
                    const rankIdx = dp * effTpSize + tp;
                    const cellColor = RANK_COLORS[rankIdx % RANK_COLORS.length];

                    let sx, sy, sw, sh;
                    const dpY0 = dpFracs[dp] * inH, dpY1 = dpFracs[dp + 1] * inH;
                    if (faceDim === 1) {
                        sx = inX + (tp / effTpSize) * inW;
                        sw = inW / effTpSize;
                        sy = blockY + dpY0;
                        sh = dpY1 - dpY0;
                    } else {
                        sx = inX;
                        sw = inW;
                        sy = blockY + dpY0 + (tp / effTpSize) * (dpY1 - dpY0);
                        sh = (dpY1 - dpY0) / effTpSize;
                    }

                    const cell = g.append('rect')
                        .attr('x', sx).attr('y', sy).attr('width', sw).attr('height', sh)
                        .attr('fill', cellColor).attr('fill-opacity', SHARD_OPACITY).attr('stroke', 'none')
                        .style('cursor', 'pointer');
                    inGridCells.push({ el: cell, dp, tp });

                    cell.on('mouseenter', () => {
                        showTooltip({ dp, tp }, input, inTooltipX, inTooltipY);
                        inGridCells.forEach(c => {
                            const match = c.dp === dp && c.tp === tp;
                            c.el.attr('fill-opacity', match ? SHARD_HIGHLIGHT_OPACITY : SHARD_OPACITY);
                            if (match) c.el.attr('stroke', '#fff').attr('stroke-width', 1.5);
                            else c.el.attr('stroke', 'none');
                        });
                        if (outApi) outApi.highlightCell(dp, tp);
                    });
                    cell.on('mouseleave', () => {
                        showTooltip(null);
                        inGridCells.forEach(c => c.el.attr('fill-opacity', SHARD_OPACITY).attr('stroke', 'none'));
                        if (outApi) outApi.clear();
                    });
                }
            }

            // TP boundary lines
            for (let tp = 1; tp < effTpSize; tp++) {
                if (faceDim === 1) {
                    const lx = inX + (tp / effTpSize) * inW;
                    g.append('line')
                        .attr('x1', lx).attr('y1', blockY).attr('x2', lx).attr('y2', blockY + inH)
                        .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
                } else {
                    const ly = blockY + (tp / effTpSize) * inH;
                    g.append('line')
                        .attr('x1', inX).attr('y1', ly).attr('x2', inX + inW).attr('y2', ly)
                        .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
                }
            }
            // DP boundary lines
            for (let dp = 1; dp < effDpSize; dp++) {
                const ly = blockY + dpFracs[dp] * inH;
                g.append('line')
                    .attr('x1', inX).attr('y1', ly).attr('x2', inX + inW).attr('y2', ly)
                    .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
            }

            inApi = {
                highlightTP(tp) {
                    inGridCells.forEach(c => {
                        const on = c.tp === tp;
                        c.el.attr('fill-opacity', on ? SHARD_HIGHLIGHT_OPACITY : SHARD_OPACITY);
                        if (on) c.el.attr('stroke', '#fff').attr('stroke-width', 1);
                        else c.el.attr('stroke', 'none');
                    });
                },
                highlightDP(dp) {
                    inGridCells.forEach(c => {
                        const on = c.dp === dp;
                        c.el.attr('fill-opacity', on ? SHARD_HIGHLIGHT_OPACITY : SHARD_OPACITY);
                        if (on) c.el.attr('stroke', '#fff').attr('stroke-width', 1);
                        else c.el.attr('stroke', 'none');
                    });
                },
                highlightCell(dp, tp) {
                    inGridCells.forEach(c => {
                        const on = c.dp === dp && c.tp === tp;
                        c.el.attr('fill-opacity', on ? SHARD_HIGHLIGHT_OPACITY : SHARD_OPACITY);
                        if (on) c.el.attr('stroke', '#fff').attr('stroke-width', 1.5);
                        else c.el.attr('stroke', 'none');
                    });
                },
                clear() {
                    inGridCells.forEach(c => c.el.attr('fill-opacity', SHARD_OPACITY).attr('stroke', 'none'));
                }
            };
        }
    } else {
        // 3D rendering
        const inCellCb = inHasParallelism ? (info) => {
            showTooltip(info, input, inTooltipX, inTooltipY);
            if (outApi) {
                if (!info) { outApi.clear(); }
                else if (info.dp != null && info.tp != null) outApi.highlightCell(info.dp, info.tp);
                else if (info.dp != null) outApi.highlightDP(info.dp);
                else if (info.tp != null) outApi.highlightTP(info.tp);
                else outApi.clear();
            }
        } : null;

        inApi = drawDetailBlock3D(g, inX, blockY, inW, inH, inD, blockColorFor(input), input.label,
            inGrp, tpInfoFor(input), dpInfoFor(input), inCellCb, input.color);
    }

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
    if (inShape.length >= 3 && !inFaceTp) {
        const depthLabel = inShape.length === 4
            ? `${inNames[0] || ''}\u00b7${inNames[1] || ''}=${inDepthVal}`
            : `${inNames[0] || ''}=${inShape[0]}`;
        g.append('text').attr('class', 'dim-label')
            .attr('x', inX + inW + inD * 0.7 / 2 + 6)
            .attr('y', blockY - inD * 0.4 / 2 - 2)
            .text(depthLabel);
    }

    // Arrow
    const arrowX = inX + inW + inD * 0.7 + arrowGap / 2;
    g.append('text').attr('x', arrowX).attr('y', blockY + inH / 2 + 4)
        .attr('fill', '#888').attr('font-size', '24px')
        .attr('text-anchor', 'middle').text('\u2192');

    // --- Draw output block ---
    const outX = inX + inW + inD * 0.7 + arrowGap;
    const outGrp = outShape.length === 4 ? { outer: outShape[0], inner: outShape[1] } : null;
    const outIsTp = output.tpSharded && tpSize > 1;
    const outIsDp = output.dpSharded && dpSize > 1;
    const outHasParallelism = outIsTp || outIsDp;
    const outTooltipX = outFaceTp ? outX : outX + outW + outD * 0.7 + 12;
    const outTooltipY = outFaceTp ? blockY + outH + 20 : blockY + outH * 0.2;

    let outApi;

    if (outFaceTp) {
        // 2D rendering for 2D tensors or tensors with face-dim TP
        drawDetailBlock(g, outX, blockY, outW, outH, blockColorFor(output), output.label, outHasParallelism);

        const outGridCells = [];
        outApi = null;

        if (outHasParallelism) {
            const tpDim = output.tpDim;
            const faceDim = outShape.length <= 2 ? tpDim : tpDim - (outShape.length - 2);
            const dpFracs = (output.dpBoundaryFracs) || Array.from({length: dpSize + 1}, (_, i) => i / dpSize);
            const effDpSize = outIsDp ? dpSize : 1;
            const effTpSize = outIsTp ? tpSize : 1;

            for (let dp = 0; dp < effDpSize; dp++) {
                for (let tp = 0; tp < effTpSize; tp++) {
                    const rankIdx = dp * effTpSize + tp;
                    const cellColor = RANK_COLORS[rankIdx % RANK_COLORS.length];

                    let sx, sy, sw, sh;
                    const dpY0 = dpFracs[dp] * outH, dpY1 = dpFracs[dp + 1] * outH;
                    if (faceDim === 1) {
                        sx = outX + (tp / effTpSize) * outW;
                        sw = outW / effTpSize;
                        sy = blockY + dpY0;
                        sh = dpY1 - dpY0;
                    } else {
                        sx = outX;
                        sw = outW;
                        sy = blockY + dpY0 + (tp / effTpSize) * (dpY1 - dpY0);
                        sh = (dpY1 - dpY0) / effTpSize;
                    }

                    const cell = g.append('rect')
                        .attr('x', sx).attr('y', sy).attr('width', sw).attr('height', sh)
                        .attr('fill', cellColor).attr('fill-opacity', SHARD_OPACITY).attr('stroke', 'none')
                        .style('cursor', 'pointer');
                    outGridCells.push({ el: cell, dp, tp });

                    cell.on('mouseenter', () => {
                        showTooltip({ dp, tp }, output, outTooltipX, outTooltipY);
                        outGridCells.forEach(c => {
                            const match = c.dp === dp && c.tp === tp;
                            c.el.attr('fill-opacity', match ? SHARD_HIGHLIGHT_OPACITY : SHARD_OPACITY);
                            if (match) c.el.attr('stroke', '#fff').attr('stroke-width', 1.5);
                            else c.el.attr('stroke', 'none');
                        });
                        if (inApi) inApi.highlightCell(dp, tp);
                    });
                    cell.on('mouseleave', () => {
                        showTooltip(null);
                        outGridCells.forEach(c => c.el.attr('fill-opacity', SHARD_OPACITY).attr('stroke', 'none'));
                        if (inApi) inApi.clear();
                    });
                }
            }

            // TP boundary lines
            for (let tp = 1; tp < effTpSize; tp++) {
                if (faceDim === 1) {
                    const lx = outX + (tp / effTpSize) * outW;
                    g.append('line')
                        .attr('x1', lx).attr('y1', blockY).attr('x2', lx).attr('y2', blockY + outH)
                        .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
                } else {
                    const ly = blockY + (tp / effTpSize) * outH;
                    g.append('line')
                        .attr('x1', outX).attr('y1', ly).attr('x2', outX + outW).attr('y2', ly)
                        .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
                }
            }
            // DP boundary lines
            for (let dp = 1; dp < effDpSize; dp++) {
                const ly = blockY + dpFracs[dp] * outH;
                g.append('line')
                    .attr('x1', outX).attr('y1', ly).attr('x2', outX + outW).attr('y2', ly)
                    .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
            }

            outApi = {
                highlightTP(tp) {
                    outGridCells.forEach(c => {
                        const on = c.tp === tp;
                        c.el.attr('fill-opacity', on ? SHARD_HIGHLIGHT_OPACITY : SHARD_OPACITY);
                        if (on) c.el.attr('stroke', '#fff').attr('stroke-width', 1);
                        else c.el.attr('stroke', 'none');
                    });
                },
                highlightDP(dp) {
                    outGridCells.forEach(c => {
                        const on = c.dp === dp;
                        c.el.attr('fill-opacity', on ? SHARD_HIGHLIGHT_OPACITY : SHARD_OPACITY);
                        if (on) c.el.attr('stroke', '#fff').attr('stroke-width', 1);
                        else c.el.attr('stroke', 'none');
                    });
                },
                highlightCell(dp, tp) {
                    outGridCells.forEach(c => {
                        const on = c.dp === dp && c.tp === tp;
                        c.el.attr('fill-opacity', on ? SHARD_HIGHLIGHT_OPACITY : SHARD_OPACITY);
                        if (on) c.el.attr('stroke', '#fff').attr('stroke-width', 1.5);
                        else c.el.attr('stroke', 'none');
                    });
                },
                clear() {
                    outGridCells.forEach(c => c.el.attr('fill-opacity', SHARD_OPACITY).attr('stroke', 'none'));
                }
            };
        }
    } else {
        // 3D rendering
        const outCellCb = outHasParallelism ? (info) => {
            showTooltip(info, output, outTooltipX, outTooltipY);
            if (!inApi) return;
            if (!info) { inApi.clear(); return; }
            if (info.dp != null && info.tp != null) inApi.highlightCell(info.dp, info.tp);
            else if (info.dp != null) inApi.highlightDP(info.dp);
            else if (info.tp != null) inApi.highlightTP(info.tp);
            else inApi.clear();
        } : null;

        outApi = drawDetailBlock3D(g, outX, blockY, outW, outH, outD, blockColorFor(output), output.label,
            outGrp, tpInfoFor(output), dpInfoFor(output), outCellCb, output.color);
    }

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
    if (outShape.length >= 3 && !outFaceTp) {
        const depthLabel = outShape.length === 4
            ? `${outNames[0] || ''}\u00b7${outNames[1] || ''}=${outDepthVal}`
            : `${outNames[0] || ''}=${outShape[0]}`;
        g.append('text').attr('class', 'dim-label')
            .attr('x', outX + outW + outD * 0.7 / 2 + 6)
            .attr('y', blockY - outD * 0.4 / 2 - 2)
            .text(depthLabel);
    }

    // Dimension annotation
    let noteY = blockY + Math.max(inH, outH) + 32;
    if (op.type === 'broadcast') {
        // Broadcast: highlight which dimensions are repeated
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
    } else {
        // Reshape: show shape transformation (shapes are already labeled on blocks)
        const fmtShape = (sh, names) => '[' + sh.map((v, i) => `${names[i] || '?'}`).join(', ') + ']';
        g.append('text').attr('class', 'dim-label')
            .attr('x', mid).attr('y', noteY)
            .attr('text-anchor', 'middle').attr('fill', '#888')
            .text(`${fmtShape(inShape, inNames)} \u2192 ${fmtShape(outShape, outNames)}`);
        noteY += 18;
    }

    svg.attr('height', noteY + 10);
    baseSvgH = noteY + 10;
}
