// tensor-shape.js — Tensor shape detail visualization
import { detailMetrics, drawDetailBlock, drawDetailBlock3D, TP_COLORS, RANK_COLORS } from './shared.js';

export function drawTensorShapeDetail(svg, tensor, params) {
    const { cx } = detailMetrics();
    const shape = tensor.shape;
    const dimNames = tensor.dimNames || [];
    const gPad = 20;
    const g = svg.append('g').attr('transform', `translate(${gPad}, 24)`);
    const mid = cx - gPad;

    const tpSize = (params && params.tp_size) || 1;
    const dpSize = (params && params.dp_size) || 1;
    const isTp = tensor.tpSharded && tpSize > 1;
    const isDp = tensor.dpSharded && dpSize > 1;
    const is4D = shape.length === 4 && tensor.type !== 'weight' && tensor.type !== 'mask';

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

    // Draw a proportional visual representation.
    // Scale dimensions so aspect ratios are preserved using sqrt scaling,
    // fitting within the available detail panel space.
    const maxBlockW = Math.min(300, mid * 1.2);
    const maxBlockH = 220;
    const minBlockDim = 32;

    // Given raw dimension values for [height, width], return scaled pixel
    // sizes that preserve the sqrt-ratio and fit within maxBlockH × maxBlockW.
    function detailScale2(rawH, rawW) {
        const sh = Math.sqrt(rawH), sw = Math.sqrt(rawW);
        // Find a single scale factor that fits both axes
        const factor = Math.min(maxBlockH / sh, maxBlockW / sw);
        return [Math.max(minBlockDim, sh * factor),
                Math.max(minBlockDim, sw * factor)];
    }

    if (shape.length === 2) {
        const [h, w] = detailScale2(shape[0], shape[1]);
        const x = mid - w / 2, y = 36;
        drawDetailBlock(g, x, y, w, h, tensor.color, tensor.label);

        const tpDim = tensor.tpDim;
        const dpFracs = (isDp && tensor.dpBoundaryFracs) || Array.from({length: dpSize + 1}, (_, i) => i / dpSize);
        const hasParallelism = (isTp && tpDim != null) || isDp;
        let cellInfoG2d = null;
        const cells2d = [];

        function show2dTooltip(info) {
            if (cellInfoG2d) cellInfoG2d.remove();
            if (!info) { cellInfoG2d = null; return; }
            cellInfoG2d = g.append('g');

            const parts = [];
            const rankIdx = (info.dp || 0) * Math.max(1, tpSize) + (info.tp || 0);
            const cellColor = (isTp && isDp) ? RANK_COLORS[rankIdx % RANK_COLORS.length]
                : isTp ? TP_COLORS[(info.tp || 0) % TP_COLORS.length]
                : RANK_COLORS[rankIdx % RANK_COLORS.length];

            if (isDp) {
                parts.push(`DP Rank ${info.dp}`);
                const B = params.B || 1;
                if (B > 1) {
                    const reqs = [];
                    for (let r = 0; r < B; r++) {
                        if (Math.floor(r * dpSize / B) === info.dp) reqs.push(r);
                    }
                    parts.push(`Req ${reqs.join(', ')}`);
                }
                const dpDimLabel = dimNames[0] || 'rows';
                const dpStart = Math.round(dpFracs[info.dp] * shape[0]);
                const dpEnd = Math.round(dpFracs[info.dp + 1] * shape[0]) - 1;
                parts.push(`${dpDimLabel} [${dpStart}\u2013${dpEnd}]`);
            }
            if (isTp && tpDim != null && info.tp != null) {
                parts.push(`TP Rank ${info.tp}`);
                const dimLabel = tpDim === 0 ? (dimNames[0] || 'rows') : (dimNames[1] || 'cols');
                const perRank = Math.round(shape[tpDim] / tpSize);
                parts.push(`${dimLabel} [${info.tp * perRank}\u2013${(info.tp + 1) * perRank - 1}]`);
            }
            if (isTp && isDp) {
                parts.unshift(`Rank ${rankIdx}`);
            }

            const padX = 10, padY = 8, lineH = 16;
            const boxW = 150, boxH = padY * 2 + parts.length * lineH;
            const ttX = x + w + 12, ttY = y + h * 0.2;

            cellInfoG2d.append('rect')
                .attr('x', ttX).attr('y', ttY)
                .attr('width', boxW).attr('height', boxH)
                .attr('rx', 6)
                .attr('fill', '#12141f').attr('fill-opacity', 0.95)
                .attr('stroke', cellColor).attr('stroke-width', 1.5).attr('stroke-opacity', 0.7);
            parts.forEach((text, i) => {
                cellInfoG2d.append('text')
                    .attr('x', ttX + padX).attr('y', ttY + padY + (i + 1) * lineH - 3)
                    .attr('fill', i === 0 ? cellColor : '#bbb')
                    .attr('font-size', '11px').attr('font-weight', i === 0 ? '600' : '400')
                    .text(text);
            });
        }

        function highlight2d(key) {
            cells2d.forEach(c => {
                const match = (key.dp == null || c.dp === key.dp) && (key.tp == null || c.tp === key.tp);
                if (match) {
                    c.el.attr('fill-opacity', 0.55).attr('stroke', '#fff').attr('stroke-width', 1.5);
                } else {
                    c.el.attr('fill-opacity', 0.3).attr('stroke', 'none');
                }
            });
        }

        function clear2d() {
            cells2d.forEach(c => c.el.attr('fill-opacity', 0.3).attr('stroke', 'none'));
        }

        if (hasParallelism) {
            const effDpSize = isDp ? dpSize : 1;
            const effTpSize = (isTp && tpDim != null) ? tpSize : 1;

            for (let dp = 0; dp < effDpSize; dp++) {
                for (let tp = 0; tp < effTpSize; tp++) {
                    const rankIdx = dp * effTpSize + tp;
                    const cellColor = (effTpSize > 1 && effDpSize > 1)
                        ? RANK_COLORS[rankIdx % RANK_COLORS.length]
                        : effTpSize > 1 ? TP_COLORS[tp % TP_COLORS.length]
                        : RANK_COLORS[rankIdx % RANK_COLORS.length];

                    let sx, sy, sw, sh;
                    if (effDpSize > 1 && effTpSize > 1) {
                        const dpY0 = dpFracs[dp] * h, dpY1 = dpFracs[dp + 1] * h;
                        if (tpDim === 1) {
                            sx = x + (tp / effTpSize) * w;
                            sy = y + dpY0;
                            sw = w / effTpSize;
                            sh = dpY1 - dpY0;
                        } else {
                            sx = x;
                            sy = y + dpY0 + (tp / effTpSize) * (dpY1 - dpY0);
                            sw = w;
                            sh = (dpY1 - dpY0) / effTpSize;
                        }
                    } else if (effTpSize > 1) {
                        if (tpDim === 1) {
                            sx = x + (tp / effTpSize) * w; sy = y; sw = w / effTpSize; sh = h;
                        } else {
                            sx = x; sy = y + (tp / effTpSize) * h; sw = w; sh = h / effTpSize;
                        }
                    } else {
                        const dpY0 = dpFracs[dp] * h, dpY1 = dpFracs[dp + 1] * h;
                        sx = x; sy = y + dpY0; sw = w; sh = dpY1 - dpY0;
                    }

                    const cell = g.append('rect')
                        .attr('x', sx).attr('y', sy).attr('width', sw).attr('height', sh)
                        .attr('fill', cellColor).attr('fill-opacity', 0.3).attr('stroke', 'none')
                        .style('cursor', 'pointer');
                    cells2d.push({ dp, tp, el: cell });

                    cell.on('mouseenter', () => {
                        highlight2d({ dp, tp });
                        show2dTooltip({ dp, tp });
                    });
                    cell.on('mouseleave', () => {
                        clear2d();
                        show2dTooltip(null);
                    });
                }
            }

            // TP boundary lines
            if (effTpSize > 1) {
                for (let tp = 1; tp < effTpSize; tp++) {
                    if (tpDim === 1) {
                        const lx = x + (tp / effTpSize) * w;
                        g.append('line')
                            .attr('x1', lx).attr('y1', y).attr('x2', lx).attr('y2', y + h)
                            .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
                    } else {
                        const ly = y + (tp / effTpSize) * h;
                        g.append('line')
                            .attr('x1', x).attr('y1', ly).attr('x2', x + w).attr('y2', ly)
                            .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
                    }
                }
            }

            // DP boundary lines
            if (effDpSize > 1) {
                for (let dp = 1; dp < effDpSize; dp++) {
                    const ly = y + dpFracs[dp] * h;
                    g.append('line')
                        .attr('x1', x).attr('y1', ly).attr('x2', x + w).attr('y2', ly)
                        .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
                }
            }
        }

        // Request boundary lines
        if (tensor.requestBoundaries && tensor.requestBoundaries.length > 0) {
            const totalH = tensor.requestBoundaryTotal || 1;
            for (const cumOffset of tensor.requestBoundaries) {
                const frac = cumOffset / totalH;
                const ly = y + frac * h;
                g.append('line')
                    .attr('x1', x).attr('y1', ly).attr('x2', x + w).attr('y2', ly)
                    .attr('stroke', '#e74c3c').attr('stroke-width', 0.75).attr('stroke-opacity', 0.5);
            }
        }

        g.append('text').attr('class', 'dim-label')
            .attr('x', x + w / 2).attr('y', y + h + 18)
            .attr('text-anchor', 'middle')
            .text(`${dimNames[1] || 'cols'} = ${shape[1]}`);
        g.append('text').attr('class', 'dim-label')
            .attr('x', x - 10).attr('y', y + h / 2 + 4)
            .attr('text-anchor', 'end')
            .text(`${dimNames[0] || 'rows'} = ${shape[0]}`);

        let noteY2d = y + h + 34;

        if (hasParallelism) {
            const effDpSize = isDp ? dpSize : 1;
            const effTpSize = (isTp && tpDim != null) ? tpSize : 1;
            const totalRanks = effDpSize * effTpSize;
            const legendCols = Math.min(totalRanks, 4);
            const legendRows = Math.ceil(totalRanks / legendCols);
            const colW = (mid * 2) / legendCols;

            if (isTp && tpDim != null) {
                const perRank = Math.round(shape[tpDim] / tpSize);
                const dimLabel = tpDim === 0 ? (dimNames[0] || 'rows') : (dimNames[1] || 'cols');
                const parallel = tpDim === 1 ? 'Column-parallel' : 'Row-parallel';
                g.append('text').attr('class', 'dim-label')
                    .attr('x', mid).attr('y', noteY2d)
                    .attr('text-anchor', 'middle').attr('fill', '#7c8cf8')
                    .text(`${parallel}: each TP rank holds ${dimLabel}=${perRank}`);
                noteY2d += 16;
            }

            if (isDp) {
                g.append('text').attr('class', 'dim-label')
                    .attr('x', mid).attr('y', noteY2d)
                    .attr('text-anchor', 'middle').attr('fill', '#7c8cf8')
                    .text(`Data-parallel: ${dpSize} ranks split ${dimNames[0] || 'rows'} dim`);
                noteY2d += 16;
            }

            for (let ri = 0; ri < totalRanks; ri++) {
                const col = ri % legendCols;
                const row = Math.floor(ri / legendCols);
                const dpRank = Math.floor(ri / effTpSize);
                const tpRank = ri % effTpSize;
                const rankColor = (effTpSize > 1 && effDpSize > 1)
                    ? RANK_COLORS[ri % RANK_COLORS.length]
                    : effTpSize > 1 ? TP_COLORS[tpRank % TP_COLORS.length]
                    : RANK_COLORS[ri % RANK_COLORS.length];
                const lx = col * colW + 12;
                const ly = noteY2d + row * 18;

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
                    highlight2d({ dp: dpRank, tp: tpRank });
                    show2dTooltip({ dp: dpRank, tp: tpRank });
                });
                legendItem.on('mouseleave', () => {
                    clear2d();
                    show2dTooltip(null);
                });
            }
            noteY2d += legendRows * 18 + 8;
        }

        if (tensor.type === 'weight') {
            g.append('text').attr('class', 'dim-label')
                .attr('x', mid).attr('y', noteY2d)
                .attr('text-anchor', 'middle').attr('fill', '#7c8cf8')
                .text('Learned weight matrix (dashed border)');
            noteY2d += 20;
        }

        const total2d = shape.reduce((a, b) => a * b, 1);
        g.append('text').attr('class', 'dim-label')
            .attr('x', mid).attr('y', noteY2d)
            .attr('text-anchor', 'middle').attr('fill', '#666')
            .text(`Total elements: ${total2d.toLocaleString()}`);
        noteY2d += 20;

        svg.attr('height', 24 + noteY2d);
    } else if (shape.length >= 3) {
        const rawW = shape[shape.length - 1];
        const rawH = shape[shape.length - 2];
        const depth = shape.length === 4 ? shape[0] * shape[1] : shape[0];
        const [h, w] = detailScale2(rawH, rawW);
        // Scale depth independently from face dimensions so that small depths
        // (e.g. B=1) look noticeably thinner than large ones (e.g. B·n_h=128).
        const d = Math.max(8, Math.min(120, Math.sqrt(depth) * 10));
        const dxTotal = d * 0.7;
        const dyTotal = -d * 0.4;
        const x = mid - w / 2, y = d * 0.4 + 36;

        const grp = shape.length === 4 ? { outer: shape[0], inner: shape[1] } : null;
        const tpInfo = isTp ? { tpSize } : null;
        const dpInfo = isDp ? { dpSize, dpFracs: tensor.dpBoundaryFracs } : null;

        // Floating tooltip for chunk info
        let cellInfoG = null;
        const tooltipX = x + w + d * 0.7 + 12;
        const tooltipY = y + h * 0.3;
        function showCellInfo(info) {
            if (cellInfoG) cellInfoG.remove();
            if (!info) { cellInfoG = null; return; }
            cellInfoG = g.append('g');
            const cellColor = RANK_COLORS[info.rankIdx % RANK_COLORS.length];
            const parts = [];
            if (dpSize > 1 && params.B > 1) {
                const B = params.B;
                const reqs = [];
                for (let r = 0; r < B; r++) {
                    if (Math.floor(r * dpSize / B) === info.dp) reqs.push(r);
                }
                parts.push(`Req ${reqs.join(', ')}`);
            }
            parts.push(`Rank ${info.rankIdx}`);
            if (dpSize > 1) parts.push(`DP ${info.dp}`);
            if (tpSize > 1 && info.tp != null) parts.push(`TP ${info.tp}`);

            const lineH = 16;
            const padX = 10, padY = 8;
            const boxW = 130;
            const boxH = padY * 2 + parts.length * lineH;

            cellInfoG.append('rect')
                .attr('x', tooltipX).attr('y', tooltipY)
                .attr('width', boxW).attr('height', boxH)
                .attr('rx', 6)
                .attr('fill', '#12141f').attr('fill-opacity', 0.95)
                .attr('stroke', cellColor).attr('stroke-width', 1.5)
                .attr('stroke-opacity', 0.7);

            parts.forEach((text, i) => {
                cellInfoG.append('text')
                    .attr('x', tooltipX + padX).attr('y', tooltipY + padY + (i + 1) * lineH - 3)
                    .attr('fill', i === 0 ? cellColor : '#bbb')
                    .attr('font-size', '11px')
                    .attr('font-weight', i === 0 ? '600' : '400')
                    .text(text);
            });
        }

        drawDetailBlock3D(g, x, y, w, h, d, tensor.color, tensor.label, grp, tpInfo, dpInfo,
            (isTp || isDp) ? showCellInfo : null);

        // Request boundary lines (when B > 1)
        if (tensor.requestBoundaries && tensor.requestBoundaries.length > 0) {
            const totalH = tensor.requestBoundaryTotal || 1;
            const dxTotal = d * 0.7;
            const dyTotal = -d * 0.4;
            for (const cumOffset of tensor.requestBoundaries) {
                const frac = cumOffset / totalH;
                const ly = y + frac * h;
                g.append('line')
                    .attr('x1', x).attr('y1', ly)
                    .attr('x2', x + w).attr('y2', ly)
                    .attr('stroke', '#e74c3c').attr('stroke-width', 0.75)
                    .attr('stroke-opacity', 0.5);
                if (d > 0) {
                    g.append('line')
                        .attr('x1', x + w).attr('y1', ly)
                        .attr('x2', x + w + dxTotal).attr('y2', ly + dyTotal)
                        .attr('stroke', '#e74c3c').attr('stroke-width', 0.75)
                        .attr('stroke-opacity', 0.35);
                }
            }
        }

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

        // KV cache "new" portion highlight (when S_q < S)
        if (tensor.cache && params && params.S_q < params.S) {
            const seqDimIdx = (dimNames || []).indexOf('S');
            if (seqDimIdx >= 0) {
                const newFrac = params.S_q / params.S;
                const nh = h * newFrac;
                const ny = y + h - nh;

                // Front face
                g.append('rect')
                    .attr('x', x).attr('y', ny)
                    .attr('width', w).attr('height', nh)
                    .attr('fill', '#fff').attr('fill-opacity', 0.12)
                    .attr('stroke', '#fff').attr('stroke-width', 1.5)
                    .attr('stroke-dasharray', '4,2').attr('stroke-opacity', 0.5)
                    .attr('rx', 1);

                // Right face (if 3D)
                if (d > 0) {
                    const pts = [
                        [x + w, ny],
                        [x + w + dxTotal, ny + dyTotal],
                        [x + w + dxTotal, ny + nh + dyTotal],
                        [x + w, ny + nh],
                    ].map(p => p.join(',')).join(' ');
                    g.append('polygon')
                        .attr('points', pts)
                        .attr('fill', '#fff').attr('fill-opacity', 0.10)
                        .attr('stroke', '#fff').attr('stroke-width', 1.5)
                        .attr('stroke-dasharray', '4,2').attr('stroke-opacity', 0.4);
                }

                // Label + bracket
                const labelSide = x + w + (d > 0 ? dxTotal : 0) + 8;
                g.append('text').attr('class', 'dim-label')
                    .attr('x', labelSide).attr('y', ny + nh / 2 + 3)
                    .attr('text-anchor', 'start').attr('fill', '#aaa').attr('font-size', '9px')
                    .text(`new (S_q=${params.S_q})`);
                g.append('line')
                    .attr('x1', labelSide - 3).attr('y1', ny)
                    .attr('x2', labelSide - 3).attr('y2', ny + nh)
                    .attr('stroke', '#888').attr('stroke-width', 1);
                g.append('line')
                    .attr('x1', labelSide - 3).attr('y1', ny)
                    .attr('x2', labelSide - 6).attr('y2', ny)
                    .attr('stroke', '#888').attr('stroke-width', 1);
                g.append('line')
                    .attr('x1', labelSide - 3).attr('y1', ny + nh)
                    .attr('x2', labelSide - 6).attr('y2', ny + nh)
                    .attr('stroke', '#888').attr('stroke-width', 1);
            }
        }

        let noteY = y + h + 40;

        // Rank legend for TP/DP
        if (isTp || isDp) {
            const totalRanks = Math.max(1, dpSize) * Math.max(1, tpSize);
            const legendCols = Math.min(totalRanks, 4);
            const legendRows = Math.ceil(totalRanks / legendCols);
            const colW = (mid * 2) / legendCols;

            for (let ri = 0; ri < totalRanks; ri++) {
                const col = ri % legendCols;
                const row = Math.floor(ri / legendCols);
                const rankColor = RANK_COLORS[ri % RANK_COLORS.length];
                const lx = col * colW + 12;
                const ly = noteY + row * 18;
                const dpRank = Math.floor(ri / Math.max(1, tpSize));
                const tpRank = ri % Math.max(1, tpSize);

                const legendItem = g.append('g').style('cursor', 'pointer');
                legendItem.append('rect')
                    .attr('x', lx - 2).attr('y', ly - 10)
                    .attr('width', colW - 4).attr('height', 16)
                    .attr('fill', 'transparent');

                legendItem.append('circle')
                    .attr('cx', lx + 4).attr('cy', ly - 3)
                    .attr('r', 5).attr('fill', rankColor).attr('fill-opacity', 0.7);

                let label = `R${ri}`;
                if (dpSize > 1 && tpSize > 1) label += ` (DP${dpRank},TP${tpRank})`;
                else if (dpSize > 1) label += ` DP${dpRank}`;
                else if (tpSize > 1) label += ` TP${tpRank}`;

                legendItem.append('text').attr('class', 'dim-label')
                    .attr('x', lx + 12).attr('y', ly)
                    .attr('text-anchor', 'start').attr('fill', rankColor).attr('font-size', '10px')
                    .text(label);

                legendItem.on('mouseenter', () => {
                    showCellInfo({ dp: dpRank, tp: tpRank, rankIdx: ri });
                });
                legendItem.on('mouseleave', () => {
                    showCellInfo(null);
                });
            }
            noteY += legendRows * 18 + 8;
        }

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
