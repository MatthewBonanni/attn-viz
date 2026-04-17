// rope.js — RoPE detail visualization
import { detailMetrics, drawDetailBlock3D, TP_COLORS, RANK_COLORS } from './shared.js';

export function drawRopeDetail(svg, op, tensorMap, params) {
    const { w: svgW, cx: svgCx } = detailMetrics();
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
    const bd = Math.max(4, Math.min(100, Math.sqrt(bDepthVal) * 8));

    const tensorBlockX = mid - bw / 2;
    const tensorBlockY = 20 + bd * 0.4;

    const tpSize = (params && params.tp_size) || 1;
    const dpSize = (params && params.dp_size) || 1;
    const isTp = input.tpSharded && tpSize > 1;
    const isDp = input.dpSharded && dpSize > 1;
    const SHARDED_NEUTRAL = '#f39c12';
    const NEUTRALIZE_COLORS = new Set(['#e74c3c', '#2ecc71', '#f39c12', '#e67e22']);
    const blockColor = (isTp || isDp) && NEUTRALIZE_COLORS.has(input.color) ? SHARDED_NEUTRAL : input.color;

    const ropeGrp = inShape.length === 4 ? { outer: inShape[0], inner: inShape[1] } : null;
    const tpInfo = isTp ? { tpSize } : null;
    const dpInfo = isDp ? { dpSize, dpFracs: input.dpBoundaryFracs } : null;
    const dpFracs = (input.dpBoundaryFracs) || Array.from({length: dpSize + 1}, (_, i) => i / dpSize);
    const B = (params && params.B) || 1;
    const d_h_param = params && params.d_h;
    const n_h = params && params.n_h;
    const n_kv = params && params.n_kv;

    let tooltipG = null;
    let baseSvgH = 0;
    function formatDimRange(dimName, start, end) {
        if (d_h_param && d_h_param > 1) {
            if (dimName === 'D' && n_h)
                return `n_h[${Math.floor(start / d_h_param)}\u2013${Math.floor(end / d_h_param)}] (${dimName}[${start}\u2013${end}])`;
            if (dimName.includes('\u00b7d_h') && n_kv)
                return `n_kv[${Math.floor(start / d_h_param)}\u2013${Math.floor(end / d_h_param)}] (${dimName}[${start}\u2013${end}])`;
        }
        return `${dimName}[${start}\u2013${end}]`;
    }

    function showTooltip(info) {
        if (tooltipG) tooltipG.remove();
        if (!info) { tooltipG = null; if (baseSvgH) svg.attr('height', baseSvgH); return; }
        tooltipG = g.append('g');

        const shape = inShape;
        const dimNames = inNames;
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
            const rowDim = shape.length >= 3 ? (dimNames[dimNames.length - 2] || 'rows') : (dimNames[0] || 'rows');
            const rowSize = shape.length >= 3 ? shape[shape.length - 2] : shape[0];
            const dpStart = Math.round(dpFracs[info.dp] * rowSize);
            const dpEnd = Math.round(dpFracs[info.dp + 1] * rowSize) - 1;
            parts.push(formatDimRange(rowDim, dpStart, dpEnd));
        }
        if (isTp && info.tp != null) {
            parts.push(`TP Rank ${info.tp}`);
            const tpDim = input.tpDim;
            if (tpDim != null) {
                const dimLabel = dimNames[tpDim] || `dim${tpDim}`;
                const perRank = Math.round(shape[tpDim] / tpSize);
                parts.push(formatDimRange(dimLabel, info.tp * perRank, (info.tp + 1) * perRank - 1));
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

    const hasParallelism = isTp || isDp;
    const onCellHover = hasParallelism ? (info) => showTooltip(info) : null;

    drawDetailBlock3D(g, tensorBlockX, tensorBlockY, bw, bh, bd, blockColor, input.label, ropeGrp, tpInfo, dpInfo, onCellHover, input.color);

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

    const truncated = d_h > dispDims;
    const ellipsisW = 20;
    const totalCells = dispDims + (truncated ? 2 : 0);
    const cellW = Math.min(28, (svgW - gPad * 2 - 20 - (truncated ? ellipsisW : 0)) / totalCells);
    const cellH = 26;
    const gridX = mid - (totalCells * cellW + (truncated ? ellipsisW : 0)) / 2;
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

    if (truncated) {
        const ellipsisX = gridX + dispDims * cellW;
        g.append('text').attr('class', 'dim-label')
            .attr('x', ellipsisX + ellipsisW / 2)
            .attr('y', gridY + cellH / 2 + 4)
            .attr('text-anchor', 'middle')
            .attr('fill', '#666')
            .text('\u2026');

        // Draw last pair
        const lastPairX = ellipsisX + ellipsisW;
        const lastPairIdx = numPairs - 1;
        const lastColor = pairColors[lastPairIdx % pairColors.length];
        for (let j = 0; j < 2; j++) {
            const cx = lastPairX + j * cellW;
            const idx = d_h - 2 + j;
            g.append('rect')
                .attr('x', cx).attr('y', gridY)
                .attr('width', cellW - 1).attr('height', cellH)
                .attr('rx', 2)
                .attr('fill', lastColor).attr('fill-opacity', 0.5)
                .attr('stroke', lastColor).attr('stroke-width', 1);
            g.append('text')
                .attr('x', cx + cellW / 2).attr('y', gridY + cellH / 2 + 4)
                .attr('text-anchor', 'middle')
                .attr('font-size', cellW >= 22 ? '10px' : '8px')
                .attr('fill', '#fff')
                .text(`x${idx}`);
        }
    }

    // Draw pair brackets underneath spanning both elements
    const bracketY = gridY + cellH + 6;
    for (let i = 0; i < dispPairs; i++) {
        const color = pairColors[i % pairColors.length];
        const x1 = gridX + i * 2 * cellW + 2;
        const x2 = gridX + (i * 2 + 2) * cellW - 3;
        const midBracketX = (x1 + x2) / 2;
        const bH = 10;

        g.append('path')
            .attr('d', `M${x1},${bracketY} L${x1},${bracketY + bH} L${x2},${bracketY + bH} L${x2},${bracketY}`)
            .attr('fill', 'none')
            .attr('stroke', color).attr('stroke-width', 1.5);

        g.append('text')
            .attr('x', midBracketX).attr('y', bracketY + bH + 12)
            .attr('text-anchor', 'middle')
            .attr('font-size', '9px').attr('fill', color)
            .text(`\u03b8${i}`);
    }

    if (truncated) {
        const lastPairIdx = numPairs - 1;
        const lastColor = pairColors[lastPairIdx % pairColors.length];
        const lastPairX = gridX + dispDims * cellW + ellipsisW;
        const x1 = lastPairX + 2;
        const x2 = lastPairX + 2 * cellW - 3;
        const midBracketX = (x1 + x2) / 2;
        const bH = 10;

        g.append('path')
            .attr('d', `M${x1},${bracketY} L${x1},${bracketY + bH} L${x2},${bracketY + bH} L${x2},${bracketY}`)
            .attr('fill', 'none')
            .attr('stroke', lastColor).attr('stroke-width', 1.5);

        g.append('text')
            .attr('x', midBracketX).attr('y', bracketY + bH + 12)
            .attr('text-anchor', 'middle')
            .attr('font-size', '9px').attr('fill', lastColor)
            .text(`\u03b8${lastPairIdx}`);
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
        .attr('font-size', '12px').attr('fill', '#ff7043').text('x\u2082\u1d62\u2032');
    g.append('text').attr('x', formulaX + 10).attr('y', y + 20)
        .attr('font-size', '12px').attr('fill', '#ff7043').text('x\u2082\u1d62\u208a\u2081\u2032');
    g.append('text')
        .attr('x', formulaX + 44).attr('y', y + 12)
        .attr('fill', '#aaa').attr('font-size', '13px')
        .text('] = [');

    // Rotation matrix
    const matX = formulaX + 72;
    g.append('text').attr('x', matX).attr('y', y + 5)
        .attr('font-size', '11px').attr('fill', '#888').text('cos m\u03b8\u1d62');
    g.append('text').attr('x', matX + 62).attr('y', y + 5)
        .attr('font-size', '11px').attr('fill', '#888').text('-sin m\u03b8\u1d62');
    g.append('text').attr('x', matX).attr('y', y + 20)
        .attr('font-size', '11px').attr('fill', '#888').text('sin m\u03b8\u1d62');
    g.append('text').attr('x', matX + 66).attr('y', y + 20)
        .attr('font-size', '11px').attr('fill', '#888').text('cos m\u03b8\u1d62');

    g.append('text')
        .attr('x', matX + 122).attr('y', y + 12)
        .attr('fill', '#aaa').attr('font-size', '13px')
        .text('][');

    // Input vector
    const vecX = matX + 134;
    g.append('text').attr('x', vecX).attr('y', y + 5)
        .attr('font-size', '12px').attr('fill', '#fff').text('x\u2082\u1d62');
    g.append('text').attr('x', vecX).attr('y', y + 20)
        .attr('font-size', '12px').attr('fill', '#fff').text('x\u2082\u1d62\u208a\u2081');
    g.append('text')
        .attr('x', vecX + 32).attr('y', y + 12)
        .attr('fill', '#aaa').attr('font-size', '13px')
        .text(']');

    y += 55;

    // --- Part 3: Rotation circle for a single pair ---
    g.append('text').attr('class', 'tensor-label')
        .attr('x', mid).attr('y', y)
        .attr('font-size', '13px')
        .text('Example: pair (x\u2080, x\u2081) at different positions');
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
        .text('m\u03b8\u2080');

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
        .text('\u03b8\u1d62 = 10000\u207b\u00b2\u2071/\u1d48 \u2014 low dims rotate fast, high dims slow');
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
    baseSvgH = y + 10;
}
