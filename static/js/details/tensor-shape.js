// tensor-shape.js — Tensor shape detail visualization
import { detailMetrics, drawDetailBlock, drawDetailBlock3D, TP_COLORS } from './shared.js';

export function drawTensorShapeDetail(svg, tensor, params) {
    const { cx } = detailMetrics();
    const shape = tensor.shape;
    const dimNames = tensor.dimNames || [];
    const gPad = 20;
    const g = svg.append('g').attr('transform', `translate(${gPad}, 24)`);
    const mid = cx - gPad;

    const tpSize = (params && params.tp_size) || 1;
    const isTp = tensor.tpSharded && tpSize > 1;
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
    const maxBlockW = Math.min(200, mid * 0.9);
    const maxBlockH = 140;
    const minBlockDim = 24;

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
        const rawW = shape[shape.length - 1];
        const rawH = shape[shape.length - 2];
        const depth = shape.length === 4 ? shape[0] * shape[1] : shape[0];
        const [h, w] = detailScale2(rawH, rawW);
        // Scale depth proportionally: use same factor as the larger face axis,
        // then cap so the isometric offset doesn't dominate the panel.
        const maxFaceSqrt = Math.max(Math.sqrt(rawH), Math.sqrt(rawW));
        const faceFactor = maxFaceSqrt > 0 ? Math.max(w, h) / maxFaceSqrt : 1;
        const d = Math.max(16, Math.min(50, Math.sqrt(depth) * faceFactor * 0.35));
        const dxTotal = d * 0.7;
        const dyTotal = -d * 0.4;
        const x = mid - w / 2, y = d * 0.4 + 36;

        const grp = shape.length === 4 ? { outer: shape[0], inner: shape[1] } : null;
        const tpInfo = (is4D && isTp) ? { tpSize } : null;
        drawDetailBlock3D(g, x, y, w, h, d, tensor.color, tensor.label, grp, tpInfo);

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

        // Batch braces and TP rank annotations for 4D + TP tensors
        if (is4D && isTp) {
            const B = shape[0];
            const n_h = shape[1];
            const headsPerRank = n_h / tpSize;
            const total = B * n_h;

            // Batch braces along the right face
            for (let b = 0; b < B; b++) {
                const f0 = (b * n_h) / total;
                const f1 = ((b + 1) * n_h) / total;
                const fMid = (f0 + f1) / 2;
                // Right face midpoint for this batch
                const braceX = x + w + dxTotal * fMid + 8;
                const braceYTop = y + h + dyTotal * f0;
                const braceYBot = y + h + dyTotal * f1;
                const braceYMid = (braceYTop + braceYBot) / 2;

                // Bracket lines
                g.append('path')
                    .attr('d', `M${x + w + dxTotal * f0},${braceYTop} L${braceX},${braceYTop} L${braceX},${braceYBot} L${x + w + dxTotal * f1},${braceYBot}`)
                    .attr('fill', 'none').attr('stroke', '#aaa').attr('stroke-width', 1);
                // Tick at midpoint
                g.append('line')
                    .attr('x1', braceX).attr('y1', braceYMid)
                    .attr('x2', braceX + 6).attr('y2', braceYMid)
                    .attr('stroke', '#aaa').attr('stroke-width', 1);
                g.append('text').attr('class', 'dim-label')
                    .attr('x', braceX + 10).attr('y', braceYMid + 3)
                    .attr('text-anchor', 'start').attr('fill', '#aaa').attr('font-size', '9px')
                    .text(`B${b}`);
            }

            // TP rank arrows pointing to stripes on right face
            for (let r = 0; r < tpSize; r++) {
                const tpColor = TP_COLORS[r % TP_COLORS.length];
                // Point to the first occurrence (batch 0) of this rank's stripe
                const sliceMid = (r * headsPerRank + (r + 1) * headsPerRank) / 2 / total;
                const arrowTargetX = x + w + dxTotal * sliceMid;
                const arrowTargetY = y + h / 2 + dyTotal * sliceMid;
                // Arrow starts from the left, pointing right to the stripe
                const arrowStartX = arrowTargetX - 30;
                const arrowStartY = arrowTargetY;

                g.append('line')
                    .attr('x1', arrowStartX).attr('y1', arrowStartY)
                    .attr('x2', arrowTargetX - 2).attr('y2', arrowTargetY)
                    .attr('stroke', tpColor).attr('stroke-width', 1.5)
                    .attr('marker-end', 'url(#arrowhead)');
                g.append('text').attr('class', 'dim-label')
                    .attr('x', arrowStartX - 4).attr('y', arrowStartY + 3)
                    .attr('text-anchor', 'end').attr('fill', tpColor).attr('font-size', '9px')
                    .text(`Rank ${r}`);
            }

            // TP rank legend below the block
            const legendY = noteY;
            const legendItemW = mid * 2 / tpSize;
            for (let r = 0; r < tpSize; r++) {
                const tpColor = TP_COLORS[r % TP_COLORS.length];
                const lx = r * legendItemW + legendItemW / 2;
                g.append('circle')
                    .attr('cx', lx - 16).attr('cy', legendY - 3)
                    .attr('r', 5).attr('fill', tpColor).attr('fill-opacity', 0.7);
                g.append('text').attr('class', 'dim-label')
                    .attr('x', lx - 8).attr('y', legendY)
                    .attr('text-anchor', 'start').attr('fill', tpColor).attr('font-size', '10px')
                    .text(`Rank ${r}`);
            }
            noteY += 24;
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
