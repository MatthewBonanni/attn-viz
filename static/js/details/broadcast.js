// broadcast.js — Broadcast detail visualization
import { detailMetrics, drawDetailBlock3D } from './shared.js';

export function drawBroadcastDetail(svg, op, tensorMap) {
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

    // Use a single scale factor across both shapes so aspect ratios are preserved.
    const maxBlockPx = 90;
    const allFaceDims = [
        inShape[inShape.length - 1], inShape[inShape.length - 2],
        outShape[outShape.length - 1], outShape[outShape.length - 2],
    ];
    const maxSqrt = Math.max(...allFaceDims.map(v => Math.sqrt(v)));
    const scaleFactor = maxSqrt > 0 ? maxBlockPx / maxSqrt : 7;
    const scale = (v) => Math.max(24, Math.min(maxBlockPx, Math.sqrt(v) * scaleFactor));

    // Input block dimensions
    const inW = scale(inShape[inShape.length - 1]);
    const inH = scale(inShape[inShape.length - 2]);
    const inDepthVal = inShape.length >= 3 ? inShape[0] * (inShape.length >= 4 ? inShape[1] : 1) : 1;
    const inD = Math.max(4, Math.min(100, Math.sqrt(inDepthVal) * 8));

    // Output block dimensions
    const outW = scale(outShape[outShape.length - 1]);
    const outH = scale(outShape[outShape.length - 2]);
    const outDepthVal = outShape.length >= 3 ? outShape[0] * (outShape.length >= 4 ? outShape[1] : 1) : 1;
    const outD = Math.max(4, Math.min(100, Math.sqrt(outDepthVal) * 8));

    // Center the two blocks with arrow in available width
    // Reshape/view ops need wider gap since input depth label + output left label crowd the arrow
    const arrowGap = 100;
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

    // Draw output block
    const outX = inX + inW + inD * 0.7 + arrowGap;
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
}
