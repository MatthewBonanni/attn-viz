// add.js — Elementwise sum of the content and positional score matrices.
//
// The point of this panel is that the add is an algebraic identity, not a kernel:
// (q_c·k_cᵀ) + (q_r·k_rᵀ) = [q_c | q_r] · [k_c | k_r]ᵀ, so a fused backend runs a
// single GEMM at QK headdim P+R and never materializes either half.
import { detailMetrics, drawDetailBlock, drawDetailBlock3D } from './shared.js';

const SCORE = '#9b59b6', CONTENT = '#e74c3c', ROPE = '#ff7043';

export function drawAddDetail(svg, op, tensorMap, params) {
    const { w: svgW } = detailMetrics();
    const A = tensorMap[op.inputs[0]];
    const Bm = tensorMap[op.inputs[1]];
    const C = tensorMap[op.output];
    if (!A || !Bm || !C) return;

    // Content/RoPE halves — absorbed (q', c_kv) or up-projected (q_nope, k_nope)
    const contentQ = tensorMap['q_lat'] || tensorMap['q_nope'];
    const ropeQ = tensorMap['q_r'] || tensorMap['q_pe'];
    const contentK = tensorMap['k_nope'] || tensorMap['c_kv_sel'] || tensorMap['c_KV'];
    const ropeK = tensorMap['k_r_sel'] || tensorMap['k_r'];
    const lastDim = (t, fallback) => (t ? t.shape[t.shape.length - 1] : fallback);
    const pDim = lastDim(contentQ, params.d_c || params.d_h);
    const rDim = lastDim(ropeQ, params.d_r || 64);
    const absorbed = !!tensorMap['q_lat'];
    const pName = absorbed ? 'd_c' : 'P';
    const rName = absorbed ? 'd_r' : 'R';

    const pad = 20, innerW = svgW - pad * 2;
    const g = svg.append('g').attr('transform', `translate(${pad}, 22)`);
    let y = 0;

    // --- A + B = C ---
    g.append('text').attr('class', 'tensor-label')
        .attr('x', innerW / 2).attr('y', y)
        .text('Content scores + positional scores');
    y += 16;
    const rowsLabel = C.dimNames ? C.dimNames.slice(-2).join(' × ') : 'S_q × S';
    g.append('text')
        .attr('x', innerW / 2).attr('y', y)
        .attr('text-anchor', 'middle').attr('font-size', '9px').attr('fill', '#888')
        .text(`elementwise over all ${C.shape.join(' × ')} entries`);
    y += 26;

    const bw = 100, bh = 66, depth = 16;
    const xs = [20, 170, 320];
    const blocks = [[A, xs[0]], [Bm, xs[1]], [C, xs[2]]];
    for (const [t, bx] of blocks) {
        g.append('text').attr('class', 'tensor-label')
            .attr('x', bx + bw / 2).attr('y', y - 8)
            .text(t.label);
        drawDetailBlock3D(g, bx, y, bw, bh, depth, t.color || SCORE, '');
        g.append('text')
            .attr('x', bx + bw / 2).attr('y', y + bh + 13)
            .attr('text-anchor', 'middle').attr('font-size', '9px').attr('fill', '#999')
            .text(rowsLabel);
    }
    for (const [glyph, gx] of [['+', (xs[0] + bw + xs[1]) / 2], ['=', (xs[1] + bw + xs[2]) / 2]]) {
        g.append('text')
            .attr('x', gx).attr('y', y + bh / 2 + 6)
            .attr('text-anchor', 'middle').attr('font-size', '17px').attr('fill', '#aaa')
            .text(glyph);
    }
    y += bh + 34;

    // --- The identity that removes the add ---
    g.append('text').attr('class', 'tensor-label')
        .attr('x', innerW / 2).attr('y', y)
        .text('...which a fused kernel never computes');
    y += 22;

    const catW = 116, catH = 34;
    const pFrac = pDim / (pDim + rDim);
    const qX = 14, kX = 178, sX = 350, sW = 74;

    // Row counts come from the score tensor's own axes, so DSA's k shows as k
    const cDims = C.dimNames || [];
    const qAxis = cDims[cDims.length - 2] || 'S_q';
    const kAxis = cDims[cDims.length - 1] || 'S';

    for (const [bx, cT, rT, cFallback, rFallback, rowsName] of [
        [qX, contentQ, ropeQ, 'q_c', 'q_r', qAxis],
        [kX, contentK, ropeK, 'k_c', 'k_r', kAxis],
    ]) {
        const cColor = cT?.color || CONTENT, rColor = rT?.color || ROPE;
        const pw = catW * pFrac;
        drawDetailBlock(g, bx, y, pw, catH, cColor, '');
        drawDetailBlock(g, bx + pw, y, catW - pw, catH, rColor, '');
        g.append('text')
            .attr('x', bx + pw / 2).attr('y', y + catH / 2 + 4)
            .attr('text-anchor', 'middle').attr('class', 'tensor-label')
            .text(cT ? cT.label : cFallback);
        if (catW - pw > 22) {
            g.append('text')
                .attr('x', bx + pw + (catW - pw) / 2).attr('y', y + catH / 2 + 4)
                .attr('text-anchor', 'middle').attr('class', 'tensor-label')
                .text(rT ? rT.label : rFallback);
        }
        // Concatenated extent — the split is spelled out, so it stays readable
        // even when one half is a sliver of the other
        g.append('line')
            .attr('x1', bx).attr('y1', y + catH + 7).attr('x2', bx + catW).attr('y2', y + catH + 7)
            .attr('stroke', '#666').attr('stroke-width', 0.75);
        const cap = g.append('text')
            .attr('x', bx + catW / 2).attr('y', y + catH + 19)
            .attr('text-anchor', 'middle').attr('font-size', '9px').attr('fill', '#999');
        cap.append('tspan').text(`${rowsName} × (`);
        cap.append('tspan').attr('fill', cColor).text(pDim);
        cap.append('tspan').text(' + ');
        cap.append('tspan').attr('fill', rColor).text(rDim);
        cap.append('tspan').text(')');
    }

    for (const [glyph, gx, size] of [
        ['×', (qX + catW + kX) / 2, 13],
        ['ᵀ', kX + catW + 3, 11],
        ['→', (kX + catW + sX) / 2 + 6, 13],
    ]) {
        g.append('text')
            .attr('x', gx).attr('y', y + catH / 2 + (glyph === 'ᵀ' ? -6 : 5))
            .attr('text-anchor', glyph === 'ᵀ' ? 'start' : 'middle')
            .attr('font-size', size + 'px').attr('fill', '#aaa')
            .text(glyph);
    }
    drawDetailBlock(g, sX, y, sW, catH, SCORE, C.label);
    g.append('text')
        .attr('x', sX + sW / 2).attr('y', y + catH + 19)
        .attr('text-anchor', 'middle').attr('font-size', '9px').attr('fill', '#999')
        .text('one GEMM');
    y += catH + 38;

    const notes = [
        `Concatenating along the head dim folds the sum into the matmul: (q·k_cᵀ) + (q_r·k_rᵀ) = [q | q_r] · [k_c | k_r]ᵀ.`,
        `One GEMM at QK headdim ${pName} + ${rName} = ${pDim} + ${rDim} = ${pDim + rDim}, and neither half of the scores is ever written out.`,
        `The cost below prices the explicit add instead: two full reads and one write of ${C.shape.join(' × ')}.`,
    ];
    for (const [i, note] of notes.entries()) {
        for (const [j, line] of wrap(note, 84).entries()) {
            g.append('text').attr('x', 0).attr('y', y + j * 12)
                .attr('font-size', '9px').attr('fill', i === 2 ? '#6b7280' : '#8a8f9e')
                .attr('font-style', 'italic').text(line);
        }
        y += wrap(note, 84).length * 12 + 4;
    }

    svg.attr('height', y + 26);
}

function wrap(text, perLine) {
    const lines = [];
    let line = '';
    for (const word of text.split(' ')) {
        if (line && (line.length + 1 + word.length) > perLine) { lines.push(line); line = word; }
        else line = line ? `${line} ${word}` : word;
    }
    if (line) lines.push(line);
    return lines;
}
