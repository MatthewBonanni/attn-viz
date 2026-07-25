// relu-wsum.js — DSA lightning-indexer head reduction:
//   I[t,s] = Σ_h w[t,h] · ReLU(q_I'[t,h] · k_I[s])
//
// Two parts: the shape story (an [n_i, S_q, S] score cube collapsing to a single
// [S_q, S] relevance map), then the per-head arithmetic for one example
// (query, key) cell — where the ReLU gate is visible as heads contributing zero.
import { detailMetrics, drawDetailBlock, drawDetailBlock3D } from './shared.js';

const SCORE_POS = '#9b59b6', SCORE_NEG = '#e74c3c';
const WEIGHT = '#f39c12', CONTRIB = '#f1c40f', GATED = '#3d4354';

// Deterministic pseudo-random in [0, 1) — same hash as mask.js/topk.js
function hash01(a, b) {
    let h = (a * 374761 + b * 668265) ^ 0x5bd1e995;
    h = Math.imul(h ^ (h >>> 13), 0x5bd1e995);
    h = h ^ (h >>> 15);
    return (h >>> 0) / 0xffffffff;
}

// Column geometry for the per-head table (inner width 440)
const COL = {
    head: 0,
    scoreCenter: 84, scoreHalf: 50,
    gate: 160,
    wX: 186, wW: 60,
    cX: 258, cW: 142, cVal: 406,
    right: 438,
};

export function drawReluWsumDetail(svg, op, tensorMap, params) {
    const { w: svgW } = detailMetrics();
    const scoresT = tensorMap[op.inputs[0]];
    const outT = tensorMap[op.output];

    const n_i = (scoresT && scoresT.shape.length === 3) ? scoresT.shape[0] : (params.n_i || 64);
    const S = params.seqLens?.[0] ?? params.S ?? 1;
    const S_q = params.queryLens?.[0] ?? params.S_q ?? 1;
    const queryOffset = Math.max(0, S - S_q);

    // Example cell: a mid-sequence query and a causal key well inside its window
    const t = queryOffset + Math.floor((S_q - 1) / 2);
    const s = Math.min(t, Math.max(0, Math.floor(t * 0.6)));
    const seed = t * 7919 + s;

    const score = [], wgt = [], contrib = [];
    let total = 0, gated = 0;
    for (let h = 0; h < n_i; h++) {
        const sc = -0.9 + hash01(h, seed) * 2.0;
        const w = 0.08 + hash01(h + 911, seed) * 0.92;
        const c = w * Math.max(0, sc);
        score.push(sc); wgt.push(w); contrib.push(c);
        total += c;
        if (sc <= 0) gated++;
    }

    const pad = 20;
    const innerW = svgW - pad * 2;
    const g = svg.append('g').attr('transform', `translate(${pad}, 20)`);
    let y = 0;

    y = drawReduction(g, y, innerW, { scoresT, outT, n_i, S, S_q, t, s, queryOffset, B: params.B });

    // --- Per-head breakdown for the marked cell ---
    // The white square echoes the marker on both blocks above
    g.append('text').attr('class', 'tensor-label')
        .attr('x', innerW / 2).attr('y', y)
        .text(`▪ Inside one cell — query t=${t}, key s=${s}`);
    y += 15;
    g.append('text')
        .attr('x', innerW / 2).attr('y', y)
        .attr('text-anchor', 'middle').attr('font-size', '10px').attr('fill', '#999')
        .text("I[t,s] = Σₕ w[t,h] · ReLU(q_I'[t,h] · k_I[s])");
    y += 22;

    const HEADS = Math.min(n_i, 8);
    const scoreMax = Math.max(...score.slice(0, HEADS).map(Math.abs)) || 1;
    const wMax = Math.max(...wgt.slice(0, HEADS)) || 1;
    const contribMax = Math.max(...contrib.slice(0, HEADS)) || 1;

    y = drawHeadTable(g, y, { HEADS, score, wgt, contrib, scoreMax, wMax, contribMax, n_i });
    y = drawTotalBar(g, y, { HEADS, contrib, total, n_i, t, s });

    // --- Footnotes ---
    const notes = [
        [`${gated} of ${n_i} heads score ≤ 0 here and contribute exactly 0 — a head can vote "relevant", never "irrelevant".`, '#8a8f9e'],
        [`Every one of the ${S_q === 1 ? S : `${S_q}×${S}`} cells in I is reduced this way: ${n_i} ReLUs, multiplies and adds each.`, '#8a8f9e'],
        [`Fused in practice (e.g. DeepGEMM's fp8_mqa_logits), so the [${n_i}, S_q, S] cube never reaches HBM — the cost below is the unfused upper bound.`, '#6b7280'],
    ];
    for (const [note, fill] of notes) {
        y += 15;
        y += wrapText(g, 0, y, COL.right, note, fill);
    }

    svg.attr('height', y + 34);
}

// --- Part 1: [n_i, S_q, S] → Σ over heads → [S_q, S] ---

function drawReduction(g, y, innerW, { scoresT, outT, n_i, S, S_q, t, s, queryOffset, B }) {
    g.append('text').attr('class', 'tensor-label')
        .attr('x', innerW / 2).attr('y', y)
        .text('Head reduction: one relevance score per (query, key)');

    const bw = 104, bh = 62, depth = 24;
    const inX = 26, outX = innerW - bw - 26;
    const by = y + 42;

    // Labels sit outside the faces so the marked cell stays legible
    drawDetailBlock3D(g, inX, by, bw, bh, depth, SCORE_POS, '');
    drawDetailBlock(g, outX, by, bw, bh, SCORE_POS, '');

    const dim = (x, yy, text, anchor, fill) => g.append('text')
        .attr('x', x).attr('y', yy).attr('text-anchor', anchor || 'middle')
        .attr('font-size', '9px').attr('fill', fill || '#999').text(text);

    for (const [bx, tensor, fallback, shape] of [
        [inX, scoresT, 'scores', `[${n_i} × S_q × S]`],
        [outX, outT, 'I', '[S_q × S]'],
    ]) {
        g.append('text').attr('class', 'tensor-label')
            .attr('x', bx + bw / 2).attr('y', by - 14)
            .text(tensor ? tensor.label : fallback);
        dim(bx + bw / 2, by + bh + 13, shape);
    }
    dim(inX + bw + depth * 0.7 + 4, by - depth * 0.4 - 4, `n_i=${n_i}`, 'start');
    dim(innerW / 2, by + bh + 26, S_q === S
        ? `S_q = S = ${S}${B > 1 ? ` (request 0 of B=${B})` : ''}`
        : `S_q = ${S_q}, S = ${S}${B > 1 ? ` (request 0 of B=${B})` : ''}`, 'middle', '#777');

    // Marked cell (t, s) on both blocks
    const fx = Math.min(0.9, Math.max(0.06, S > 1 ? s / (S - 1) : 0.5));
    const fy = Math.min(0.9, Math.max(0.06, S_q > 1 ? (t - queryOffset) / (S_q - 1) : 0.5));
    const markX = (bx) => bx + 4 + fx * (bw - 12);
    const markY = by + 4 + fy * (bh - 12);
    for (const bx of [inX, outX]) {
        g.append('rect')
            .attr('x', markX(bx) - 3.5).attr('y', markY - 3.5)
            .attr('width', 7).attr('height', 7)
            .attr('fill', '#fff').attr('fill-opacity', 0.95)
            .attr('stroke', '#0f1117').attr('stroke-width', 0.75);
    }

    // Σ node between the blocks
    const cx = (inX + bw + outX) / 2, cy = by + bh / 2;
    for (const [x1, x2] of [[inX + bw + depth * 0.7 + 6, cx - 16], [cx + 16, outX - 6]]) {
        g.append('line')
            .attr('x1', x1).attr('y1', cy).attr('x2', x2).attr('y2', cy)
            .attr('stroke', '#666').attr('stroke-width', 1.5)
            .attr('marker-end', 'url(#arrowhead)');
    }
    g.append('circle')
        .attr('cx', cx).attr('cy', cy).attr('r', 13)
        .attr('fill', '#1e2030').attr('stroke', CONTRIB).attr('stroke-width', 2);
    g.append('text')
        .attr('x', cx).attr('y', cy + 5)
        .attr('text-anchor', 'middle').attr('font-size', '14px').attr('fill', CONTRIB)
        .text('Σ');
    g.append('text')
        .attr('x', cx).attr('y', cy + 28)
        .attr('text-anchor', 'middle').attr('font-size', '9px').attr('fill', '#888')
        .text('ReLU, ×w, Σₕ');

    return by + bh + 48;
}

// --- Part 2: per-head score → gate → weight → contribution ---

function drawHeadTable(g, y, { HEADS, score, wgt, contrib, scoreMax, wMax, contribMax, n_i }) {
    const hdr = (x, text, anchor) => g.append('text')
        .attr('x', x).attr('y', y).attr('text-anchor', anchor || 'middle')
        .attr('font-size', '8px').attr('fill', '#777').text(text);
    hdr(COL.head, 'head', 'start');
    hdr(COL.scoreCenter, "q_I'·k_I");
    hdr(COL.gate, 'ReLU');
    hdr(COL.wX + COL.wW / 2, 'w[t,h]');
    hdr(COL.cX, 'w · ReLU(score)', 'start');

    // Mini hinge plot under the ReLU header — the gate, as a function
    const hy = y + 17, hx = COL.gate;
    g.append('polyline')
        .attr('points', `${hx - 12},${hy} ${hx},${hy} ${hx + 11},${hy - 11}`)
        .attr('fill', 'none').attr('stroke', CONTRIB).attr('stroke-width', 1.3);
    g.append('polyline')
        .attr('points', `${hx - 13},${hy - 12} ${hx - 13},${hy + 2} ${hx + 13},${hy + 2}`)
        .attr('fill', 'none').attr('stroke', '#4a4d5a').attr('stroke-width', 0.6);

    const rowH = 17;
    const top = y + 26;

    // Zero axis for the signed score column
    g.append('line')
        .attr('x1', COL.scoreCenter).attr('y1', top)
        .attr('x2', COL.scoreCenter).attr('y2', top + HEADS * rowH)
        .attr('stroke', '#4a4d5a').attr('stroke-width', 0.75);

    for (let h = 0; h < HEADS; h++) {
        const yr = top + h * rowH;
        const yc = yr + rowH / 2;
        const sc = score[h], w = wgt[h], c = contrib[h];
        const off = sc <= 0;

        if (off) {
            g.append('rect')
                .attr('x', -4).attr('y', yr).attr('width', COL.right + 8).attr('height', rowH)
                .attr('fill', '#1a1d2a').attr('rx', 2);
        }

        g.append('text')
            .attr('x', COL.head).attr('y', yc + 3)
            .attr('font-size', '9px').attr('fill', off ? '#5a5f70' : '#999')
            .text(`h${h}`);

        // Signed score bar
        const len = Math.abs(sc) / scoreMax * COL.scoreHalf;
        g.append('rect')
            .attr('x', sc >= 0 ? COL.scoreCenter : COL.scoreCenter - len)
            .attr('y', yc - 4.5).attr('width', Math.max(len, 2)).attr('height', 9).attr('rx', 1)
            .attr('fill', sc >= 0 ? SCORE_POS : SCORE_NEG)
            .attr('fill-opacity', sc >= 0 ? 0.85 : 0.5);

        // Gate marker
        g.append('text')
            .attr('x', COL.gate).attr('y', yc + 3.5)
            .attr('text-anchor', 'middle').attr('font-size', off ? '10px' : '9px')
            .attr('fill', off ? GATED : '#777')
            .style('font-weight', off ? '700' : '400')
            .text(off ? '0' : '→');

        // Head weight
        g.append('rect')
            .attr('x', COL.wX).attr('y', yc - 4.5)
            .attr('width', Math.max(w / wMax * COL.wW, 1)).attr('height', 9).attr('rx', 1)
            .attr('fill', WEIGHT).attr('fill-opacity', off ? 0.2 : 0.8);

        // Contribution
        if (off) {
            g.append('line')
                .attr('x1', COL.cX).attr('y1', yc).attr('x2', COL.cX + 5).attr('y2', yc)
                .attr('stroke', GATED).attr('stroke-width', 2);
            g.append('text')
                .attr('x', COL.cX + 11).attr('y', yc + 3)
                .attr('font-size', '8px').attr('fill', '#5a5f70').text('gated off');
        } else {
            g.append('rect')
                .attr('x', COL.cX).attr('y', yc - 4.5)
                .attr('width', Math.max(c / contribMax * COL.cW, 1)).attr('height', 9).attr('rx', 1)
                .attr('fill', CONTRIB).attr('fill-opacity', 0.85);
            g.append('text')
                .attr('x', COL.cVal).attr('y', yc + 3)
                .attr('font-size', '8px').attr('fill', '#bbb').text(c.toFixed(2));
        }
    }

    let yNext = top + HEADS * rowH;
    if (n_i > HEADS) {
        yNext += 13;
        g.append('text')
            .attr('x', COL.head).attr('y', yNext)
            .attr('font-size', '9px').attr('fill', '#666')
            .text(`+ ${n_i - HEADS} more heads, same shape of arithmetic`);
        yNext += 6;
    }
    return yNext + 12;
}

// --- Part 3: the sum itself ---

function drawTotalBar(g, y, { HEADS, contrib, total, n_i, t, s }) {
    g.append('line')
        .attr('x1', 0).attr('y1', y).attr('x2', COL.right).attr('y2', y)
        .attr('stroke', '#2a2d3a').attr('stroke-width', 1);
    y += 20;

    g.append('text')
        .attr('x', COL.head).attr('y', y + 4)
        .attr('font-size', '10px').attr('fill', '#aaa')
        .text(`I[${t},${s}] =`);

    const barX = 78, barW = 268, barH = 14;
    let shown = 0;
    for (let h = 0; h < HEADS; h++) shown += contrib[h];
    const rest = Math.max(0, total - shown);

    let x = barX;
    for (let h = 0; h < HEADS; h++) {
        const seg = total > 0 ? contrib[h] / total * barW : 0;
        if (seg <= 0) continue;
        g.append('rect')
            .attr('x', x).attr('y', y - barH / 2)
            .attr('width', Math.max(seg - 0.5, 0.5)).attr('height', barH)
            .attr('fill', CONTRIB).attr('fill-opacity', 0.55 + (h % 2) * 0.3);
        x += seg;
    }
    if (rest > 0) {
        g.append('rect')
            .attr('x', x).attr('y', y - barH / 2)
            .attr('width', Math.max(barX + barW - x, 0.5)).attr('height', barH)
            .attr('fill', GATED).attr('fill-opacity', 0.7);
    }
    g.append('rect')
        .attr('x', barX).attr('y', y - barH / 2)
        .attr('width', barW).attr('height', barH).attr('rx', 2)
        .attr('fill', 'none').attr('stroke', '#2a2d3a').attr('stroke-width', 0.75);

    g.append('text')
        .attr('x', barX + barW + 8).attr('y', y + 4)
        .attr('font-size', '11px').attr('fill', CONTRIB)
        .style('font-weight', '600')
        .text(total.toFixed(2));

    y += barH / 2 + 12;
    const legend = n_i > HEADS
        ? `${HEADS} heads shown · remaining ${n_i - HEADS} heads in grey`
        : `${HEADS} head${HEADS === 1 ? '' : 's'}`;
    g.append('text')
        .attr('x', barX).attr('y', y)
        .attr('font-size', '8px').attr('fill', '#666').text(legend);

    return y + 6;
}

// Greedy word wrap for footnote text; returns the extra height consumed
function wrapText(g, x, y, maxW, text, fill) {
    const perLine = Math.floor(maxW / 5.1);  // ~5.1px per char at 9px italic
    const lines = [];
    let line = '';
    for (const word of text.split(' ')) {
        if (line && (line.length + 1 + word.length) > perLine) {
            lines.push(line);
            line = word;
        } else {
            line = line ? `${line} ${word}` : word;
        }
    }
    if (line) lines.push(line);

    for (const [i, l] of lines.entries()) {
        g.append('text').attr('x', x).attr('y', y + i * 12)
            .attr('font-size', '9px').attr('fill', fill).attr('font-style', 'italic')
            .text(l);
    }
    return (lines.length - 1) * 12;
}
