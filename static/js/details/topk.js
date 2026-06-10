// topk.js — Top-k selection detail visualization (DSA lightning indexer)
//
// Two modes:
//  - Exact: when every (query, key) pair fits as its own cell, render the true
//    selection tensor — keep-all rows where q+1 ≤ k, discrete top-k picks elsewhere.
//  - Schematic: a density map where brightness = kept fraction min(1, k/(q+1)).
//    The fraction depends only on the query row, so it renders as horizontal
//    strips clipped by a sharp causal boundary (no per-cell coverage artifacts),
//    plus a zoomed inset showing discrete token selection for one example query.

// Deterministic pseudo-random score for cell (i, j) — same hash as mask.js/softmax.js
function pseudoScore(i, j) {
    let h = (i * 374761 + j * 668265) ^ 0x5bd1e995;
    h = Math.imul(h ^ (h >>> 13), 0x5bd1e995);
    h = h ^ (h >>> 15);
    return -0.8 + ((h >>> 0) / 0xffffffff) * 2.4;  // range [-0.8, 1.6]
}

// Top keptCount of `width` positions for row q, ranked by pseudo-score
function pickKept(q, s0, width, keptCount) {
    const ranked = [];
    for (let j = 0; j < width; j++) ranked.push([pseudoScore(q, s0 + j), j]);
    ranked.sort((a, b) => b[0] - a[0]);
    return new Set(ranked.slice(0, keptCount).map(r => r[1]));
}

const KEPT = '#f1c40f', DROPPED = '#9b59b6', FUTURE = '#2c3e50';

export function drawTopkDetail(svg, op, tensorMap, params) {
    // Use first-request lengths
    const S = params.seqLens?.[0] ?? params.S;
    const S_q = params.queryLens?.[0] ?? params.S_q;
    const topk = params.topk || 2048;
    const k = Math.min(topk, S);
    const dense = topk >= S;
    const queryOffset = S - S_q;

    const pad = 20;
    const availW = 440;
    const g = svg.append('g').attr('transform', `translate(${pad}, 16)`);
    let y = 0;

    // Header
    g.append('text').attr('class', 'tensor-label')
        .attr('x', availW / 2).attr('y', y)
        .text(dense
            ? `k ≥ S: every causal position selected (dense fallback)`
            : `Each query keeps its top k = ${k} of S = ${S} keys (${(k / S * 100).toFixed(1)}%)`);
    y += 24;

    // Exact mode when each (query, key) pair gets its own legible cell
    const cw = Math.min(13, Math.floor(availW / S));
    const ch = Math.min(13, Math.floor(260 / S_q));
    const exact = cw >= 4 && ch >= 4;

    let legendKeptLabel;
    if (exact) {
        y = drawExactGrid(g, y, availW, { S, S_q, k, queryOffset, cellSize: Math.min(cw, ch) });
        legendKeptLabel = 'kept (top-k)';
    } else {
        y = drawDensitySchematic(g, y, availW, { S, S_q, k, dense, queryOffset });
        legendKeptLabel = 'kept (brightness = fraction)';
    }

    // Legend
    const legend = [
        { color: KEPT, opacity: 0.92, label: legendKeptLabel },
        { color: DROPPED, opacity: 0.25, label: 'causal, dropped' },
        { color: FUTURE, opacity: 0.35, label: 'future (masked)' },
    ];
    let lx = 16;
    for (const item of legend) {
        g.append('rect').attr('x', lx).attr('y', y - 8)
            .attr('width', 10).attr('height', 10).attr('rx', 1)
            .attr('fill', item.color).attr('fill-opacity', item.opacity);
        g.append('text').attr('x', lx + 14).attr('y', y + 1)
            .attr('font-size', '9px').attr('fill', '#aaa').text(item.label);
        lx += 14 + item.label.length * 5.2 + 16;
    }
    y += 24;

    // --- Output sketch: topk_idx [S_q, k] int32 ---
    const out = tensorMap[op.output];
    const blockW = 130, blockH = 44;
    const bx = (availW - blockW) / 2;
    g.append('rect').attr('x', bx).attr('y', y)
        .attr('width', blockW).attr('height', blockH).attr('rx', 3)
        .attr('fill', KEPT).attr('fill-opacity', 0.15)
        .attr('stroke', KEPT).attr('stroke-width', 1);
    g.append('text').attr('class', 'tensor-label')
        .attr('x', bx + blockW / 2).attr('y', y + blockH / 2 - 2)
        .text(out ? out.label : 'top-k idx');
    g.append('text').attr('class', 'dim-label')
        .attr('x', bx + blockW / 2).attr('y', y + blockH / 2 + 12)
        .attr('text-anchor', 'middle').attr('font-size', '9px')
        .text(out ? `[${out.shape.join(' × ')}] int32` : '');
    y += blockH + 18;

    g.append('text')
        .attr('x', availW / 2).attr('y', y)
        .attr('text-anchor', 'middle')
        .attr('fill', '#666').attr('font-size', '9px').attr('font-style', 'italic')
        .text('Only indices flow on — values are gathered from the cache by the sparse kernel.');

    svg.attr('height', y + 30);
}

// --- Exact mode: the true selection tensor, one cell per (query, key) pair ---

function drawExactGrid(g, y, availW, { S, S_q, k, queryOffset, cellSize }) {
    const gridW = S * cellSize;
    const gridX = Math.max((availW - gridW) / 2, 16);

    for (let i = 0; i < S_q; i++) {
        const q = queryOffset + i;
        const eligible = q + 1;
        // Real top-k: keep everything when a query has ≤ k causal keys
        const kept = eligible <= k ? null : pickKept(q, 0, eligible, k);
        for (let s = 0; s < S; s++) {
            const future = s > q;
            const sel = !future && (kept === null || kept.has(s));
            g.append('rect')
                .attr('x', gridX + s * cellSize).attr('y', y + i * cellSize)
                .attr('width', cellSize - 1).attr('height', cellSize - 1)
                .attr('rx', 1)
                .attr('fill', future ? FUTURE : sel ? KEPT : DROPPED)
                .attr('fill-opacity', future ? 0.35 : sel ? 0.92 : 0.25);
        }
    }

    drawAxes(g, gridX, y, gridW, S_q * cellSize, S, 'queries');
    return y + S_q * cellSize + 26;
}

// --- Schematic mode: causal-clipped density strips + discrete zoom inset ---

function drawDensitySchematic(g, y, availW, { S, S_q, k, dense, queryOffset }) {
    const gridW = 408;
    const gridH = 150;
    const gridX = Math.max((availW - gridW) / 2, 16);

    // Future background (sharp — drawn behind the causal polygon)
    g.append('rect')
        .attr('x', gridX).attr('y', y)
        .attr('width', gridW).attr('height', gridH)
        .attr('fill', FUTURE).attr('fill-opacity', 0.35).attr('rx', 2);

    // Causal region: trapezoid from row 0's cutoff to full width at the bottom
    const fTop = Math.min(1, (queryOffset + 1) / S);
    const polyPts = [
        [gridX, y], [gridX + fTop * gridW, y],
        [gridX + gridW, y + gridH], [gridX, y + gridH],
    ].map(p => p.join(',')).join(' ');

    // Purple base = causal-but-dropped; yellow strips brighten by kept fraction
    g.append('polygon').attr('points', polyPts)
        .attr('fill', DROPPED).attr('fill-opacity', 0.25);

    const clipId = 'topk-causal-clip';
    g.append('clipPath').attr('id', clipId)
        .append('polygon').attr('points', polyPts);

    // Kept fraction depends only on the query row → vertical gradient (smooth,
    // no strip banding), clipped to the causal region so the diagonal stays sharp
    const gradId = 'topk-density-grad';
    const grad = g.append('linearGradient')
        .attr('id', gradId)
        .attr('x1', 0).attr('y1', 0).attr('x2', 0).attr('y2', 1);
    const N_STOPS = 30;
    for (let m = 0; m <= N_STOPS; m++) {
        const q = queryOffset + (m / N_STOPS) * S_q;
        const frac = Math.min(1, k / (q + 1));
        grad.append('stop')
            .attr('offset', `${(m / N_STOPS * 100).toFixed(1)}%`)
            .attr('stop-color', KEPT)
            .attr('stop-opacity', 0.92 * frac);
    }
    g.append('rect')
        .attr('x', gridX).attr('y', y)
        .attr('width', gridW).attr('height', gridH)
        .attr('clip-path', `url(#${clipId})`)
        .attr('fill', `url(#${gradId})`);

    // Causal (diagonal) boundary line
    g.append('line')
        .attr('x1', gridX + fTop * gridW).attr('y1', y)
        .attr('x2', gridX + gridW).attr('y2', y + gridH)
        .attr('stroke', '#fff').attr('stroke-width', 1.5).attr('stroke-opacity', 0.4);

    // Boundary where queries stop keeping everything (q + 1 = k)
    if (!dense && k > queryOffset && k < S) {
        const by = y + ((k - queryOffset) / S_q) * gridH;
        if (by > y + 4 && by < y + gridH - 4) {
            g.append('line')
                .attr('x1', gridX).attr('y1', by)
                .attr('x2', gridX + gridW).attr('y2', by)
                .attr('stroke', '#fff').attr('stroke-width', 0.75)
                .attr('stroke-dasharray', '3,2').attr('stroke-opacity', 0.6);
            g.append('text')
                .attr('x', gridX + gridW - 4).attr('y', by - 4)
                .attr('text-anchor', 'end').attr('font-size', '8px')
                .attr('fill', '#fff').attr('fill-opacity', 0.7)
                .text('↑ ≤ k causal keys: all kept');
        }
    }

    drawAxes(g, gridX, y, gridW, gridH, S, 'queries');
    g.append('text').attr('class', 'dim-label')
        .attr('x', gridX + gridW / 2).attr('y', y + gridH + 24)
        .attr('text-anchor', 'middle').attr('fill', '#555').attr('font-size', '9px')
        .text('Schematic — reduce S and S_q to see every (query, key) cell');
    let yNext = y + gridH + 38;

    // --- Zoom inset: discrete selection for one example query ---
    // Pick the query so the kept fraction lands near 1/3 — sparsity is visible
    const qEx = Math.max(queryOffset, Math.min(S - 1, 3 * k - 1));
    const fracEx = k / (qEx + 1);
    if (!dense && fracEx <= 0.85) {
        const W_IN = 24;
        const cellIn = 14;
        const insetW = W_IN * cellIn;
        const insetX = (availW - insetW) / 2;
        const insetY = yNext + 14;
        // Window of consecutive keys, fully causal, away from the edges
        const s0 = Math.max(0, Math.min(Math.round(qEx * 0.6), qEx + 1 - W_IN));
        const keptCount = Math.max(1, Math.min(W_IN - 1, Math.round(W_IN * fracEx)));
        const kept = pickKept(qEx, s0, W_IN, keptCount);

        // Marker on the big map + dashed connectors to the inset
        const mx = gridX + ((s0 + W_IN / 2) / S) * gridW;
        const my = y + ((qEx - queryOffset + 0.5) / S_q) * gridH;
        g.append('rect')
            .attr('x', mx - 4).attr('y', my - 4)
            .attr('width', 8).attr('height', 8)
            .attr('fill', 'none').attr('stroke', '#fff')
            .attr('stroke-width', 1).attr('stroke-opacity', 0.85);
        for (const [tx, ty] of [[insetX, insetY], [insetX + insetW, insetY]]) {
            g.append('line')
                .attr('x1', tx < mx ? mx - 4 : mx + 4).attr('y1', my + 4)
                .attr('x2', tx).attr('y2', ty)
                .attr('stroke', '#fff').attr('stroke-width', 0.5)
                .attr('stroke-dasharray', '2,3').attr('stroke-opacity', 0.5);
        }

        for (let j = 0; j < W_IN; j++) {
            const sel = kept.has(j);
            g.append('rect')
                .attr('x', insetX + j * cellIn).attr('y', insetY)
                .attr('width', cellIn - 1).attr('height', cellIn - 1)
                .attr('rx', 2)
                .attr('fill', sel ? KEPT : DROPPED)
                .attr('fill-opacity', sel ? 0.92 : 0.25);
        }
        g.append('text').attr('class', 'dim-label')
            .attr('x', availW / 2).attr('y', insetY + cellIn + 13)
            .attr('text-anchor', 'middle').attr('font-size', '9px')
            .text(`zoom: keys ${s0} … ${s0 + W_IN - 1} for query ${qEx} — ` +
                  `${keptCount} of ${W_IN} kept (k/(q+1) ≈ ${(fracEx * 100).toFixed(0)}%)`);
        yNext = insetY + cellIn + 26;
    }
    return yNext;
}

function drawAxes(g, gridX, y, gridW, gridH, S, rowLabel) {
    const qlx = gridX - 8, qly = y + gridH / 2;
    g.append('text').attr('class', 'dim-label')
        .attr('x', qlx).attr('y', qly)
        .attr('transform', `rotate(-90, ${qlx}, ${qly})`)
        .attr('text-anchor', 'middle').attr('font-size', '9px')
        .text(rowLabel);
    g.append('text').attr('class', 'dim-label')
        .attr('x', gridX + gridW / 2).attr('y', y + gridH + 12)
        .attr('text-anchor', 'middle').attr('font-size', '9px')
        .text(`key positions 0 … ${S - 1}`);
}
