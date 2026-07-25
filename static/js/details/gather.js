// gather.js — DSA sparse gather: each query pulls its k selected rows out of the
// KV cache. The point is non-contiguity — the rows a query needs are scattered
// across all S cached positions, so this is a gather, not a slice.
import { detailMetrics } from './shared.js';
import { fmtBytes, fmtNum } from '../costs.js';

const SEL = '#f1c40f', TILE = '#e67e22', IDX = '#f1c40f';

// Deterministic pseudo-random score — same hash as topk.js, so the rows shown
// here are the rows the top-k detail view would have picked
function pseudoScore(i, j) {
    let h = (i * 374761 + j * 668265) ^ 0x5bd1e995;
    h = Math.imul(h ^ (h >>> 13), 0x5bd1e995);
    h = h ^ (h >>> 15);
    return -0.8 + ((h >>> 0) / 0xffffffff) * 2.4;
}

// The `count` highest-scoring causal positions for query q, in ascending order
function selectedRows(q, eligible, count) {
    const ranked = [];
    for (let j = 0; j < eligible; j++) ranked.push([pseudoScore(q, j), j]);
    ranked.sort((a, b) => b[0] - a[0]);
    return ranked.slice(0, count).map(r => r[1]).sort((a, b) => a - b);
}

export function drawGatherDetail(svg, op, tensorMap, params) {
    const { w: svgW } = detailMetrics();
    const idx = tensorMap[op.inputs[0]];
    const src = tensorMap[op.inputs[1]];
    const out = tensorMap[op.output];
    if (!src || !out) return;

    const S = params.seqLens?.[0] ?? params.S ?? 1;
    const S_q = params.queryLens?.[0] ?? params.S_q ?? 1;
    const queryOffset = Math.max(0, S - S_q);
    const k = out.shape.length === 3 ? out.shape[1] : out.shape[0];
    const rowWidth = out.shape[out.shape.length - 1];
    const bytesPerEl = src.bytesPerEl || 2;
    const rowBytes = rowWidth * bytesPerEl;
    const dense = k >= S;

    // Example query: mid-sequence, so it has a decent causal window to choose from
    const q = queryOffset + Math.floor((S_q - 1) / 2);
    const eligible = Math.min(q + 1, S);
    const kEff = Math.min(k, eligible);
    const rows = selectedRows(q, eligible, kEff);
    // The strip shows true selection density; only a few rows get a drawn
    // connector, or the lines become an unreadable mat
    const SHOWN = Math.min(kEff, 14);
    const shownRows = SHOWN === kEff
        ? rows
        : Array.from({ length: SHOWN }, (_, i) => rows[Math.floor((i + 0.5) * kEff / SHOWN)]);

    const pad = 20, innerW = svgW - pad * 2;
    const g = svg.append('g').attr('transform', `translate(${pad}, 20)`);
    let y = 0;

    g.append('text').attr('class', 'tensor-label')
        .attr('x', innerW / 2).attr('y', y)
        .text(`Gather ${dense ? 'every causal row' : `k = ${k} rows`} per query from ${src.label}`);
    y += 15;
    g.append('text')
        .attr('x', innerW / 2).attr('y', y)
        .attr('text-anchor', 'middle').attr('font-size', '9px').attr('fill', '#888')
        .text(dense
            ? 'k ≥ S — the selection covers everything, so this degenerates to a full read'
            : `Selected positions are scattered across all ${S} cached rows — non-contiguous by construction`);
    y += 22;

    // --- Scattered source rows → contiguous SRAM tile ---
    const stripX = 46, stripW = 46, stripH = 224;
    const tileX = 300, tileW = 86;
    const tileRowH = Math.min(9, stripH / Math.max(SHOWN, 1));
    const tileH = SHOWN * tileRowH;
    const tileY = y + (stripH - tileH) / 2;

    g.append('text').attr('class', 'tensor-label')
        .attr('x', stripX + stripW / 2).attr('y', y - 8)
        .text(src.label);
    g.append('text').attr('class', 'tensor-label')
        .attr('x', tileX + tileW / 2).attr('y', y - 8)
        .text(out.label);

    // Cache strip
    g.append('rect')
        .attr('x', stripX).attr('y', y).attr('width', stripW).attr('height', stripH)
        .attr('fill', src.color).attr('fill-opacity', 0.16)
        .attr('stroke', src.color).attr('stroke-width', 1).attr('rx', 2);

    // Rows above the example query are not causally eligible
    if (eligible < S) {
        const cutY = y + (eligible / S) * stripH;
        g.append('rect')
            .attr('x', stripX).attr('y', cutY)
            .attr('width', stripW).attr('height', y + stripH - cutY)
            .attr('fill', '#0f1117').attr('fill-opacity', 0.72);
        g.append('line')
            .attr('x1', stripX).attr('y1', cutY).attr('x2', stripX + stripW).attr('y2', cutY)
            .attr('stroke', '#fff').attr('stroke-width', 1).attr('stroke-opacity', 0.35);
        g.append('text')
            .attr('x', stripX + stripW / 2).attr('y', cutY + 12)
            .attr('text-anchor', 'middle').attr('font-size', '8px').attr('fill', '#7d8290')
            .text('future');
    }

    // Selection density: one band per strip pixel, shaded by how many of that
    // band's cache rows the query picked. Truthful whether k/S is 1% or 50%.
    const bands = Math.max(1, Math.round(stripH));
    const rowsPerBand = S / bands;
    const cover = new Float64Array(bands);
    for (const pos of rows) cover[Math.min(bands - 1, Math.floor(pos / S * bands))]++;
    const bandH = stripH / bands;
    for (let bi = 0; bi < bands; bi++) {
        if (cover[bi] === 0) continue;
        const frac = Math.min(1, cover[bi] / Math.max(1, rowsPerBand));
        g.append('rect')
            .attr('x', stripX).attr('y', y + bi * bandH)
            .attr('width', stripW).attr('height', Math.max(bandH, 0.9))
            .attr('fill', SEL).attr('fill-opacity', 0.25 + 0.7 * frac);
    }

    const rowPx = Math.max(2, stripH / S);
    for (const [i, pos] of shownRows.entries()) {
        const ry = y + (pos / S) * stripH;
        g.append('rect')
            .attr('x', stripX - 3).attr('y', ry - rowPx / 2)
            .attr('width', stripW + 6).attr('height', rowPx)
            .attr('fill', '#fff').attr('fill-opacity', 0.9);

        const ty = tileY + i * tileRowH;
        g.append('line')
            .attr('x1', stripX + stripW + 3).attr('y1', ry)
            .attr('x2', tileX).attr('y2', ty + tileRowH / 2)
            .attr('stroke', SEL).attr('stroke-width', 0.5).attr('stroke-opacity', 0.45);
        g.append('rect')
            .attr('x', tileX).attr('y', ty)
            .attr('width', tileW).attr('height', Math.max(tileRowH - 1, 1))
            .attr('fill', TILE).attr('fill-opacity', 0.85);
    }

    // Tile outline
    g.append('rect')
        .attr('x', tileX).attr('y', tileY).attr('width', tileW).attr('height', tileH)
        .attr('fill', 'none').attr('stroke', TILE).attr('stroke-width', 1).attr('rx', 2);

    const label = (x, yy, text, anchor, fill) => g.append('text')
        .attr('x', x).attr('y', yy).attr('text-anchor', anchor || 'middle')
        .attr('font-size', '9px').attr('fill', fill || '#999').text(text);
    label(stripX + stripW / 2, y + stripH + 13, `S = ${S} rows`);
    label(stripX + stripW / 2, y + stripH + 25, `${rowWidth} wide · ${fmtBytes(rowBytes)}/row`, 'middle', '#777');
    label(tileX + tileW / 2, tileY + tileH + 13, `[${dense ? 'S' : 'k'}, ${rowWidth}] in SRAM`);
    label(innerW / 2, y + stripH + 42,
        SHOWN < kEff
            ? `shading = selection density · ${SHOWN} of query ${q}'s ${kEff} rows traced`
            : `query ${q} selected ${kEff} row${kEff === 1 ? '' : 's'}`,
        'middle', '#666');

    // Index arrow annotation
    if (idx) {
        label((stripX + stripW + tileX) / 2, y - 2, `via ${idx.label}`, 'middle', '#888');
    }
    y += stripH + 66;

    // --- Traffic bounds ---
    g.append('text').attr('class', 'tensor-label')
        .attr('x', innerW / 2).attr('y', y)
        .text('How much does this actually read?');
    y += 20;

    const totalQ = (params.queryLens || [S_q]).slice(0, params.B || 1).reduce((a, b) => a + b, 0);
    const totalS = (params.seqLens || [S]).slice(0, params.B || 1).reduce((a, b) => a + b, 0);
    const noReuse = totalQ * k * rowBytes;
    const unionBound = totalS * rowBytes;
    const ratio = unionBound > 0 ? noReuse / unionBound : 1;

    const bounds = [
        ['No reuse', `${fmtNum(totalQ * k)} row-reads = ${fmtBytes(noReuse)}`, '#e74c3c',
         'every query fetches its own k rows — what the cost panel charges'],
        ['Union bound', `≤ ${fmtNum(totalS)} distinct rows = ${fmtBytes(unionBound)}`, '#2ecc71',
         'no row exists more than once in the cache'],
    ];
    for (const [name, value, color, note] of bounds) {
        g.append('rect').attr('x', 0).attr('y', y - 8).attr('width', 3).attr('height', 26)
            .attr('fill', color).attr('rx', 1.5);
        g.append('text').attr('x', 12).attr('y', y)
            .attr('font-size', '10px').attr('fill', '#bbb').style('font-weight', '600').text(name);
        g.append('text').attr('x', 92).attr('y', y)
            .attr('font-size', '10px').attr('fill', color).text(value);
        g.append('text').attr('x', 12).attr('y', y + 12)
            .attr('font-size', '9px').attr('fill', '#777').attr('font-style', 'italic').text(note);
        y += 32;
    }

    y += 2;
    const gapNote = ratio > 1.05
        ? `Real kernels land between these: they process queries in tiles so a row fetched for one query serves its neighbours. The gap here is ${ratio < 10 ? ratio.toFixed(1) : fmtNum(Math.round(ratio))}×.`
        : `With these parameters the two bounds coincide — there is nothing to amortize.`;
    for (const [i, line] of wrap(gapNote, 84).entries()) {
        g.append('text').attr('x', 0).attr('y', y + i * 12)
            .attr('font-size', '9px').attr('fill', '#8a8f9e').attr('font-style', 'italic').text(line);
    }
    y += wrap(gapNote, 84).length * 12 + 4;

    if (out.sramOnly) {
        g.append('text').attr('x', 0).attr('y', y)
            .attr('font-size', '9px').attr('fill', '#6b7280').attr('font-style', 'italic')
            .text(`${out.label} is never written back to HBM — the tile is consumed in SRAM.`);
        y += 12;
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
