// cache.js — KV cache detail visualizations
//  - drawCacheOpDetail: the append itself (contiguous view), per request
//  - drawPagedCacheDetail: how new tokens get sliced into fixed-size blocks and
//    placed into a block table
//
// The cache op delegates to the paged view when PagedAttention is on, since the
// block layout is then the more truthful picture of what the append does.

import { fmtBytes } from '../costs.js';
import { RANK_COLORS } from '../render.js';

const SEQ_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];

// Split a cache tensor's shape into (sequence length, per-token dims).
// 3D [n_kv, S, d_h] → per-token [n_kv, d_h]; 2D [S, d_c] → per-token [d_c].
function perTokenLayout(tensor) {
    const is3D = tensor.shape.length === 3;
    const dimNames = tensor.dimNames || [];
    return {
        S: is3D ? tensor.shape[1] : tensor.shape[0],
        dims: is3D ? [dimNames[0], dimNames[2]] : dimNames.slice(1),
        shape: is3D ? [tensor.shape[0], tensor.shape[2]] : tensor.shape.slice(1),
    };
}

export function drawCacheOpDetail(svg, op, tensorMap, params) {
    const out = tensorMap[op.output];
    const src = tensorMap[op.inputs[0]];
    if (!out) return;

    // With PagedAttention on, the block layout is the real story — reuse that view
    if (params.pagedAttn && out.badge === 'PAGED') {
        drawPagedCacheDetail(svg, out, params);
        return;
    }

    const B = params.B || 1;
    const sLens = (params.seqLens || [params.S]).slice(0, B);
    const sqLens = (params.queryLens || [params.S_q]).slice(0, B);
    const { dims, shape: perTokenShape } = perTokenLayout(out);
    const bytesPerEl = out.bytesPerEl || 2;
    const elemsPerToken = perTokenShape.reduce((a, b) => a * b, 1);
    const bytesPerToken = elemsPerToken * bytesPerEl;

    const swaActive = !!params.slidingWindow;
    const swaW = params.window_size;
    const color = out.color || '#16a085';

    const pad = 20, innerW = 440;
    const g = svg.append('g').attr('transform', `translate(${pad}, 20)`);
    let y = 0;

    g.append('text').attr('class', 'tensor-label')
        .attr('x', innerW / 2).attr('y', y)
        .text(`Append new tokens to ${out.label}`);
    y += 15;
    g.append('text')
        .attr('x', innerW / 2).attr('y', y)
        .attr('text-anchor', 'middle').attr('font-size', '9px').attr('fill', '#888')
        .text(src ? `${src.label} [${src.shape.join(' × ')}] → rows ${'S − S_q'} … ${'S − 1'} of the cache` : '');
    y += 24;

    // --- Per-request growth bars, all scaled by the longest sequence ---
    const barX = 52, barW = 292, barH = 15, rowGap = 7;
    const maxS = Math.max(...sLens, 1);

    for (let i = 0; i < B; i++) {
        const S = sLens[i], S_q = Math.min(sqLens[i], S);
        const existing = S - S_q;
        const scale = (n) => (n / maxS) * barW;

        g.append('text')
            .attr('x', 0).attr('y', y + barH / 2 + 3)
            .attr('font-size', '9px').attr('fill', '#999')
            .text(B > 1 ? `Req ${i}` : 'Cache');

        // Track
        g.append('rect')
            .attr('x', barX).attr('y', y).attr('width', barW).attr('height', barH)
            .attr('fill', '#12141c').attr('rx', 2);

        if (existing > 0) {
            g.append('rect')
                .attr('x', barX).attr('y', y)
                .attr('width', scale(existing)).attr('height', barH)
                .attr('fill', color).attr('fill-opacity', 0.28);
        }
        // The append itself
        g.append('rect')
            .attr('x', barX + scale(existing)).attr('y', y)
            .attr('width', Math.max(scale(S_q), 1.5)).attr('height', barH)
            .attr('fill', color).attr('fill-opacity', 0.9)
            .attr('stroke', '#fff').attr('stroke-width', 0.75)
            .attr('stroke-dasharray', '2,1').attr('stroke-opacity', 0.55);

        // Rows before S-W fall outside the window — freeable regardless of whether
        // they were appended in this step, so this is an overlay, not a segment
        if (swaActive && swaW < S) {
            g.append('rect')
                .attr('x', barX).attr('y', y)
                .attr('width', scale(S - swaW)).attr('height', barH).attr('rx', 2)
                .attr('fill', '#0f1117').attr('fill-opacity', 0.72)
                .attr('stroke', '#5a6070').attr('stroke-width', 0.75)
                .attr('stroke-dasharray', '3,2');
        }

        const kind = S_q === 1 ? 'decode' : S_q >= S ? 'prefill' : S_q < 16 ? 'spec' : 'extend';
        g.append('text')
            .attr('x', barX + barW + 6).attr('y', y + barH / 2 + 3)
            .attr('font-size', '9px').attr('fill', '#888')
            .text(`S=${S} (+${S_q}, ${kind})`);
        y += barH + rowGap;
    }
    y += 6;

    const legend = [
        [color, 0.28, 'already cached', 'none'],
        [color, 0.9, 'appended now', '#fff'],
        ...(swaActive ? [['#0f1117', 0.72, 'outside window', '#5a6070']] : []),
    ];
    let lx = barX;
    for (const [c, opacity, label, stroke] of legend) {
        g.append('rect').attr('x', lx).attr('y', y - 8)
            .attr('width', 10).attr('height', 10).attr('rx', 1)
            .attr('fill', c).attr('fill-opacity', opacity)
            .attr('stroke', stroke).attr('stroke-width', stroke === 'none' ? 0 : 0.75)
            .attr('stroke-dasharray', stroke === '#fff' ? '2,1' : '3,2');
        g.append('text').attr('x', lx + 14).attr('y', y + 1)
            .attr('font-size', '9px').attr('fill', '#aaa').text(label);
        lx += 14 + label.length * 5.2 + 14;
    }
    y += 26;

    // --- What one token costs ---
    g.append('text').attr('class', 'tensor-label')
        .attr('x', innerW / 2).attr('y', y)
        .text('Cost per cached token');
    y += 18;

    const tokW = 132, tokH = 34, tokX = (innerW - tokW) / 2;
    g.append('rect')
        .attr('x', tokX).attr('y', y).attr('width', tokW).attr('height', tokH).attr('rx', 3)
        .attr('fill', color).attr('fill-opacity', 0.5)
        .attr('stroke', color).attr('stroke-width', 1);
    g.append('text').attr('class', 'tensor-label')
        .attr('x', innerW / 2).attr('y', y + tokH / 2 + 4)
        .text('1 token');
    g.append('text')
        .attr('x', tokX - 8).attr('y', y + tokH / 2 + 3)
        .attr('text-anchor', 'end').attr('font-size', '9px').attr('fill', '#999')
        .text(dims.filter(Boolean).join(' × ') || 'd');
    g.append('text')
        .attr('x', tokX + tokW + 8).attr('y', y + tokH / 2 + 3)
        .attr('font-size', '9px').attr('fill', '#999')
        .text(`${fmtBytes(bytesPerToken)}`);
    y += tokH + 20;

    const dtype = { 1: 'fp8', 2: 'bf16', 4: 'int32' }[bytesPerEl] || `${bytesPerEl}B/el`;
    const totalNew = sqLens.reduce((a, b, i) => a + Math.min(b, sLens[i]), 0);
    const totalS = sLens.reduce((a, b) => a + b, 0);
    const rows = [
        [`Per token`, `${perTokenShape.join(' × ')} = ${elemsPerToken.toLocaleString()} el × ${bytesPerEl}B (${dtype}) = ${fmtBytes(bytesPerToken)}`],
        [`Appended`, `${totalNew.toLocaleString()} token${totalNew === 1 ? '' : 's'} = ${fmtBytes(totalNew * bytesPerToken)}`],
        [`Cache after`, `${totalS.toLocaleString()} tokens = ${fmtBytes(totalS * bytesPerToken)}`],
    ];
    const growth = [1024, 8192, 131072]
        .map(n => `${n >= 1024 ? `${n / 1024}K` : n} → ${fmtBytes(n * bytesPerToken)}`)
        .join('   ·   ');
    rows.push(['Grows as', `${growth}   (one request)`]);

    for (const [label, value] of rows) {
        g.append('text').attr('x', 0).attr('y', y)
            .attr('font-size', '10px').attr('fill', '#888').text(label + ':');
        g.append('text').attr('x', 76).attr('y', y)
            .attr('font-size', '10px').attr('fill', '#bbb').text(value);
        y += 15;
    }

    y += 8;
    g.append('text').attr('x', 0).attr('y', y)
        .attr('font-size', '9px').attr('fill', '#8a8f9e').attr('font-style', 'italic')
        .text('No math — a scatter-write kernel (vLLM: reshape_and_cache) reads the new');
    y += 12;
    g.append('text').attr('x', 0).attr('y', y)
        .attr('font-size', '9px').attr('fill', '#8a8f9e').attr('font-style', 'italic')
        .text('rows and writes them into cache memory. Pure bandwidth.');

    svg.attr('height', y + 30);
}

export function drawPagedCacheDetail(svg, tensor, params) {
    const bs = params.block_size;
    const B = params.B || 1;
    const sLens = params.seqLens.slice(0, B);
    const sqLens = params.queryLens.slice(0, B);
    const blocksPerSeq = sLens.map(s => Math.ceil(s / bs));
    const totalBlocks = blocksPerSeq.reduce((a, b) => a + b, 0);
    const pad = 20;

    // Sliding-window eviction: blocks lying entirely before the active window
    // [S-W, S-1] can be freed — only the window's worth of blocks stays resident.
    const swaActive = !!params.slidingWindow;
    const swaW = params.window_size;
    const evictedFor = s => (swaActive && swaW < s) ? Math.max(0, Math.floor((s - swaW) / bs)) : 0;
    const evictedPerSeq = sLens.map(evictedFor);
    const liveBlocksPerSeq = blocksPerSeq.map((nb, i) => nb - evictedPerSeq[i]);
    const totalLiveBlocks = liveBlocksPerSeq.reduce((a, b) => a + b, 0);

    // Per-token dims from tensor annotation
    const perTokenDims = tensor.pagedBlockDims || ['n_h', 'd_h'];
    const perTokenShape = tensor.pagedBlockShape || [];
    const perTokenSize = perTokenShape.reduce((a, b) => a * b, 1) * (tensor.bytesPerEl || 2);
    const blockBytes = bs * perTokenSize;

    // Source tensor info (attached during annotation)
    const src = tensor.cacheSource;

    let selectedReq = 0;

    function redraw() {
        svg.selectAll('*').remove();
        const g = svg.append('g').attr('transform', `translate(${pad}, 10)`);
        const availW = 440;
        let y = 0;

        // Title
        g.append('text').attr('class', 'tensor-label')
            .attr('x', availW / 2).attr('y', y)
            .text(`Paged Cache: ${tensor.label}`);
        y += 22;

        // --- Request selector (top, prominent, wrapping) ---
        if (B > 1) {
            g.append('text')
                .attr('x', availW / 2).attr('y', y)
                .attr('text-anchor', 'middle')
                .attr('fill', '#666').attr('font-size', '10px').attr('font-style', 'italic')
                .text('Select a request to view its cache layout');
            y += 14;
            const btnW = 56;
            const btnH = 22;
            const btnGap = 6;
            const btnsPerRow = Math.max(1, Math.floor(availW / (btnW + btnGap)));
            const numRows = Math.ceil(B / btnsPerRow);

            for (let ri = 0; ri < B; ri++) {
                const bcolor = SEQ_COLORS[ri % SEQ_COLORS.length];
                const isSelected = ri === selectedReq;
                const col = ri % btnsPerRow;
                const row = Math.floor(ri / btnsPerRow);
                const rowCount = Math.min(B - row * btnsPerRow, btnsPerRow);
                const rowW = rowCount * (btnW + btnGap) - btnGap;
                const rowStartX = (availW - rowW) / 2;
                const bx = rowStartX + col * (btnW + btnGap);
                const by = y + row * (btnH + btnGap);
                const rSq = sqLens[ri], rS = sLens[ri];
                const rType = rSq === 1 ? 'decode' : rSq >= rS ? 'prefill' : rSq < 16 ? 'spec' : 'extend';

                const btn = g.append('g').style('cursor', 'pointer')
                    .on('click', () => { selectedReq = ri; redraw(); });

                btn.append('rect')
                    .attr('x', bx).attr('y', by - 2)
                    .attr('width', btnW).attr('height', btnH)
                    .attr('fill', isSelected ? bcolor : '#1e2030')
                    .attr('fill-opacity', isSelected ? 0.6 : 1)
                    .attr('stroke', bcolor).attr('stroke-width', isSelected ? 2.5 : 1)
                    .attr('rx', 5);

                btn.append('text')
                    .attr('x', bx + btnW / 2).attr('y', by + 7)
                    .attr('text-anchor', 'middle')
                    .attr('fill', isSelected ? '#fff' : '#aaa').attr('font-size', '10px')
                    .attr('font-weight', isSelected ? '700' : '400')
                    .text(`Req ${ri}`);

                btn.append('text')
                    .attr('x', bx + btnW / 2).attr('y', by + 17)
                    .attr('text-anchor', 'middle')
                    .attr('fill', isSelected ? '#ddd' : '#666').attr('font-size', '9px')
                    .text(rType);
            }
            y += numRows * (btnH + btnGap) + 4;
        }

        // --- Section 1: Source tensor → slicing → blocks ---
        const srcLabel = src ? src.label : tensor.label;
        const srcColor = src ? src.color : tensor.color || '#4a90d9';
        const srcShapeStr = src ? `[${src.shape.join(', ')}]` : '';
        const srcDimStr = src ? src.dimNames.join(', ') : '';

        g.append('text')
            .attr('x', 0).attr('y', y)
            .attr('fill', '#bbb').attr('font-size', '11px').attr('font-weight', '600')
            .text(`Source: ${srcLabel} (new tokens)`);
        y += 4;
        if (srcShapeStr) {
            g.append('text')
                .attr('x', 0).attr('y', y + 10)
                .attr('fill', '#666').attr('font-size', '9px')
                .text(`Shape: ${srcShapeStr} (${srcDimStr})`);
            y += 14;
        }
        y += 6;

        // Draw the source tensor as a tall bar, then show it being sliced into blocks
        // Source bar height proportional to S_q for selected request
        const si = selectedReq;
        const sq = sqLens[si];
        const s = sLens[si];
        const color = SEQ_COLORS[si % SEQ_COLORS.length];
        const typeStr = sq === 1 ? 'decode' : sq >= s ? 'prefill' : sq < 16 ? 'spec decode' : 'extend';
        const cachedTokens = s - sq;

        // How many free slots in the last cached block?
        const freeInLastBlock = cachedTokens > 0 ? (bs - cachedTokens % bs) % bs : 0;
        const tokensFillingExisting = Math.min(sq, freeInLastBlock);
        const tokensNeedingNewBlocks = sq - tokensFillingExisting;
        const nNewBlocks = Math.ceil(tokensNeedingNewBlocks / bs);

        // Layout: source tensor on left, arrow in middle, blocks on right
        // Blocks use a grid when there are too many to stack vertically
        const blockDrawW = 44;
        const blockGap = 4;
        const srcW = 50;
        const arrowW = 78;

        // Determine block grid dimensions — fit within available space
        const labelMargin = 55;
        const maxBlocksRightW = availW - labelMargin - srcW - arrowW - 10;
        const maxBlockCols = Math.max(1, Math.floor(maxBlocksRightW / (blockDrawW + blockGap)));
        const displayBlocks = Math.max(1, nNewBlocks); // at least 1 for layout sizing
        const blockCols = Math.min(maxBlockCols, Math.ceil(displayBlocks / 4));
        const blockRowCount = Math.ceil(displayBlocks / Math.max(blockCols, 1));
        // Scale block height to fit, with bounds
        const maxSectionH = 200;
        const blockDrawH = Math.max(14, Math.min(36, (maxSectionH - 20) / blockRowCount - blockGap));
        const blocksAreaW = blockCols * (blockDrawW + blockGap) - blockGap;
        const blocksAreaH = blockRowCount * (blockDrawH + blockGap) - blockGap;

        // Source tensor height matches block area
        const showExistingBlock = cachedTokens > 0;
        const existingBlockExtra = showExistingBlock ? blockDrawH + blockGap + 14 : 0;
        const srcH = Math.max(40, blocksAreaH + existingBlockExtra);
        const sectionH = Math.max(srcH, blocksAreaH + existingBlockExtra);

        // Position: source is pinned left, blocks right — leave room for S_q label on left
        const totalSectionW = srcW + arrowW + blocksAreaW;
        const srcX = Math.max(labelMargin, (availW - totalSectionW) / 2);

        // Draw source tensor
        const srcDrawY = y + (sectionH - srcH) / 2;
        g.append('rect')
            .attr('x', srcX).attr('y', srcDrawY)
            .attr('width', srcW).attr('height', srcH)
            .attr('fill', srcColor).attr('fill-opacity', 0.5)
            .attr('stroke', srcColor).attr('stroke-width', 1.5)
            .attr('rx', 3);

        // Slice lines on source tensor
        // First line separates tokens filling existing partial block from new-block tokens
        if (tokensFillingExisting > 0 && tokensNeedingNewBlocks > 0) {
            const sliceY = srcDrawY + (tokensFillingExisting / sq) * srcH;
            g.append('line')
                .attr('x1', srcX).attr('y1', sliceY)
                .attr('x2', srcX + srcW).attr('y2', sliceY)
                .attr('stroke', '#f8e45c').attr('stroke-width', 1).attr('stroke-opacity', 0.7)
                .attr('stroke-dasharray', '3,2');
        }
        // Remaining lines separate new blocks
        for (let bi = 1; bi < nNewBlocks; bi++) {
            const sliceY = srcDrawY + ((tokensFillingExisting + bi * bs) / sq) * srcH;
            if (sliceY < srcDrawY + srcH - 1) {
                g.append('line')
                    .attr('x1', srcX).attr('y1', sliceY)
                    .attr('x2', srcX + srcW).attr('y2', sliceY)
                    .attr('stroke', '#fff').attr('stroke-width', 1).attr('stroke-opacity', 0.5);
            }
        }

        // Label inside source
        g.append('text')
            .attr('x', srcX + srcW / 2).attr('y', srcDrawY + srcH / 2 + 3)
            .attr('text-anchor', 'middle')
            .attr('fill', '#fff').attr('font-size', '9px').attr('font-weight', '600')
            .text(`${srcLabel}`);
        // S_q label
        g.append('text')
            .attr('x', srcX - 4).attr('y', srcDrawY + srcH / 2 + 3)
            .attr('text-anchor', 'end')
            .attr('fill', '#aaa').attr('font-size', '10px')
            .text(`S_q=${sq}`);

        // Arrow from source to blocks
        const arrowX = srcX + srcW + 15;
        const arrowMidY = y + sectionH / 2;
        g.append('text')
            .attr('x', arrowX).attr('y', arrowMidY - 4)
            .attr('fill', '#888').attr('font-size', '10px')
            .text('slice by');
        g.append('text')
            .attr('x', arrowX).attr('y', arrowMidY + 6)
            .attr('fill', '#7c8cf8').attr('font-size', '9px').attr('font-weight', '600')
            .text(`bs=${bs}`);
        g.append('text')
            .attr('x', arrowX + 28).attr('y', arrowMidY + 1)
            .attr('fill', '#888').attr('font-size', '14px')
            .text('\u2192');

        // Draw resulting blocks — laid out top-to-bottom with existing block first
        const blocksX = arrowX + 50;
        const totalBlocksH = existingBlockExtra + blocksAreaH;
        const blocksTopY = y + (sectionH - totalBlocksH) / 2;
        const blocksStartY = blocksTopY + existingBlockExtra;

        // Show last cached block (partial or full) when there are cached tokens
        if (showExistingBlock) {
            const existBlockX = blocksX;
            const existBlockY = blocksTopY + 10;

            // Connection line from top of source to existing block (only when tokens fill it)
            if (blockCols === 1 && tokensFillingExisting > 0) {
                const srcSliceY = srcDrawY + (tokensFillingExisting / 2 / sq) * srcH;
                g.append('line')
                    .attr('x1', srcX + srcW).attr('y1', srcSliceY)
                    .attr('x2', existBlockX).attr('y2', existBlockY + blockDrawH / 2)
                    .attr('stroke', '#444').attr('stroke-width', 0.5)
                    .attr('stroke-dasharray', '2,2');
            }

            // Last cached block — show cached portion + new fill (if any)
            const cachedInExisting = freeInLastBlock > 0 ? bs - freeInLastBlock : bs;
            const cachedFrac = cachedInExisting / bs;
            const newFrac = tokensFillingExisting / bs;

            g.append('rect')
                .attr('x', existBlockX).attr('y', existBlockY)
                .attr('width', blockDrawW).attr('height', blockDrawH)
                .attr('fill', '#1e2030')
                .attr('stroke', srcColor).attr('stroke-width', 1.5)
                .attr('stroke-dasharray', '4,2')
                .attr('rx', 3);
            // Cached portion
            g.append('rect')
                .attr('x', existBlockX + 1).attr('y', existBlockY + 1)
                .attr('width', (blockDrawW - 2) * cachedFrac).attr('height', blockDrawH - 2)
                .attr('fill', srcColor).attr('fill-opacity', 0.25)
                .attr('rx', 2);
            // New fill portion
            if (tokensFillingExisting > 0) {
                g.append('rect')
                    .attr('x', existBlockX + 1 + (blockDrawW - 2) * cachedFrac).attr('y', existBlockY + 1)
                    .attr('width', (blockDrawW - 2) * newFrac).attr('height', blockDrawH - 2)
                    .attr('fill', srcColor).attr('fill-opacity', 0.55)
                    .attr('stroke', '#fff').attr('stroke-width', 0.5)
                    .attr('stroke-dasharray', '2,1').attr('stroke-opacity', 0.4)
                    .attr('rx', 2);
            }
            if (blockDrawH >= 14) {
                g.append('text')
                    .attr('x', existBlockX + blockDrawW / 2).attr('y', existBlockY + blockDrawH / 2 + 3)
                    .attr('text-anchor', 'middle')
                    .attr('fill', '#ddd').attr('font-size', '10px')
                    .text(tokensFillingExisting > 0 ? `+${tokensFillingExisting}` : 'full');
            }
            g.append('text')
                .attr('x', existBlockX + blockDrawW / 2).attr('y', existBlockY - 4)
                .attr('text-anchor', 'middle')
                .attr('fill', '#f8e45c').attr('font-size', '10px')
                .text('existing block');
        }

        for (let bi = 0; bi < nNewBlocks; bi++) {
            const col = bi % blockCols;
            const row = Math.floor(bi / blockCols);
            const bx = blocksX + col * (blockDrawW + blockGap);
            const by = blocksStartY + row * (blockDrawH + blockGap);
            const tokensInBlock = Math.min(bs, tokensNeedingNewBlocks - bi * bs);
            const fillRatio = tokensInBlock / bs;

            // Connection line from source slice to block (only for single column)
            if (blockCols === 1) {
                const srcSliceY = srcDrawY + ((tokensFillingExisting + (bi + 0.5) * bs) / sq) * srcH;
                g.append('line')
                    .attr('x1', srcX + srcW).attr('y1', Math.min(srcSliceY, srcDrawY + srcH))
                    .attr('x2', bx).attr('y2', by + blockDrawH / 2)
                    .attr('stroke', '#444').attr('stroke-width', 0.5)
                    .attr('stroke-dasharray', '2,2');
            }

            // Block
            g.append('rect')
                .attr('x', bx).attr('y', by)
                .attr('width', blockDrawW).attr('height', blockDrawH)
                .attr('fill', '#1e2030')
                .attr('stroke', srcColor).attr('stroke-width', 1.5)
                .attr('rx', 3);

            // Fill portion
            g.append('rect')
                .attr('x', bx + 1).attr('y', by + 1)
                .attr('width', (blockDrawW - 2) * fillRatio).attr('height', blockDrawH - 2)
                .attr('fill', srcColor).attr('fill-opacity', 0.55)
                .attr('rx', 2);

            if (blockDrawH >= 14) {
                g.append('text')
                    .attr('x', bx + blockDrawW / 2).attr('y', by + blockDrawH / 2 + 3)
                    .attr('text-anchor', 'middle')
                    .attr('fill', '#ddd').attr('font-size', '10px')
                    .text(`${tokensInBlock}/${bs}`);
            }
        }

        // Label: "new blocks"
        if (nNewBlocks > 0) {
            g.append('text')
                .attr('x', blocksX + blocksAreaW / 2).attr('y', blocksStartY - 6)
                .attr('text-anchor', 'middle')
                .attr('fill', '#888').attr('font-size', '10px')
                .text(`${nNewBlocks} new block${nNewBlocks !== 1 ? 's' : ''}`);
        } else if (tokensFillingExisting > 0) {
            g.append('text')
                .attr('x', blocksX + blocksAreaW / 2).attr('y', blocksStartY - 6)
                .attr('text-anchor', 'middle')
                .attr('fill', '#888').attr('font-size', '10px')
                .text('no new blocks needed');
        }

        y += sectionH + 16;

        // Request info line
        g.append('text')
            .attr('x', 0).attr('y', y)
            .attr('fill', '#888').attr('font-size', '9px')
            .text(`S=${s}, S_q=${sq} (${typeStr}), cached=${cachedTokens}, ${nNewBlocks} new block${nNewBlocks !== 1 ? 's' : ''}${tokensFillingExisting > 0 ? ` (+${tokensFillingExisting} into existing)` : ''}`);
        y += 16;

        // --- Section 2: Full block table for selected request ---
        const nBlocks = blocksPerSeq[si];
        const nEvicted = evictedPerSeq[si];

        g.append('text')
            .attr('x', 0).attr('y', y)
            .attr('fill', '#bbb').attr('font-size', '11px').attr('font-weight', '600')
            .text(nEvicted > 0
                ? `Block table — Req ${si} (${liveBlocksPerSeq[si]} live / ${nBlocks} blocks)`
                : `Block table — Req ${si} (${nBlocks} blocks)`);
        y += 16;
        if (swaActive && swaW < s) {
            g.append('text')
                .attr('x', 0).attr('y', y - 3)
                .attr('fill', '#e67e22').attr('font-size', '9px').attr('font-style', 'italic')
                .text(`Sliding window W=${swaW}: only blocks covering the last ${swaW} keys stay resident; older blocks are evicted.`);
            y += 12;
        }

        const blockW = Math.min(54, Math.max(30, (availW - 20) / Math.max(nBlocks, 1)));
        const blockH = 40;
        const blocksPerRow = Math.floor(availW / (blockW + 3)) || 1;
        const blockTableW = Math.min(nBlocks, blocksPerRow) * (blockW + 3) - 3;
        const blockTableOffX = (availW - blockTableW) / 2;

        for (let bi = 0; bi < nBlocks; bi++) {
            const col = bi % blocksPerRow;
            const row = Math.floor(bi / blocksPerRow);
            const bx = blockTableOffX + col * (blockW + 3);
            const by = y + row * (blockH + 4);
            const blockStart = bi * bs;
            const blockEnd = Math.min((bi + 1) * bs, s);
            const tokensInBlock = blockEnd - blockStart;

            // How many tokens in this block are cached vs new
            const cachedInBlock = Math.max(0, Math.min(tokensInBlock, cachedTokens - blockStart));
            const newInBlock = tokensInBlock - cachedInBlock;
            const cachedFrac = cachedInBlock / bs;
            const newFrac = newInBlock / bs;
            const isEvicted = bi < nEvicted;

            // Evicted block — freed slot, drawn dim/dashed with no contents
            if (isEvicted) {
                g.append('rect')
                    .attr('x', bx).attr('y', by)
                    .attr('width', blockW).attr('height', blockH)
                    .attr('fill', '#15171f')
                    .attr('stroke', '#444').attr('stroke-width', 1)
                    .attr('stroke-dasharray', '3,2')
                    .attr('rx', 3);
                if (blockW >= 22) {
                    g.append('text')
                        .attr('x', bx + blockW / 2).attr('y', by + blockH / 2 - 1)
                        .attr('text-anchor', 'middle')
                        .attr('fill', '#666').attr('font-size', '9px')
                        .text('evicted');
                    g.append('text')
                        .attr('x', bx + blockW / 2).attr('y', by + blockH / 2 + 10)
                        .attr('text-anchor', 'middle')
                        .attr('fill', '#555').attr('font-size', '9px')
                        .text(`blk ${bi}`);
                }
                continue;
            }

            // Block background
            g.append('rect')
                .attr('x', bx).attr('y', by)
                .attr('width', blockW).attr('height', blockH)
                .attr('fill', '#1e2030')
                .attr('stroke', color).attr('stroke-width', 1.5)
                .attr('rx', 3);

            // Cached portion
            if (cachedInBlock > 0) {
                g.append('rect')
                    .attr('x', bx + 1).attr('y', by + 1)
                    .attr('width', (blockW - 2) * cachedFrac).attr('height', blockH - 2)
                    .attr('fill', color).attr('fill-opacity', 0.25)
                    .attr('rx', 2);
            }

            // New portion (brighter)
            if (newInBlock > 0) {
                const newX = bx + 1 + (blockW - 2) * cachedFrac;
                g.append('rect')
                    .attr('x', newX).attr('y', by + 1)
                    .attr('width', (blockW - 2) * newFrac).attr('height', blockH - 2)
                    .attr('fill', color).attr('fill-opacity', 0.55)
                    .attr('stroke', '#fff').attr('stroke-width', 0.5)
                    .attr('stroke-dasharray', '2,1').attr('stroke-opacity', 0.4)
                    .attr('rx', 2);
            }

            // Block labels
            if (blockW >= 22) {
                g.append('text')
                    .attr('x', bx + blockW / 2).attr('y', by + blockH / 2)
                    .attr('text-anchor', 'middle')
                    .attr('fill', '#ddd').attr('font-size', '10px')
                    .text(`${tokensInBlock}/${bs}`);
                g.append('text')
                    .attr('x', bx + blockW / 2).attr('y', by + blockH / 2 + 10)
                    .attr('text-anchor', 'middle')
                    .attr('fill', '#888').attr('font-size', '9px')
                    .text(`blk ${bi}`);
            }
        }
        const tableRows = Math.ceil(nBlocks / blocksPerRow);
        y += tableRows * (blockH + 4) + 2;

        // Legend
        const legY = y;
        g.append('rect').attr('x', 0).attr('y', legY).attr('width', 10).attr('height', 10)
            .attr('fill', color).attr('fill-opacity', 0.25).attr('rx', 1)
            .attr('stroke', color).attr('stroke-width', 1);
        g.append('text').attr('x', 14).attr('y', legY + 8)
            .attr('fill', '#aaa').attr('font-size', '9px').text(`Cached (${cachedTokens} tokens)`);
        g.append('rect').attr('x', 140).attr('y', legY).attr('width', 10).attr('height', 10)
            .attr('fill', color).attr('fill-opacity', 0.55).attr('rx', 1)
            .attr('stroke', '#fff').attr('stroke-width', 0.5).attr('stroke-dasharray', '2,1');
        g.append('text').attr('x', 154).attr('y', legY + 8)
            .attr('fill', '#aaa').attr('font-size', '9px').text(`New S_q=${sq} tokens`);
        if (nEvicted > 0) {
            g.append('rect').attr('x', 280).attr('y', legY).attr('width', 10).attr('height', 10)
                .attr('fill', '#15171f').attr('rx', 1)
                .attr('stroke', '#444').attr('stroke-width', 1).attr('stroke-dasharray', '3,2');
            g.append('text').attr('x', 294).attr('y', legY + 8)
                .attr('fill', '#aaa').attr('font-size', '9px').text(`Evicted (${nEvicted} block${nEvicted !== 1 ? 's' : ''})`);
        }
        y += 18;

        // --- Section 3: Logical blocks per DP rank ---
        y += 8;
        const dp = params.dp_size || 1;

        if (dp > 1) {
            g.append('text').attr('x', 0).attr('y', y)
                .attr('fill', '#bbb').attr('font-size', '11px').attr('font-weight', '600')
                .text('Logical blocks per DP rank (non-contiguous in memory)');
            y += 16;
        }

        for (let rank = 0; rank < dp; rank++) {
            // Determine which requests belong to this rank
            const rankReqs = [];
            for (let r = 0; r < B; r++) {
                if (dp === 1 || Math.floor(r * dp / B) === rank) rankReqs.push(r);
            }
            if (rankReqs.length === 0) continue;

            if (dp > 1) {
                const rankColor = RANK_COLORS[rank % RANK_COLORS.length];
                g.append('text').attr('x', 0).attr('y', y)
                    .attr('fill', rankColor).attr('font-size', '10px').attr('font-weight', '600')
                    .text(`DP Rank ${rank} (Req ${rankReqs.join(', ')})`);
                y += 14;
            } else {
                g.append('text').attr('x', 0).attr('y', y)
                    .attr('fill', '#bbb').attr('font-size', '11px').attr('font-weight', '600')
                    .text('Logical blocks (non-contiguous in memory)');
                y += 16;
            }

            // Collect blocks for this rank's requests — evicted blocks have freed
            // their physical slots, so only live blocks remain resident.
            const rankBlocks = [];
            for (const ri of rankReqs) {
                for (let bi = evictedPerSeq[ri]; bi < blocksPerSeq[ri]; bi++) {
                    rankBlocks.push({
                        seq: ri, block: bi,
                        tokens: Math.min(bs, sLens[ri] - bi * bs),
                    });
                }
            }

            // Deterministic shuffle (vary seed per rank so layouts differ)
            const shuffled = [...rankBlocks];
            for (let i = shuffled.length - 1; i > 0; i--) {
                const j = ((i * 2654435761 + rank * 1234567) >>> 0) % (i + 1);
                [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
            }

            const physBlockW = Math.max(18, Math.min(36, (availW - 20) / Math.max(shuffled.length, 1)));
            const physBlockH = 24;
            const physPerRow = Math.floor(availW / (physBlockW + 3)) || 1;
            const physRowW = Math.min(shuffled.length, physPerRow) * (physBlockW + 3) - 3;
            const physOffX = (availW - physRowW) / 2;

            for (let i = 0; i < shuffled.length; i++) {
                const pb = shuffled[i];
                const col = i % physPerRow;
                const row = Math.floor(i / physPerRow);
                const pcolor = SEQ_COLORS[pb.seq % SEQ_COLORS.length];
                const bx = physOffX + col * (physBlockW + 3);
                const by = y + row * (physBlockH + 3);
                const isSelReq = pb.seq === selectedReq;

                g.append('rect')
                    .attr('x', bx).attr('y', by)
                    .attr('width', physBlockW).attr('height', physBlockH)
                    .attr('fill', pcolor).attr('fill-opacity', isSelReq ? 0.45 : 0.2)
                    .attr('stroke', pcolor).attr('stroke-width', isSelReq ? 1.5 : 0.5)
                    .attr('rx', 3);

                if (physBlockW >= 18) {
                    g.append('text')
                        .attr('x', bx + physBlockW / 2).attr('y', by + physBlockH / 2 + 3)
                        .attr('text-anchor', 'middle')
                        .attr('fill', '#ddd').attr('font-size', '9px')
                        .text(`R${pb.seq}`);
                }
            }
            const physRows = Math.ceil(shuffled.length / physPerRow);
            y += physRows * (physBlockH + 3) + (dp > 1 ? 12 : 8);
        }

        // --- Section 4: Block shape info ---
        y += 4;
        g.append('text').attr('x', 0).attr('y', y).attr('fill', '#777').attr('font-size', '10px')
            .text(`Block shape: [${bs}, ${perTokenDims.join(', ')}]`);
        y += 14;
        g.append('text').attr('x', 0).attr('y', y).attr('fill', '#777').attr('font-size', '10px')
            .text(`Block size: ${fmtBytes(blockBytes)} (${bs} tokens × ${fmtBytes(perTokenSize)}/token)`);
        y += 14;
        if (dp > 1) {
            for (let rank = 0; rank < dp; rank++) {
                const rankReqs = [];
                for (let r = 0; r < B; r++) {
                    if (Math.floor(r * dp / B) === rank) rankReqs.push(r);
                }
                const rankBlocks = rankReqs.reduce((sum, ri) => sum + liveBlocksPerSeq[ri], 0);
                const rankColor = RANK_COLORS[rank % RANK_COLORS.length];
                g.append('text').attr('x', 0).attr('y', y).attr('fill', rankColor).attr('font-size', '10px')
                    .text(`DP Rank ${rank}: ${rankBlocks} resident blocks, ${fmtBytes(rankBlocks * blockBytes)}`);
                y += 14;
            }
        }
        if (totalLiveBlocks < totalBlocks) {
            const liveBreakdown = B > 1 ? ` (${liveBlocksPerSeq.join(' + ')})` : '';
            g.append('text').attr('x', 0).attr('y', y).attr('fill', '#777').attr('font-size', '10px')
                .text(`Resident: ${totalLiveBlocks} live blocks${liveBreakdown}, ${fmtBytes(totalLiveBlocks * blockBytes)}`);
            y += 14;
            const saved = totalBlocks - totalLiveBlocks;
            g.append('text').attr('x', 0).attr('y', y).attr('fill', '#2ecc71').attr('font-size', '10px')
                .text(`Evicted: ${saved} block${saved !== 1 ? 's' : ''} freed (${fmtBytes(saved * blockBytes)} reclaimed vs full cache of ${fmtBytes(totalBlocks * blockBytes)})`);
        } else {
            g.append('text').attr('x', 0).attr('y', y).attr('fill', '#777').attr('font-size', '10px')
                .text(`Total: ${totalBlocks} blocks (${blocksPerSeq.join(' + ')}), ${fmtBytes(totalBlocks * blockBytes)}`);
        }

        svg.attr('height', y + 20);
    }

    redraw();
}
