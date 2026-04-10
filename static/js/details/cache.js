// cache.js — Paged KV cache detail visualization

export function drawPagedCacheDetail(svg, _tensor, params) {
    const bs = params.block_size;
    const ctxLens = params.seqLens.slice(0, params.B);
    const queryLens = params.queryLens.slice(0, params.B);
    const totalLens = ctxLens.map((c, i) => c + queryLens[i]);
    const blocksPerSeq = totalLens.map(s => Math.ceil(s / bs));
    const totalBlocks = blocksPerSeq.reduce((a, b) => a + b, 0);
    const pad = 20;

    const g = svg.append('g').attr('transform', `translate(${pad}, 10)`);

    g.append('text').attr('class', 'tensor-label')
        .attr('x', 130).attr('y', 0)
        .text('Paged KV Cache Layout');

    // Block table section
    let y = 20;
    g.append('text').attr('class', 'dim-label')
        .attr('x', 0).attr('y', y).attr('fill', '#aaa').attr('font-size', '10px')
        .text('Block Table (logical \u2192 physical mapping):');
    y += 16;

    const seqColors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];
    const blockW = Math.min(36, 280 / Math.max(totalBlocks, 4));
    const blockH = 24;

    // Draw block table per sequence
    for (let si = 0; si < totalLens.length; si++) {
        const sLen = totalLens[si];
        const nBlocks = blocksPerSeq[si];
        const color = seqColors[si % seqColors.length];
        const typeStr = queryLens[si] > 1 ? 'prefill' : 'decode';

        g.append('text').attr('class', 'dim-label')
            .attr('x', 0).attr('y', y + blockH / 2 + 3)
            .attr('fill', '#aaa').attr('font-size', '9px')
            .text(`Req ${si} (${ctxLens[si]}+${queryLens[si]}, ${typeStr}):`);

        const tableX = 120;
        for (let bi = 0; bi < nBlocks; bi++) {
            const bx = tableX + bi * (blockW + 3);
            const tokensInBlock = Math.min(bs, sLen - bi * bs);
            const fillRatio = tokensInBlock / bs;

            // Block background (full)
            g.append('rect')
                .attr('x', bx).attr('y', y)
                .attr('width', blockW).attr('height', blockH)
                .attr('fill', '#1e2030')
                .attr('stroke', color).attr('stroke-width', 1.5)
                .attr('rx', 3);

            // Filled portion
            g.append('rect')
                .attr('x', bx + 1).attr('y', y + 1)
                .attr('width', (blockW - 2) * fillRatio).attr('height', blockH - 2)
                .attr('fill', color).attr('fill-opacity', 0.4)
                .attr('rx', 2);

            // Block label
            if (blockW >= 20) {
                g.append('text')
                    .attr('x', bx + blockW / 2).attr('y', y + blockH / 2 + 4)
                    .attr('text-anchor', 'middle')
                    .attr('fill', '#ddd').attr('font-size', '8px')
                    .text(`${tokensInBlock}/${bs}`);
            }
        }
        y += blockH + 8;
    }

    // Physical memory layout
    y += 8;
    g.append('text').attr('class', 'dim-label')
        .attr('x', 0).attr('y', y).attr('fill', '#aaa').attr('font-size', '10px')
        .text('Physical blocks (non-contiguous in memory):');
    y += 16;

    // Draw physical blocks interleaved
    const physicalBlocks = [];
    for (let si = 0; si < totalLens.length; si++) {
        for (let bi = 0; bi < blocksPerSeq[si]; bi++) {
            physicalBlocks.push({ seq: si, block: bi, tokens: Math.min(bs, totalLens[si] - bi * bs) });
        }
    }
    // Shuffle to show non-contiguous nature
    const shuffled = [...physicalBlocks];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = (i * 7 + 3) % (i + 1); // deterministic shuffle
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }

    const physBlockW = Math.min(36, 280 / Math.max(shuffled.length, 4));
    for (let i = 0; i < shuffled.length; i++) {
        const pb = shuffled[i];
        const color = seqColors[pb.seq % seqColors.length];
        const bx = i * (physBlockW + 3);

        g.append('rect')
            .attr('x', bx).attr('y', y)
            .attr('width', physBlockW).attr('height', blockH)
            .attr('fill', color).attr('fill-opacity', 0.35)
            .attr('stroke', color).attr('stroke-width', 1)
            .attr('rx', 3);

        if (physBlockW >= 20) {
            g.append('text')
                .attr('x', bx + physBlockW / 2).attr('y', y + blockH / 2 + 4)
                .attr('text-anchor', 'middle')
                .attr('fill', '#ddd').attr('font-size', '8px')
                .text(`S${pb.seq}`);
        }
    }
    y += blockH + 8;

    // Block shape info
    y += 4;
    const blockDimStr = _tensor.pagedBlockDims
        ? _tensor.pagedBlockDims.map((d, i) => `${d}=${_tensor.pagedBlockShape[i]}`).join(', ')
        : 'n_heads, d_h';
    g.append('text').attr('class', 'dim-label')
        .attr('x', 0).attr('y', y).attr('fill', '#777')
        .text(`Each block: [${bs}, ${blockDimStr}]`);
    y += 14;
    g.append('text').attr('class', 'dim-label')
        .attr('x', 0).attr('y', y).attr('fill', '#777')
        .text(`Total blocks: ${totalBlocks} (${blocksPerSeq.join(' + ')})`);
    y += 14;
    g.append('text').attr('class', 'dim-label')
        .attr('x', 0).attr('y', y).attr('fill', '#777')
        .text(`Block size: ${bs} tokens/block`);

    svg.attr('height', y + 20);
}
