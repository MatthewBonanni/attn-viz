// shared.js — Shared utilities and helpers for detail panel visualizations
import { TP_COLORS } from '../render.js';
export { TP_COLORS };

// Get available detail SVG width and center x, accounting for padding/margins
export function detailMetrics() {
    // Use a fixed width rather than clientWidth — the panel may still be
    // CSS-transitioning from width:0 to width:520px on first open, so
    // clientWidth can return an intermediate value causing content to
    // render at the wrong size.
    const w = 480;
    return { w, cx: w / 2, pad: 20 };
}

// Compute cell size, label density, and whether to use schematic mode for a mask grid.
// In schematic mode the grid fits the panel but individual cells are replaced by shapes.
const MAX_DETAIL_CELLS = 10000;

export function maskLayout(svgW, rows, cols) {
    const maxGridW = svgW - 80;
    const maxGridH = 300;
    const rawCellSize = Math.min(28, maxGridW / cols, maxGridH / rows);
    const schematic = rawCellSize < 2 || rows * cols > MAX_DETAIL_CELLS;
    // In schematic mode, shrink cells so the grid fits; otherwise floor at 2px
    const cellSize = schematic ? Math.min(maxGridW / cols, maxGridH / rows) : rawCellSize;

    let labelEvery;
    if (schematic) labelEvery = 0;
    else if (cellSize >= 16) labelEvery = 1;
    else if (cellSize >= 8) labelEvery = Math.ceil(5 / cellSize) * 2;
    else if (cellSize >= 4) labelEvery = Math.ceil(20 / cellSize);
    else labelEvery = 0;

    return { cellSize, labelEvery, schematic };
}

export function drawDetailBlock(g, x, y, w, h, color, label) {
    g.append('rect')
        .attr('x', x).attr('y', y).attr('width', w).attr('height', h)
        .attr('fill', color).attr('fill-opacity', 0.8)
        .attr('stroke', d3.color(color).darker(0.3)).attr('stroke-width', 1)
        .attr('rx', 3);
    g.append('text')
        .attr('class', 'tensor-label')
        .attr('x', x + w / 2).attr('y', y + h / 2 + 4)
        .text(label);
}

export function drawDetailBlock3D(g, x, y, w, h, d, color, label, grouping, tpInfo) {
    const dx = d * 0.7;
    const dy = -d * 0.4;

    // Top face base
    g.append('polygon')
        .attr('points', `${x},${y} ${x+dx},${y+dy} ${x+w+dx},${y+dy} ${x+w},${y}`)
        .attr('fill', d3.color(color).darker(0.4)).attr('stroke', 'none');
    // Right face base
    g.append('polygon')
        .attr('points', `${x+w},${y} ${x+w+dx},${y+dy} ${x+w+dx},${y+h+dy} ${x+w},${y+h}`)
        .attr('fill', d3.color(color).darker(0.8)).attr('stroke', 'none');

    // TP-colored stripes on depth faces (4D + TP)
    if (grouping && tpInfo && tpInfo.tpSize > 1) {
        const B = grouping.outer;
        const n_h = grouping.inner;
        const tpSize = tpInfo.tpSize;
        const headsPerRank = n_h / tpSize;
        const total = B * n_h;

        for (let b = 0; b < B; b++) {
            for (let r = 0; r < tpSize; r++) {
                const startSlice = b * n_h + r * headsPerRank;
                const endSlice = startSlice + headsPerRank;
                const f0 = startSlice / total;
                const f1 = endSlice / total;
                const tpColor = TP_COLORS[r % TP_COLORS.length];

                // Right face stripe
                const rx0 = dx * f0, ry0 = dy * f0;
                const rx1 = dx * f1, ry1 = dy * f1;
                g.append('polygon')
                    .attr('points', [
                        `${x+w+rx0},${y+ry0}`, `${x+w+rx1},${y+ry1}`,
                        `${x+w+rx1},${y+h+ry1}`, `${x+w+rx0},${y+h+ry0}`
                    ].join(' '))
                    .attr('fill', tpColor).attr('fill-opacity', 0.5).attr('stroke', 'none');

                // Top face stripe
                g.append('polygon')
                    .attr('points', [
                        `${x+rx0},${y+ry0}`, `${x+rx1},${y+ry1}`,
                        `${x+w+rx1},${y+ry1}`, `${x+w+rx0},${y+ry0}`
                    ].join(' '))
                    .attr('fill', tpColor).attr('fill-opacity', 0.35).attr('stroke', 'none');
            }
        }

        // Boundary lines
        for (let i = 1; i < total; i++) {
            const isBatch = (i % n_h === 0);
            const isTp = (i % headsPerRank === 0);
            if (!isBatch && !isTp) continue;
            const frac = i / total;
            const lx = dx * frac, ly = dy * frac;
            const opacity = isBatch ? 0.6 : 0.35;
            const strokeW = isBatch ? 1 : 0.75;
            g.append('polyline')
                .attr('points', `${x+lx},${y+ly} ${x+w+lx},${y+ly} ${x+w+lx},${y+h+ly}`)
                .attr('fill', 'none')
                .attr('stroke', '#fff').attr('stroke-opacity', opacity)
                .attr('stroke-width', strokeW).attr('stroke-linejoin', 'round');
        }
    } else if (grouping) {
        // 4D depth grouping lines (no TP)
        const { outer, inner } = grouping;
        const total = outer * inner;
        const showInnerLines = total <= 16;
        for (let i = 1; i < total; i++) {
            const isBatch = (i % inner === 0);
            if (!showInnerLines && !isBatch) continue;
            const frac = i / total;
            const lx = dx * frac;
            const ly = dy * frac;
            const opacity = isBatch ? 0.6 : 0.25;
            const strokeW = isBatch ? 1 : 0.75;
            g.append('polyline')
                .attr('points', `${x+lx},${y+ly} ${x+w+lx},${y+ly} ${x+w+lx},${y+h+ly}`)
                .attr('fill', 'none')
                .attr('stroke', '#fff').attr('stroke-opacity', opacity)
                .attr('stroke-width', strokeW).attr('stroke-linejoin', 'round');
        }
    }

    // Front face
    g.append('rect')
        .attr('x', x).attr('y', y).attr('width', w).attr('height', h)
        .attr('fill', color).attr('fill-opacity', 0.85)
        .attr('stroke', d3.color(color).darker(0.3)).attr('stroke-width', 1);
    g.append('text')
        .attr('class', 'tensor-label')
        .attr('x', x + w / 2).attr('y', y + h / 2 + 4)
        .text(label);
}
