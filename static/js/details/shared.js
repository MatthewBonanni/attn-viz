// shared.js — Shared utilities and helpers for detail panel visualizations
import { TP_COLORS, RANK_COLORS } from '../render.js';
export { TP_COLORS, RANK_COLORS };

export function detailMetrics() {
    const w = 480;
    return { w, cx: w / 2, pad: 20 };
}

const MAX_DETAIL_CELLS = 10000;

export function maskLayout(svgW, rows, cols) {
    const maxGridW = svgW - 80;
    const maxGridH = 300;
    const rawCellSize = Math.min(28, maxGridW / cols, maxGridH / rows);
    const schematic = rawCellSize < 2 || rows * cols > MAX_DETAIL_CELLS;
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

export function drawDetailBlock3D(g, x, y, w, h, d, color, label, grouping, tpInfo, dpInfo, onCellClick) {
    const dx = d * 0.7;
    const dy = -d * 0.4;

    const tpSize = (tpInfo && tpInfo.tpSize > 1) ? tpInfo.tpSize : 1;
    const dpSize = (dpInfo && dpInfo.dpSize > 1) ? dpInfo.dpSize : 1;
    const dpFracs = (dpInfo && dpInfo.dpFracs) || Array.from({length: dpSize + 1}, (_, i) => i / dpSize);
    const hasParallelism = tpSize > 1 || dpSize > 1;

    // Element registry for edge-aware 3D highlighting.
    const elements = [];
    for (let dp = 0; dp < dpSize; dp++) {
        elements[dp] = [];
        for (let tp = 0; tp < tpSize; tp++) {
            elements[dp][tp] = { front: [], right: [], top: [] };
        }
    }

    const restoreOps = { front: [0.4, 'none', 0], right: [0.5, 'none', 0], top: [0.35, 'none', 0] };
    const highlightOps = { front: [0.65, '#fff', 1.5], right: [0.85, '#fff', 1.5], top: [0.55, '#fff', 1] };

    function setFace(face, dp, tp, on) {
        const [opacity, stroke, sw] = on ? highlightOps[face] : restoreOps[face];
        for (const el of elements[dp][tp][face]) {
            el.attr('fill-opacity', opacity);
            if (stroke === 'none') el.attr('stroke', 'none');
            else el.attr('stroke', stroke).attr('stroke-width', sw);
        }
    }

    function clearAll() {
        for (let dp = 0; dp < dpSize; dp++)
            for (let tp = 0; tp < tpSize; tp++)
                for (const face of ['front', 'right', 'top'])
                    setFace(face, dp, tp, false);
    }

    function highlightRightCell(dp, tp) {
        clearAll();
        setFace('right', dp, tp, true);
        if (tp === 0 && tpSize > 1) {
            for (let t = 0; t < tpSize; t++) setFace('front', dp, t, true);
        }
        if (dp === 0 && tpSize > 1) {
            for (let d2 = 0; d2 < dpSize; d2++) setFace('top', d2, tp, true);
        }
    }

    function highlightFrontStripe(dp) {
        clearAll();
        for (let t = 0; t < tpSize; t++) setFace('front', dp, t, true);
        setFace('right', dp, 0, true);
        if (dp === 0 && tpSize > 1) {
            for (let d2 = 0; d2 < dpSize; d2++) setFace('top', d2, 0, true);
        }
    }

    function highlightTopBand(tp) {
        clearAll();
        for (let d2 = 0; d2 < dpSize; d2++) setFace('top', d2, tp, true);
        setFace('right', 0, tp, true);
        if (tp === 0 && tpSize > 1) {
            for (let t = 0; t < tpSize; t++) setFace('front', 0, t, true);
        }
    }

    // Top face base
    g.append('polygon')
        .attr('points', `${x},${y} ${x+dx},${y+dy} ${x+w+dx},${y+dy} ${x+w},${y}`)
        .attr('fill', d3.color(color).darker(0.4)).attr('stroke', 'none');
    // Right face base
    g.append('polygon')
        .attr('points', `${x+w},${y} ${x+w+dx},${y+dy} ${x+w+dx},${y+h+dy} ${x+w},${y+h}`)
        .attr('fill', d3.color(color).darker(0.8)).attr('stroke', 'none');

    if (hasParallelism) {
        const rx = x + w;

        for (let dp = 0; dp < dpSize; dp++) {
            for (let tp = 0; tp < tpSize; tp++) {
                const rankIdx = dp * tpSize + tp;
                const cellColor = RANK_COLORS[rankIdx % RANK_COLORS.length];
                const tf0 = tp / tpSize, tf1 = (tp + 1) / tpSize;
                const df0 = dpFracs[dp], df1 = dpFracs[dp + 1];

                const cell = g.append('polygon')
                    .attr('points', [
                        `${rx + dx*tf0},${y + h*df0 + dy*tf0}`,
                        `${rx + dx*tf1},${y + h*df0 + dy*tf1}`,
                        `${rx + dx*tf1},${y + h*df1 + dy*tf1}`,
                        `${rx + dx*tf0},${y + h*df1 + dy*tf0}`,
                    ].join(' '))
                    .attr('fill', cellColor).attr('fill-opacity', 0.5).attr('stroke', 'none')
                    .style('cursor', 'pointer');

                elements[dp][tp].right.push(cell);

                cell.on('mouseenter', () => {
                    highlightRightCell(dp, tp);
                    if (onCellClick) onCellClick({ dp, tp, rankIdx });
                });
                cell.on('mouseleave', () => {
                    clearAll();
                    if (onCellClick) onCellClick(null);
                });

                if (dp === 0 && tpSize > 1) {
                    const topEl = g.append('polygon')
                        .attr('points', [
                            `${x + dx*tf0},${y + dy*tf0}`,
                            `${x + dx*tf1},${y + dy*tf1}`,
                            `${x + w + dx*tf1},${y + dy*tf1}`,
                            `${x + w + dx*tf0},${y + dy*tf0}`,
                        ].join(' '))
                        .attr('fill', cellColor).attr('fill-opacity', 0.35).attr('stroke', 'none')
                        .style('cursor', 'default');

                    elements[dp][tp].top.push(topEl);

                    topEl.on('mouseenter', () => {
                        highlightTopBand(tp);
                        if (onCellClick) onCellClick({ dp: null, tp, rankIdx: tp });
                    });
                    topEl.on('mouseleave', () => {
                        clearAll();
                        if (onCellClick) onCellClick(null);
                    });
                }
            }
        }

        for (let tp = 1; tp < tpSize; tp++) {
            const frac = tp / tpSize;
            const lx = dx * frac, ly = dy * frac;
            g.append('polyline')
                .attr('points', `${x+lx},${y+ly} ${x+w+lx},${y+ly} ${x+w+lx},${y+h+ly}`)
                .attr('fill', 'none')
                .attr('stroke', '#fff').attr('stroke-opacity', 0.35)
                .attr('stroke-width', 0.75).attr('stroke-linejoin', 'round');
        }

        for (let dp = 1; dp < dpSize; dp++) {
            const yy = y + h * dpFracs[dp];
            g.append('line')
                .attr('x1', rx).attr('y1', yy)
                .attr('x2', rx + dx).attr('y2', yy + dy)
                .attr('stroke', '#fff').attr('stroke-opacity', 0.35)
                .attr('stroke-width', 0.75);
        }
    }

    // Front face
    g.append('rect')
        .attr('x', x).attr('y', y).attr('width', w).attr('height', h)
        .attr('fill', color).attr('fill-opacity', 0.85)
        .attr('stroke', d3.color(color).darker(0.3)).attr('stroke-width', 1);

    if (hasParallelism) {
        for (let dp = 0; dp < dpSize; dp++) {
            const rankIdx = dp * tpSize;
            const cellColor = RANK_COLORS[rankIdx % RANK_COLORS.length];
            const bandY = y + dpFracs[dp] * h;
            const bandH = (dpFracs[dp + 1] - dpFracs[dp]) * h;

            const cell = g.append('rect')
                .attr('x', x).attr('y', bandY)
                .attr('width', w).attr('height', bandH)
                .attr('fill', dpSize > 1 ? cellColor : 'transparent')
                .attr('fill-opacity', dpSize > 1 ? 0.4 : 0)
                .attr('stroke', 'none')
                .style('cursor', 'pointer');

            elements[dp][0].front.push(cell);

            cell.on('mouseenter', () => {
                highlightFrontStripe(dp);
                if (onCellClick) onCellClick({ dp, tp: null, rankIdx: dp * tpSize });
            });
            cell.on('mouseleave', () => {
                clearAll();
                if (onCellClick) onCellClick(null);
            });
        }
        for (let dp = 1; dp < dpSize; dp++) {
            const ly = y + dpFracs[dp] * h;
            g.append('line')
                .attr('x1', x).attr('y1', ly).attr('x2', x + w).attr('y2', ly)
                .attr('stroke', '#fff').attr('stroke-width', 0.75).attr('stroke-opacity', 0.4);
        }
    }

    g.append('text')
        .attr('class', 'tensor-label')
        .attr('x', x + w / 2).attr('y', y + h / 2 + 4)
        .text(label);
}
