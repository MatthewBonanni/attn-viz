// render.js — Isometric tensor block rendering, layout, arrows, op nodes

const ISO_ANGLE = Math.PI / 6;
const ISO_COS = Math.cos(ISO_ANGLE);
const ISO_SIN = Math.sin(ISO_ANGLE);
const DEPTH_SCALE = 0.4;
const STAGE_GAP = 130;
const ROW_GAP = 30;
const DIM_LABEL_OFFSET = 8;
const OP_RADIUS = 18;
const ARROW_MARGIN = 4;
const ARROWHEAD_LEN = 8;  // must match markerWidth in index.html

// TP rank colors
export const TP_COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#c0392b'];

// --- Scale ---

export function dimScale(value) {
    const scaled = Math.sqrt(value) * 10;
    return Math.max(24, Math.min(180, scaled));
}

// --- Isometric helpers ---

function depthOffset(d) {
    return { dx: d * ISO_COS, dy: -d * ISO_SIN };
}

function polyStr(pts) {
    return pts.map(p => p.join(',')).join(' ');
}

// --- Tensor geometry ---

export function tensorGeometry(shape) {
    let w, h, d;
    if (shape.length === 2) {
        w = dimScale(shape[1]);
        h = dimScale(shape[0]);
        d = 0;
    } else if (shape.length === 3) {
        w = dimScale(shape[2]);
        h = dimScale(shape[1]);
        d = dimScale(shape[0]) * DEPTH_SCALE;
    } else if (shape.length === 4) {
        w = dimScale(shape[3]);
        h = dimScale(shape[2]);
        d = dimScale(shape[0] * shape[1]) * DEPTH_SCALE;
    } else {
        w = 40; h = 40; d = 0;
    }
    return { w, h, d };
}

export function tensorBounds(shape) {
    const { w, h, d } = tensorGeometry(shape);
    const off = depthOffset(d);
    return {
        totalW: w + Math.abs(off.dx),
        totalH: h + Math.abs(off.dy),
        w, h, d
    };
}

// --- Draw tensor block ---

export function drawTensorBlock(g, x, y, tensor, dimNames) {
    const { shape, color, label, type, id } = tensor;
    const { w, h, d } = tensorGeometry(shape);
    const off = depthOffset(d);

    const group = g.append('g')
        .attr('class', 'tensor-block')
        .attr('data-id', id);

    if (d > 0) {
        // Top face
        group.append('polygon')
            .attr('points', polyStr([
                [x, y], [x + off.dx, y + off.dy],
                [x + w + off.dx, y + off.dy], [x + w, y]
            ]))
            .attr('fill', d3.color(color).darker(0.4))
            .attr('stroke', type === 'weight' ? '#aaa' : 'none')
            .attr('stroke-dasharray', type === 'weight' ? '4,2' : 'none')
            .attr('stroke-width', 1);

        // Right face
        group.append('polygon')
            .attr('points', polyStr([
                [x + w, y], [x + w + off.dx, y + off.dy],
                [x + w + off.dx, y + h + off.dy], [x + w, y + h]
            ]))
            .attr('fill', d3.color(color).darker(0.8))
            .attr('stroke', type === 'weight' ? '#aaa' : 'none')
            .attr('stroke-dasharray', type === 'weight' ? '4,2' : 'none')
            .attr('stroke-width', 1);

        // Depth face decorations: TP stripes and/or 4D grouping lines
        if (shape.length === 4 && type !== 'weight' && type !== 'mask' && tensor.tpSharded && tensor.tpSize > 1) {
            draw4DTpDepth(group, x, y, w, h, off, shape[0], shape[1], tensor.tpSize);
        } else if (tensor.tpSharded && tensor.tpSize > 1) {
            drawTpStripes(group, x + w, y, off, h, tensor.tpSize);
        } else if (shape.length === 4 && type !== 'weight' && type !== 'mask') {
            draw4DDepthLines(group, x, y, w, h, off, shape[0], shape[1]);
        }
    }

    // Front face
    if (type === 'mask') {
        if (tensor.pagedMask && tensor.seqLens) {
            drawPagedMaskFace(group, x, y, w, h, tensor.seqLens, tensor.queryLens);
        } else {
            drawMaskFace(group, x, y, w, h, shape, color);
        }
    } else {
        const frontRect = group.append('rect')
            .attr('x', x).attr('y', y)
            .attr('width', w).attr('height', h)
            .attr('fill', type === 'weight' ? d3.color(color).brighter(0.3) : color)
            .attr('fill-opacity', type === 'weight' ? 0.5 : 0.85);
        if (type === 'weight') {
            frontRect.attr('stroke', '#aaa').attr('stroke-width', 1.5)
                .attr('stroke-dasharray', '4,2');
        } else if (d === 0) {
            // Only stroke flat (2D) tensors — 3D faces define their own edges
            frontRect.attr('stroke', d3.color(color).darker(0.3))
                .attr('stroke-width', 1);
        }
    }

    // Tensor name label
    group.append('text')
        .attr('class', 'tensor-label')
        .attr('x', x + w / 2)
        .attr('y', y + h / 2 + 4)
        .text(label);

    // Dimension annotations
    drawDimAnnotations(group, x, y, w, h, d, off, shape, dimNames);

    // Badge
    if (tensor.badge) {
        const bw = tensor.badge.length * 6 + 12;
        const badgeY = y - 18 + (d > 0 ? off.dy : 0);
        group.append('rect')
            .attr('class', 'badge-bg')
            .attr('x', x + w / 2 - bw / 2)
            .attr('y', badgeY)
            .attr('width', bw).attr('height', 14)
            .attr('fill', tensor.badge === 'ABSORBED' ? '#3b5bdb' :
                          tensor.badge === 'LATENT' ? '#9b59b6' :
                          tensor.badge === 'PAGED' ? '#16a085' : '#e67e22');
        group.append('text')
            .attr('class', 'badge')
            .attr('x', x + w / 2)
            .attr('y', badgeY + 10)
            .attr('text-anchor', 'middle')
            .text(tensor.badge);
    }

    // Store position data
    tensor._x = x;
    tensor._y = y;
    tensor._w = w;
    tensor._h = h;
    tensor._d = d;
    tensor._off = off;

    return group;
}

// --- TP sharding stripes on depth face ---

function drawTpStripes(group, rx, fy, off, h, tpSize) {
    for (let r = 0; r < tpSize; r++) {
        group.append('polygon')
            .attr('points', polyStr([
                [rx + r * (off.dx / tpSize), fy + r * (off.dy / tpSize)],
                [rx + (r + 1) * (off.dx / tpSize), fy + (r + 1) * (off.dy / tpSize)],
                [rx + (r + 1) * (off.dx / tpSize), fy + h + (r + 1) * (off.dy / tpSize)],
                [rx + r * (off.dx / tpSize), fy + h + r * (off.dy / tpSize)],
            ]))
            .attr('fill', TP_COLORS[r % TP_COLORS.length])
            .attr('fill-opacity', 0.5)
            .attr('stroke', 'none');
    }
}

// --- 4D depth layer lines (B × n_h grouping) ---

function draw4DDepthLines(group, x, y, w, h, off, B, n_h) {
    const total = B * n_h;
    // Skip individual head lines if too many; only show batch boundaries
    const showHeadLines = total <= 16;
    const slices = total;

    for (let i = 1; i < slices; i++) {
        const isBatchBoundary = (i % n_h === 0);
        if (!showHeadLines && !isBatchBoundary) continue;

        const frac = i / slices;
        const lx = off.dx * frac;
        const ly = off.dy * frac;

        const opacity = isBatchBoundary ? 0.6 : 0.25;
        const strokeW = isBatchBoundary ? 1 : 0.75;

        // L-shaped line across top face and down right face, joined at corner
        group.append('polyline')
            .attr('points', `${x + lx},${y + ly} ${x + w + lx},${y + ly} ${x + w + lx},${y + h + ly}`)
            .attr('fill', 'none')
            .attr('stroke', '#fff').attr('stroke-opacity', opacity)
            .attr('stroke-width', strokeW).attr('stroke-linejoin', 'round');
    }
}

// --- Combined 4D + TP depth rendering ---

function draw4DTpDepth(group, x, y, w, h, off, B, n_h, tpSize) {
    const total = B * n_h;
    const headsPerRank = n_h / tpSize;
    const rx = x + w;  // right face x origin

    // Draw TP-colored stripes on right face and top face
    for (let b = 0; b < B; b++) {
        for (let r = 0; r < tpSize; r++) {
            const startSlice = b * n_h + r * headsPerRank;
            const endSlice = startSlice + headsPerRank;
            const f0 = startSlice / total;
            const f1 = endSlice / total;
            const color = TP_COLORS[r % TP_COLORS.length];

            // Right face stripe
            group.append('polygon')
                .attr('points', polyStr([
                    [rx + off.dx * f0, y + off.dy * f0],
                    [rx + off.dx * f1, y + off.dy * f1],
                    [rx + off.dx * f1, y + h + off.dy * f1],
                    [rx + off.dx * f0, y + h + off.dy * f0],
                ]))
                .attr('fill', color)
                .attr('fill-opacity', 0.5)
                .attr('stroke', 'none');

            // Top face stripe
            group.append('polygon')
                .attr('points', polyStr([
                    [x + off.dx * f0, y + off.dy * f0],
                    [x + off.dx * f1, y + off.dy * f1],
                    [x + w + off.dx * f1, y + off.dy * f1],
                    [x + w + off.dx * f0, y + off.dy * f0],
                ]))
                .attr('fill', color)
                .attr('fill-opacity', 0.35)
                .attr('stroke', 'none');
        }
    }

    // Boundary lines on both faces
    for (let i = 1; i < total; i++) {
        const isBatchBoundary = (i % n_h === 0);
        const isTpBoundary = (i % headsPerRank === 0);
        if (!isBatchBoundary && !isTpBoundary) continue;

        const frac = i / total;
        const lx = off.dx * frac;
        const ly = off.dy * frac;

        const opacity = isBatchBoundary ? 0.6 : 0.35;
        const strokeW = isBatchBoundary ? 1 : 0.75;

        // L-shaped line across top face and down right face, joined at corner
        group.append('polyline')
            .attr('points', `${x + lx},${y + ly} ${x + w + lx},${y + ly} ${x + w + lx},${y + h + ly}`)
            .attr('fill', 'none')
            .attr('stroke', '#fff').attr('stroke-opacity', opacity)
            .attr('stroke-width', strokeW).attr('stroke-linejoin', 'round');
    }
}

// --- Causal mask face ---

function drawMaskFace(group, x, y, w, h, shape, color) {
    const S = shape[shape.length - 1];
    const S_q = shape.length >= 2 ? shape[shape.length - 2] : S;
    const blocked = '#2c3e50';
    // Causal offset: query row i attends to keys 0..S-S_q+i
    const offset = S - S_q;

    if (S <= 16 && S_q <= 16) {
        const cellW = w / S;
        const cellH = h / S_q;
        for (let i = 0; i < S_q; i++) {
            for (let j = 0; j < S; j++) {
                const allowed = j <= offset + i;
                group.append('rect')
                    .attr('x', x + j * cellW)
                    .attr('y', y + i * cellH)
                    .attr('width', cellW)
                    .attr('height', cellH)
                    .attr('fill', allowed ? color : blocked)
                    .attr('fill-opacity', allowed ? 0.85 : 0.6)
                    .attr('stroke', '#1a1d2a')
                    .attr('stroke-width', 0.5);
            }
        }
    } else {
        group.append('rect')
            .attr('x', x).attr('y', y)
            .attr('width', w).attr('height', h)
            .attr('fill', blocked).attr('fill-opacity', 0.6)
            .attr('stroke', d3.color(color).darker(0.3))
            .attr('stroke-width', 1);
        // Causal triangle: diagonal goes from (offset/S * w, 0) to (w, h)
        const diagX = (offset / S) * w;
        group.append('polygon')
            .attr('points', polyStr([
                [x, y], [x, y + h], [x + w, y + h], [x + diagX, y]
            ]))
            .attr('fill', color).attr('fill-opacity', 0.85);
        group.append('line')
            .attr('x1', x + diagX).attr('y1', y)
            .attr('x2', x + w).attr('y2', y + h)
            .attr('stroke', '#fff').attr('stroke-width', 1).attr('stroke-opacity', 0.3);
    }
}

// --- Paged/variable-length mask face ---

function drawPagedMaskFace(group, x, y, w, h, seqLens, queryLens) {
    const sqLens = queryLens || seqLens;
    const totalS = seqLens.reduce((a, b) => a + b, 0);
    const totalSq = sqLens.reduce((a, b) => a + b, 0);
    const cellW = w / totalS;
    const cellH = h / totalSq;
    const color = '#1abc9c';
    const blocked = '#2c3e50';
    const crossSeq = '#1a1520';
    const totalCells = totalSq * totalS;

    let rowOffset = 0;
    for (let si = 0; si < sqLens.length; si++) {
        const qLen = sqLens[si];
        const sLen = seqLens[si];
        let colOffset = 0;

        for (let sj = 0; sj < seqLens.length; sj++) {
            const sLenJ = seqLens[sj];

            for (let i = 0; i < qLen; i++) {
                for (let j = 0; j < sLenJ; j++) {
                    const globalI = rowOffset + i;
                    const globalJ = colOffset + j;
                    let fill, opacity;

                    if (si !== sj) {
                        fill = crossSeq;
                        opacity = 0.8;
                    } else {
                        const allowed = j <= (sLen - qLen + i);
                        fill = allowed ? color : blocked;
                        opacity = allowed ? 0.85 : 0.5;
                    }

                    if (totalCells <= 400) {
                        group.append('rect')
                            .attr('x', x + globalJ * cellW)
                            .attr('y', y + globalI * cellH)
                            .attr('width', cellW)
                            .attr('height', cellH)
                            .attr('fill', fill)
                            .attr('fill-opacity', opacity)
                            .attr('stroke', '#1a1d2a')
                            .attr('stroke-width', 0.3);
                    }
                }
            }
            colOffset += sLenJ;
        }
        rowOffset += qLen;
    }

    // If too many cells, draw simplified block-diagonal
    if (totalCells > 400) {
        group.append('rect')
            .attr('x', x).attr('y', y)
            .attr('width', w).attr('height', h)
            .attr('fill', crossSeq).attr('fill-opacity', 0.8);

        let rowOff = 0, colOff = 0;
        for (let si = 0; si < sqLens.length; si++) {
            const qLen = sqLens[si];
            const sLen = seqLens[si];
            const bx = x + colOff * cellW;
            const by = y + rowOff * cellH;
            const bw = sLen * cellW;
            const bh = qLen * cellH;
            // Causal triangle within block
            group.append('polygon')
                .attr('points', polyStr([[bx, by], [bx, by + bh], [bx + bw, by + bh]]))
                .attr('fill', color).attr('fill-opacity', 0.85);
            // Upper triangle
            group.append('polygon')
                .attr('points', polyStr([[bx, by], [bx + bw, by], [bx + bw, by + bh]]))
                .attr('fill', blocked).attr('fill-opacity', 0.5);
            group.append('line')
                .attr('x1', bx).attr('y1', by)
                .attr('x2', bx + bw).attr('y2', by + bh)
                .attr('stroke', '#fff').attr('stroke-width', 0.5).attr('stroke-opacity', 0.3);
            rowOff += qLen;
            colOff += sLen;
        }
    }

    // Border
    group.append('rect')
        .attr('x', x).attr('y', y)
        .attr('width', w).attr('height', h)
        .attr('fill', 'none')
        .attr('stroke', '#555').attr('stroke-width', 1);
}

// --- Dimension annotations ---

function drawDimAnnotations(group, x, y, w, h, d, off, shape, dimNames) {
    if (!dimNames || dimNames.length === 0) return;

    const names = [...dimNames];

    if (shape.length === 2) {
        annotateEdge(group, x, y + h, x + w, y + h, names[1] || '', shape[1], 'bottom');
        annotateEdge(group, x, y, x, y + h, names[0] || '', shape[0], 'left');
    } else if (shape.length === 3) {
        annotateEdge(group, x, y + h, x + w, y + h, names[2] || '', shape[2], 'bottom');
        annotateEdge(group, x, y, x, y + h, names[1] || '', shape[1], 'left');
        if (d > 0) {
            annotateEdge(group, x + w, y, x + w + off.dx, y + off.dy, names[0] || '', shape[0], 'depth');
        }
    } else if (shape.length === 4) {
        annotateEdge(group, x, y + h, x + w, y + h, names[3] || '', shape[3], 'bottom');
        annotateEdge(group, x, y, x, y + h, names[2] || '', shape[2], 'left');
        if (d > 0) {
            const depthLabel = `${names[0] || ''}·${names[1] || ''}`;
            annotateEdge(group, x + w, y, x + w + off.dx, y + off.dy,
                depthLabel, shape[0] * shape[1], 'depth');
        }
    }
}

function annotateEdge(group, x1, y1, x2, y2, name, value, position) {
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    let tx, ty, anchor;

    if (position === 'bottom') {
        tx = mx; ty = my + DIM_LABEL_OFFSET; anchor = 'middle';
    } else if (position === 'left') {
        tx = x1 - 3; ty = my + 14; anchor = 'end';
    } else if (position === 'depth') {
        tx = mx + 8; ty = my - 6; anchor = 'start';
    }

    const text = name ? `${name}=${value}` : `${value}`;
    group.append('text')
        .attr('class', 'dim-label')
        .attr('x', tx).attr('y', ty)
        .attr('text-anchor', anchor)
        .text(text);
}

// --- Shared stage positions (for aligning stacked graphs) ---

export function computeSharedStagePositions(...graphs) {
    const stageMaxW = {};

    for (const graph of graphs) {
        const stages = {};
        for (const t of graph.tensors) {
            if (!stages[t.stage]) stages[t.stage] = [];
            stages[t.stage].push(t);
        }
        for (const [sk, tensors] of Object.entries(stages)) {
            const maxW = Math.max(...tensors.map(t => tensorBounds(t.shape).totalW));
            stageMaxW[sk] = Math.max(stageMaxW[sk] || 0, maxW);
        }
    }

    const stageKeys = Object.keys(stageMaxW).map(Number).sort((a, b) => a - b);
    const positions = {};
    let xCursor = 50;

    for (const sk of stageKeys) {
        positions[sk] = { x: xCursor, w: stageMaxW[sk] };
        xCursor += stageMaxW[sk] + STAGE_GAP;
    }

    return positions;
}

// --- Layout ---

export function computeLayout(graph, sharedStageX) {
    const stages = {};
    for (const t of graph.tensors) {
        if (!stages[t.stage]) stages[t.stage] = [];
        stages[t.stage].push(t);
    }

    const stageKeys = Object.keys(stages).map(Number).sort((a, b) => a - b);

    const stageInfo = {};
    for (const sk of stageKeys) {
        const tensors = stages[sk];
        tensors.sort((a, b) => a.row - b.row);
        stageInfo[sk] = {
            tensors,
            maxW: Math.max(...tensors.map(t => tensorBounds(t.shape).totalW)),
        };
    }

    // Row-aligned layout: tensors with the same row number share
    // the same y-position across all stages (grid-based alignment).
    const rowMaxH = {};
    for (const t of graph.tensors) {
        const bounds = tensorBounds(t.shape);
        rowMaxH[t.row] = Math.max(rowMaxH[t.row] || 0, bounds.totalH);
    }
    const rowKeys = Object.keys(rowMaxH).map(Number).sort((a, b) => a - b);

    let totalH = 0;
    for (const rk of rowKeys) totalH += rowMaxH[rk] + ROW_GAP;
    totalH -= ROW_GAP;

    const centerY = totalH / 2 + 80;
    const rowY = {};
    let rCursor = centerY - totalH / 2;
    for (const rk of rowKeys) {
        rowY[rk] = rCursor;
        rCursor += rowMaxH[rk] + ROW_GAP;
    }

    // Store stage x-ranges for arrow routing
    const stageXRanges = {};

    if (sharedStageX) {
        for (const sk of stageKeys) {
            const { tensors } = stageInfo[sk];
            const shared = sharedStageX[sk];
            const stageX = shared.x;
            const stageW = shared.w;

            for (const t of tensors) {
                const bounds = tensorBounds(t.shape);
                const geo = tensorGeometry(t.shape);
                const off = depthOffset(geo.d);

                t._layoutX = stageX + (stageW - bounds.totalW) / 2;
                t._layoutY = rowY[t.row] + (rowMaxH[t.row] - bounds.totalH) / 2 + Math.abs(off.dy);
            }

            stageXRanges[sk] = { left: stageX, right: stageX + stageW };
        }
    } else {
        let xCursor = 50;

        for (const sk of stageKeys) {
            const { tensors, maxW } = stageInfo[sk];

            for (const t of tensors) {
                const bounds = tensorBounds(t.shape);
                const geo = tensorGeometry(t.shape);
                const off = depthOffset(geo.d);

                t._layoutX = xCursor + (maxW - bounds.totalW) / 2;
                t._layoutY = rowY[t.row] + (rowMaxH[t.row] - bounds.totalH) / 2 + Math.abs(off.dy);
            }

            stageXRanges[sk] = { left: xCursor, right: xCursor + maxW };
            xCursor += maxW + STAGE_GAP;
        }
    }

    // Position ops between their inputs and outputs
    const tensorMap = {};
    for (const t of graph.tensors) tensorMap[t.id] = t;

    // Collect all tensor bounding boxes for arrow avoidance
    graph._tensorRects = graph.tensors.map(t => {
        if (t._layoutX == null) return null;
        const bounds = tensorBounds(t.shape);
        const geo = tensorGeometry(t.shape);
        const off = depthOffset(geo.d);
        return {
            left: t._layoutX,
            right: t._layoutX + bounds.totalW,
            top: t._layoutY + Math.min(0, off.dy),
            bottom: t._layoutY + geo.h + DIM_LABEL_OFFSET,
            id: t.id,
        };
    }).filter(Boolean);

    for (const op of graph.ops) {
        const inputs = op.inputs.map(id => tensorMap[id]).filter(Boolean);
        const output = tensorMap[op.output];
        if (!output || inputs.length === 0) continue;

        const inRight = Math.max(...inputs.map(t => {
            const b = tensorBounds(t.shape);
            return (t._layoutX || 0) + b.totalW;
        }));
        const outLeft = output._layoutX || 0;
        op._x = (inRight + outLeft) / 2;

        // Align op with its output tensor's y-center so linear ops
        // sit on the same row as their weight and output tensors.
        op._y = (output._layoutY || 0) + tensorGeometry(output.shape).h / 2;
    }

    // Align ops that share the same alignX group
    const alignGroups = {};
    for (const op of graph.ops) {
        if (op.alignX) {
            if (!alignGroups[op.alignX]) alignGroups[op.alignX] = [];
            alignGroups[op.alignX].push(op);
        }
    }
    for (const group of Object.values(alignGroups)) {
        const maxX = Math.max(...group.map(o => o._x));
        for (const op of group) op._x = maxX;
    }

    // Clamp ops so circles don't overlap output tensor's left dim labels.
    // Left labels use text-anchor:end and extend leftward from the tensor edge,
    // so we must account for the approximate text width of the label.
    for (const op of graph.ops) {
        const output = tensorMap[op.output];
        if (!output || output._layoutX == null || op._x == null) continue;
        // Estimate left dim label text width (format: "name=value", ~6px per char at 9px font)
        let leftLabelW = 0;
        if (output.dimNames && output.shape) {
            const dimIdx = output.shape.length >= 3 ? output.shape.length - 2 : 0;
            const name = output.dimNames[dimIdx] || '';
            const value = output.shape[dimIdx];
            const text = name ? `${name}=${value}` : `${value}`;
            leftLabelW = text.length * 6;
        }
        const maxX = (output._layoutX || 0) - OP_RADIUS - DIM_LABEL_OFFSET - ARROW_MARGIN - leftLabelW;
        if (op._x > maxX) op._x = maxX;
    }

    graph._stageXRanges = stageXRanges;
    graph._centerY = centerY;
    graph._maxTotalH = totalH;
}

// --- Draw op nodes ---

export function drawOpNode(g, op, onClick) {
    const group = g.append('g')
        .attr('class', 'op-node')
        .attr('data-id', op.id)
        .on('click', (event) => {
            event.stopPropagation();
            onClick(op);
        });

    group.append('circle')
        .attr('cx', op._x).attr('cy', op._y)
        .attr('r', OP_RADIUS)
        .attr('fill', '#1e2030')
        .attr('stroke', opColor(op.type))
        .attr('stroke-width', 2);

    const sym = opSymbol(op.type);
    // Per-glyph nudges: some Unicode symbols have asymmetric shapes
    // that shift their visual center away from the text anchor
    const glyphTweak = { rope: { dx: 2, dy: 6, size: 26 },
                         mask: { dx: 0, dy: 6, size: 16 } };
    const tw = glyphTweak[op.type] || { dx: 1, dy: 7, size: 21 };
    const label = group.append('text')
        .attr('class', 'op-label')
        .attr('x', op._x + tw.dx)
        .attr('y', op._y + tw.dy)
        .text(sym);
    if (tw.size) label.style('font-size', tw.size + 'px');

    group.append('text')
        .attr('class', 'dim-label')
        .attr('x', op._x).attr('y', op._y + OP_RADIUS + 12)
        .attr('text-anchor', 'middle')
        .attr('fill', '#777')
        .text(op.label);

    // TP all-reduce marker
    if (op.tpAllReduce) {
        group.append('text')
            .attr('class', 'badge')
            .attr('x', op._x).attr('y', op._y - OP_RADIUS - 6)
            .attr('text-anchor', 'middle')
            .attr('fill', '#3498db')
            .attr('font-size', '8px')
            .text('ALL-REDUCE');
    }

    return group;
}

function opColor(type) {
    const colors = {
        matmul: '#e74c3c', mask: '#1abc9c', softmax: '#f39c12',
        broadcast: '#3498db', reshape: '#95a5a6',
        compress: '#e67e22', decompress: '#e67e22',
        rope: '#ff7043', add: '#3498db',
        cache: '#16a085',
    };
    return colors[type] || '#95a5a6';
}

function opSymbol(type) {
    const symbols = {
        matmul: '×', mask: '▽', softmax: 'σ',
        broadcast: '⇒', reshape: '↺',
        compress: '↓', decompress: '↑',
        rope: '⟳', add: '+',
        cache: '⤓',
    };
    return symbols[type] || '?';
}

// --- Draw arrows (with routing to avoid tensor overlap) ---

export function drawArrows(g, graph) {
    const tensorMap = {};
    for (const t of graph.tensors) tensorMap[t.id] = t;

    for (const op of graph.ops) {
        // Arrows from inputs to op
        const validInputs = op.inputs.filter(id => tensorMap[id] && tensorMap[id]._layoutX != null);
        for (let idx = 0; idx < op.inputs.length; idx++) {
            const t = tensorMap[op.inputs[idx]];
            if (!t || t._layoutX == null) continue;
            const bounds = tensorBounds(t.shape);
            const geo = tensorGeometry(t.shape);
            const sx = t._layoutX + bounds.totalW + ARROW_MARGIN;
            const sy = t._layoutY + geo.h / 2;

            // Offset multiple arrows entering an op so arrowheads don't overlap
            const yOff = validInputs.length > 1 ? (idx - (validInputs.length - 1) / 2) * 8 : 0;

            // Pull back endpoint by arrowhead length so tip touches the op circle
            const forceBelow = op.routeBelow && op.routeBelow.includes(t.id);
            drawRoutedArrow(g, sx, sy, op._x - OP_RADIUS - ARROWHEAD_LEN, op._y + yOff, graph, [t.id], forceBelow);
        }

        // Arrow from op to output
        const out = tensorMap[op.output];
        if (!out || out._layoutX == null) continue;
        const outGeo = tensorGeometry(out.shape);
        // Pull back endpoint by arrowhead length so tip touches the tensor edge
        drawRoutedArrow(g, op._x + OP_RADIUS + ARROW_MARGIN, op._y,
            out._layoutX - ARROWHEAD_LEN, out._layoutY + outGeo.h / 2, graph, [out.id]);

    }
}

// Check if a straight line from (x1,y1) to (x2,y2) passes through a rect
function lineHitsRect(x1, y1, x2, y2, rect, margin) {
    // Rect must be horizontally between arrow endpoints
    if (rect.right <= x1 || rect.left >= x2) return false;
    // Compute arrow's y at the rect's horizontal center
    const cx = Math.max(x1, Math.min(x2, (rect.left + rect.right) / 2));
    const t = (x2 === x1) ? 0.5 : (cx - x1) / (x2 - x1);
    const arrowY = y1 + t * (y2 - y1);
    return arrowY >= rect.top - margin && arrowY <= rect.bottom + margin;
}

function drawRoutedArrow(g, x1, y1, x2, y2, graph, excludeIds, forceBelow) {
    const COLLISION_MARGIN = 8;
    let d;

    if (graph._tensorRects) {
        const colliders = graph._tensorRects.filter(r => {
            if (excludeIds && excludeIds.includes(r.id)) return false;
            return lineHitsRect(x1, y1, x2, y2, r, COLLISION_MARGIN);
        });

        if (colliders.length > 0) {
            const topBound = Math.min(...colliders.map(r => r.top)) - 20;
            const bottomBound = Math.max(...colliders.map(r => r.bottom)) + 20;
            // Route toward the side closest to where the arrow starts,
            // unless forceBelow is set
            const routeY = forceBelow ? bottomBound
                : y1 <= (topBound + bottomBound) / 2 ? topBound : bottomBound;

            const dx = x2 - x1;
            const bend = Math.min(dx * 0.2, 30);
            d = `M${x1},${y1} C${x1 + bend},${y1} ${x1 + bend},${routeY} ${(x1+x2)/2},${routeY} S${x2 - bend},${y2} ${x2},${y2}`;
        }
    }

    if (!d) {
        const dx = x2 - x1;
        const cpx = dx * 0.4;
        d = `M${x1},${y1} C${x1 + cpx},${y1} ${x2 - cpx},${y2} ${x2},${y2}`;
    }

    const ag = g.append('g').attr('class', 'arrow-group');
    ag.append('path').attr('class', 'arrow-hit').attr('d', d);
    ag.append('path').attr('class', 'arrow-path').attr('d', d);
    // Inline arrowhead so CSS :hover on the group highlights it
    ag.append('polygon').attr('class', 'arrow-head')
        .attr('points', `${x2},${y2-3} ${x2+ARROWHEAD_LEN},${y2} ${x2},${y2+3}`);
}

// --- Group enclosures ---

function drawGroupEnclosures(g, graph, onGroupClick, deselectScope) {
    if (!graph.groups) return;

    const tensorMap = {};
    for (const t of graph.tensors) tensorMap[t.id] = t;
    const opMap = {};
    for (const op of graph.ops) opMap[op.id] = op;

    for (const group of graph.groups) {
        const memberRects = [];

        // Collect tensor bounding boxes
        for (const tid of (group.tensors || [])) {
            const t = tensorMap[tid];
            if (!t || t._layoutX == null) continue;
            const b = tensorBounds(t.shape);
            const geo = tensorGeometry(t.shape);
            memberRects.push({
                left: t._layoutX,
                top: t._layoutY,
                right: t._layoutX + b.totalW,
                bottom: t._layoutY + geo.h + DIM_LABEL_OFFSET,
            });
        }

        // Collect op bounding boxes
        for (const oid of (group.ops || [])) {
            const op = opMap[oid];
            if (!op || op._x == null) continue;
            memberRects.push({
                left: op._x - OP_RADIUS,
                top: op._y - OP_RADIUS,
                right: op._x + OP_RADIUS,
                bottom: op._y + OP_RADIUS,
            });
        }

        if (memberRects.length === 0) continue;

        const pad = 10;
        const extraTop = group.padTop || 0;
        const x0 = Math.min(...memberRects.map(r => r.left)) - pad - 40;
        const y0 = Math.min(...memberRects.map(r => r.top)) + 4 - extraTop;
        const x1 = Math.max(...memberRects.map(r => r.right)) + pad + 40;
        const y1 = Math.max(...memberRects.map(r => r.bottom)) + pad;
        const labelH = 14;
        const color = group.color || '#888';

        const groupG = g.append('g')
            .attr('class', 'group-enclosure');

        // Visible border
        const borderEl = groupG.append('rect')
            .attr('x', x0).attr('y', y0 - labelH)
            .attr('width', x1 - x0).attr('height', y1 - y0 + labelH)
            .attr('rx', 8).attr('ry', 8)
            .attr('fill', 'none')
            .attr('stroke', color)
            .attr('stroke-width', 1.5)
            .attr('stroke-dasharray', '6,4')
            .attr('stroke-opacity', 0.4);

        // Thick invisible stroke for click/hover hit area on the border
        groupG.append('rect')
            .attr('x', x0).attr('y', y0 - labelH)
            .attr('width', x1 - x0).attr('height', y1 - y0 + labelH)
            .attr('rx', 8).attr('ry', 8)
            .attr('fill', 'none')
            .attr('stroke', 'transparent')
            .attr('stroke-width', 12)
            .style('cursor', 'pointer')
            .style('pointer-events', 'stroke');

        // Label text
        groupG.append('text')
            .attr('x', (x0 + x1) / 2).attr('y', y0 - 4)
            .attr('text-anchor', 'middle')
            .attr('fill', color)
            .attr('font-size', '10px')
            .attr('font-weight', 'bold')
            .attr('fill-opacity', 0.7)
            .style('cursor', 'pointer')
            .text(group.label);

        // Label background hit area
        groupG.append('rect')
            .attr('x', x0).attr('y', y0 - labelH)
            .attr('width', x1 - x0).attr('height', labelH)
            .attr('fill', 'transparent')
            .style('cursor', 'pointer');

        groupG.on('mouseenter', function() {
            borderEl.attr('stroke-opacity', 0.8).attr('stroke-width', 2);
        });
        groupG.on('mouseleave', function() {
            if (!d3.select(this).classed('selected')) {
                borderEl.attr('stroke-opacity', 0.4).attr('stroke-width', 1.5);
            }
        });

        groupG.on('click', (event) => {
            event.stopPropagation();
            const scope = deselectScope || g;
            scope.selectAll('.tensor-block').classed('selected', false).attr('filter', null);
            scope.selectAll('.op-node').classed('selected', false);
            scope.selectAll('.group-enclosure').classed('selected', false)
                .each(function() {
                    d3.select(this).selectAll('rect,path').filter(function() {
                        return d3.select(this).attr('stroke-dasharray');
                    }).attr('stroke-opacity', 0.4).attr('stroke-width', 1.5);
                });
            groupG.classed('selected', true);
            borderEl.attr('stroke-opacity', 0.8).attr('stroke-width', 2);
            if (onGroupClick) onGroupClick(group);
        });
    }
}

// --- Full render ---

export function renderGraph(g, graph, _params, onOpClick, onTensorClick, deselectScope, sharedStageX, onGroupClick) {
    g.selectAll('*').remove();

    computeLayout(graph, sharedStageX);

    // Draw arrows first (behind everything)
    drawArrows(g, graph);

    // Draw dashed enclosure around KV cache tensors
    const cacheTensors = graph.tensors.filter(t => t.cache && t._layoutX != null);
    if (cacheTensors.length > 0) {
        const pad = 12;
        const cacheRects = cacheTensors.map(t => {
            const b = tensorBounds(t.shape);
            const geo = tensorGeometry(t.shape);
            return { left: t._layoutX, top: t._layoutY - 18, right: t._layoutX + b.totalW, bottom: t._layoutY + geo.h + DIM_LABEL_OFFSET };
        });
        const cx0 = Math.min(...cacheRects.map(r => r.left)) - pad - 20;
        const cy0 = Math.min(...cacheRects.map(r => r.top)) - pad;
        const cx1 = Math.max(...cacheRects.map(r => r.right)) + pad + 40;
        const cy1 = Math.max(...cacheRects.map(r => r.bottom)) + pad;
        const cLabelH = 16;
        const cacheColor = '#16a085';

        const cacheG = g.append('g').attr('class', 'group-enclosure');

        const cacheBorderRect = cacheG.append('rect')
            .attr('x', cx0).attr('y', cy0 - cLabelH)
            .attr('width', cx1 - cx0).attr('height', cy1 - cy0 + cLabelH)
            .attr('rx', 8).attr('ry', 8)
            .attr('fill', 'none')
            .attr('stroke', cacheColor)
            .attr('stroke-width', 1.5)
            .attr('stroke-dasharray', '6,4')
            .attr('stroke-opacity', 0.5);

        // Thick invisible stroke hit area on border
        cacheG.append('rect')
            .attr('x', cx0).attr('y', cy0 - cLabelH)
            .attr('width', cx1 - cx0).attr('height', cy1 - cy0 + cLabelH)
            .attr('rx', 8).attr('ry', 8)
            .attr('fill', 'none')
            .attr('stroke', 'transparent')
            .attr('stroke-width', 12)
            .style('cursor', 'pointer')
            .style('pointer-events', 'stroke');

        cacheG.append('text')
            .attr('x', (cx0 + cx1) / 2).attr('y', cy0 - 4)
            .attr('text-anchor', 'middle')
            .attr('fill', cacheColor)
            .attr('font-size', '10px')
            .attr('font-weight', 'bold')
            .attr('fill-opacity', 0.7)
            .style('cursor', 'pointer')
            .text('KV CACHE');

        // Label background hit area
        cacheG.append('rect')
            .attr('x', cx0).attr('y', cy0 - cLabelH)
            .attr('width', cx1 - cx0).attr('height', cLabelH)
            .attr('fill', 'transparent')
            .style('cursor', 'pointer');

        const cacheDesc = cacheTensors.map(t => `${t.label} ${t.shape.map((d,i) => t.dimNames?.[i] ? t.dimNames[i]+'='+d : d).join('×')}`).join(', ');
        const kvCacheGroup = {
            label: 'KV CACHE',
            desc: `The KV cache stores previously computed keys and values so they don't need to be recomputed at each generation step. Currently holding: ${cacheDesc}. New tokens (S_q) are appended each step; the full S tokens are used as context for attention.`,
            color: cacheColor,
        };

        cacheG.on('mouseenter', function() {
            cacheBorderRect.attr('stroke-opacity', 0.8).attr('stroke-width', 2);
        });
        cacheG.on('mouseleave', function() {
            if (!d3.select(this).classed('selected')) {
                cacheBorderRect.attr('stroke-opacity', 0.5).attr('stroke-width', 1.5);
            }
        });
        cacheG.on('click', (event) => {
            event.stopPropagation();
            const scope = deselectScope || g;
            scope.selectAll('.tensor-block').classed('selected', false).attr('filter', null);
            scope.selectAll('.op-node').classed('selected', false);
            scope.selectAll('.group-enclosure').classed('selected', false)
                .each(function() { d3.select(this).selectAll('rect,path').filter(function() { return d3.select(this).attr('stroke-dasharray'); }).attr('stroke-opacity', 0.4).attr('stroke-width', 1.5); });
            cacheG.classed('selected', true);
            cacheBorderRect.attr('stroke-opacity', 0.8).attr('stroke-width', 2);
            if (onGroupClick) onGroupClick(kvCacheGroup);
        });
    }

    // Draw dashed enclosures around groups
    drawGroupEnclosures(g, graph, onGroupClick, deselectScope);

    // Draw ops before tensors so tensor dim labels aren't hidden behind op circles
    for (const op of graph.ops) {
        if (op._x == null) continue;
        const opGroup = drawOpNode(g, op, (clickedOp) => {
            const scope = deselectScope || g;
            scope.selectAll('.tensor-block').classed('selected', false).attr('filter', null);
            scope.selectAll('.op-node').classed('selected', false);
            scope.selectAll('.group-enclosure').classed('selected', false)
                .each(function() { d3.select(this).selectAll('rect,path').filter(function() { return d3.select(this).attr('stroke-dasharray'); }).attr('stroke-opacity', 0.4).attr('stroke-width', 1.5); });
            opGroup.classed('selected', true);
            if (onOpClick) onOpClick(clickedOp);
        });
    }

    // Draw tensors (on top of ops so dim labels are visible)
    for (const t of graph.tensors) {
        if (t._layoutX == null) continue;
        const dimNames = t.dimNames || [];
        const block = drawTensorBlock(g, t._layoutX, t._layoutY, t, dimNames);

        block.on('click', (event) => {
            event.stopPropagation();
            const scope = deselectScope || g;
            scope.selectAll('.tensor-block').classed('selected', false).attr('filter', null);
            scope.selectAll('.op-node').classed('selected', false);
            scope.selectAll('.group-enclosure').classed('selected', false)
                .each(function() { d3.select(this).selectAll('rect,path').filter(function() { return d3.select(this).attr('stroke-dasharray'); }).attr('stroke-opacity', 0.4).attr('stroke-width', 1.5); });
            block.classed('selected', true).attr('filter', 'url(#selected-glow)');
            if (onTensorClick) onTensorClick(t);
        });

        block.on('mouseenter', function() {
            // Apply hover glow unless already selected
            if (!d3.select(this).classed('selected')) {
                d3.select(this).attr('filter', 'url(#hover-glow)');
            }
        });

        block.on('mouseleave', function() {
            // Remove hover glow unless selected
            if (!d3.select(this).classed('selected')) {
                d3.select(this).attr('filter', null);
            }
        });

        block.on('mouseenter.tooltip', () => {
            const tooltip = d3.select('#tooltip');
            const shapeStr = `[${t.shape.join(', ')}]`;
            const dimStr = t.dimNames ? t.dimNames.map((n, i) => `${n}=${t.shape[i]}`).join(', ') : '';
            tooltip.select('.tt-label').text(t.label);
            tooltip.select('.tt-shape').text(`Shape: ${shapeStr}`);
            tooltip.select('.tt-dims').text(dimStr);
            tooltip.select('.tt-desc').text(t.desc || '');
            tooltip.classed('visible', true);
        });

        block.on('mousemove.tooltip', (event) => {
            d3.select('#tooltip')
                .style('left', (event.clientX + 12) + 'px')
                .style('top', (event.clientY - 10) + 'px');
        });

        block.on('mouseleave.tooltip', () => {
            d3.select('#tooltip').classed('visible', false);
        });
    }
}
