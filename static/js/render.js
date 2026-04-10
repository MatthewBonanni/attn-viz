// render.js — Isometric tensor block rendering, layout, arrows, op nodes

const ISO_ANGLE = Math.PI / 6;
const ISO_COS = Math.cos(ISO_ANGLE);
const ISO_SIN = Math.sin(ISO_ANGLE);
const DEPTH_SCALE = 0.4;
const STAGE_GAP = 100;
const ROW_GAP = 30;
const DIM_LABEL_OFFSET = 14;
const OP_RADIUS = 18;
const ARROW_MARGIN = 4;
const ARROWHEAD_LEN = 8;  // must match markerWidth in index.html

// TP rank colors
const TP_COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#c0392b'];

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

        // TP sharding stripes on right face
        if (tensor.tpSharded && tensor.tpSize > 1) {
            drawTpStripes(group, x + w, y, off, h, tensor.tpSize);
        }
    }

    // Front face
    if (type === 'mask') {
        if (tensor.pagedMask && tensor.seqLens) {
            drawPagedMaskFace(group, x, y, w, h, tensor.seqLens);
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

// --- Causal mask face ---

function drawMaskFace(group, x, y, w, h, shape, color) {
    const S = shape[shape.length - 1];
    const blocked = '#2c3e50';

    if (S <= 16) {
        const cellW = w / S;
        const cellH = h / S;
        for (let i = 0; i < S; i++) {
            for (let j = 0; j < S; j++) {
                group.append('rect')
                    .attr('x', x + j * cellW)
                    .attr('y', y + i * cellH)
                    .attr('width', cellW)
                    .attr('height', cellH)
                    .attr('fill', i >= j ? color : blocked)
                    .attr('fill-opacity', i >= j ? 0.85 : 0.6)
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
        group.append('polygon')
            .attr('points', polyStr([[x, y], [x, y + h], [x + w, y + h]]))
            .attr('fill', color).attr('fill-opacity', 0.85);
        group.append('line')
            .attr('x1', x).attr('y1', y)
            .attr('x2', x + w).attr('y2', y + h)
            .attr('stroke', '#fff').attr('stroke-width', 1).attr('stroke-opacity', 0.3);
    }
}

// --- Paged/variable-length mask face ---

function drawPagedMaskFace(group, x, y, w, h, seqLens) {
    const totalS = seqLens.reduce((a, b) => a + b, 0);
    const cellW = w / totalS;
    const cellH = h / totalS;
    const color = '#1abc9c';
    const blocked = '#2c3e50';
    const crossSeq = '#1a1520';

    let rowOffset = 0;
    for (let si = 0; si < seqLens.length; si++) {
        const sLen = seqLens[si];
        let colOffset = 0;

        for (let sj = 0; sj < seqLens.length; sj++) {
            const sLenJ = seqLens[sj];

            for (let i = 0; i < sLen; i++) {
                for (let j = 0; j < sLenJ; j++) {
                    const globalI = rowOffset + i;
                    const globalJ = colOffset + j;
                    let fill, opacity;

                    if (si !== sj) {
                        fill = crossSeq;
                        opacity = 0.8;
                    } else if (i >= j) {
                        fill = color;
                        opacity = 0.85;
                    } else {
                        fill = blocked;
                        opacity = 0.5;
                    }

                    if (totalS <= 20) {
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
        rowOffset += sLen;
    }

    // If too many tokens, draw simplified block-diagonal
    if (totalS > 20) {
        group.append('rect')
            .attr('x', x).attr('y', y)
            .attr('width', w).attr('height', h)
            .attr('fill', crossSeq).attr('fill-opacity', 0.8);

        let offset = 0;
        for (let si = 0; si < seqLens.length; si++) {
            const sLen = seqLens[si];
            const bx = x + offset * cellW;
            const by = y + offset * cellH;
            const bw = sLen * cellW;
            const bh = sLen * cellH;
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
            offset += sLen;
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
        tx = x1 - DIM_LABEL_OFFSET; ty = my + 3; anchor = 'end';
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

// --- Layout ---

export function computeLayout(graph) {
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
        let totalH = 0;
        for (const t of tensors) {
            totalH += tensorBounds(t.shape).totalH + ROW_GAP;
        }
        totalH -= ROW_GAP;
        stageInfo[sk] = {
            tensors,
            totalH,
            maxW: Math.max(...tensors.map(t => tensorBounds(t.shape).totalW)),
        };
    }

    const maxTotalH = Math.max(...Object.values(stageInfo).map(s => s.totalH));
    const centerY = maxTotalH / 2 + 80;

    // Store stage x-ranges for arrow routing
    const stageXRanges = {};
    let xCursor = 50;

    for (const sk of stageKeys) {
        const { tensors, totalH, maxW } = stageInfo[sk];
        const startY = centerY - totalH / 2;

        let yCursor = startY;
        for (const t of tensors) {
            const bounds = tensorBounds(t.shape);
            const geo = tensorGeometry(t.shape);
            const off = depthOffset(geo.d);

            t._layoutX = xCursor + (maxW - bounds.totalW) / 2;
            t._layoutY = yCursor + Math.abs(off.dy);
            yCursor += bounds.totalH + ROW_GAP;
        }

        stageXRanges[sk] = { left: xCursor, right: xCursor + maxW };
        xCursor += maxW + STAGE_GAP;
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

        const allY = [...inputs, output].map(t => (t._layoutY || 0) + tensorGeometry(t.shape).h / 2);
        op._y = allY.reduce((a, b) => a + b, 0) / allY.length;
    }

    graph._stageXRanges = stageXRanges;
    graph._centerY = centerY;
    graph._maxTotalH = maxTotalH;
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

    group.append('text')
        .attr('class', 'op-label')
        .attr('x', op._x).attr('y', op._y + 3)
        .text(opSymbol(op.type));

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
    };
    return colors[type] || '#95a5a6';
}

function opSymbol(type) {
    const symbols = {
        matmul: '×', mask: '▽', softmax: 'σ',
        broadcast: '⇒', reshape: '↺',
        compress: '↓', decompress: '↑',
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
            drawRoutedArrow(g, sx, sy, op._x - OP_RADIUS - ARROWHEAD_LEN, op._y + yOff, graph, [t.id]);
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

function drawRoutedArrow(g, x1, y1, x2, y2, graph, excludeIds) {
    const COLLISION_MARGIN = 8;

    if (graph._tensorRects) {
        const colliders = graph._tensorRects.filter(r => {
            if (excludeIds && excludeIds.includes(r.id)) return false;
            return lineHitsRect(x1, y1, x2, y2, r, COLLISION_MARGIN);
        });

        if (colliders.length > 0) {
            const topBound = Math.min(...colliders.map(r => r.top)) - 20;
            const bottomBound = Math.max(...colliders.map(r => r.bottom)) + 20;
            // Route toward whichever side the target is closer to
            const routeY = y2 <= (topBound + bottomBound) / 2 ? topBound : bottomBound;

            const dx = x2 - x1;
            const bend = Math.min(dx * 0.2, 30);
            g.append('path')
                .attr('class', 'arrow-path')
                .attr('d', `M${x1},${y1} C${x1 + bend},${y1} ${x1 + bend},${routeY} ${(x1+x2)/2},${routeY} S${x2 - bend},${y2} ${x2},${y2}`)
                .attr('marker-end', 'url(#arrowhead)');
            return;
        }
    }

    // Standard bezier curve — no collisions
    const dx = x2 - x1;
    const cpx = dx * 0.4;
    g.append('path')
        .attr('class', 'arrow-path')
        .attr('d', `M${x1},${y1} C${x1 + cpx},${y1} ${x2 - cpx},${y2} ${x2},${y2}`)
        .attr('marker-end', 'url(#arrowhead)');
}

// --- Full render ---

export function renderGraph(g, graph, _params, onOpClick, onTensorClick) {
    g.selectAll('*').remove();

    computeLayout(graph);

    // Draw arrows first (behind everything)
    drawArrows(g, graph);

    // Draw tensors
    for (const t of graph.tensors) {
        if (t._layoutX == null) continue;
        const dimNames = t.dimNames || [];
        const block = drawTensorBlock(g, t._layoutX, t._layoutY, t, dimNames);

        block.on('click', (event) => {
            event.stopPropagation();
            g.selectAll('.tensor-block').classed('selected', false);
            block.classed('selected', true);
            if (onTensorClick) onTensorClick(t);
        });

        block.on('mouseenter', () => {
            const tooltip = d3.select('#tooltip');
            const shapeStr = `[${t.shape.join(', ')}]`;
            const dimStr = t.dimNames ? t.dimNames.map((n, i) => `${n}=${t.shape[i]}`).join(', ') : '';
            tooltip.select('.tt-label').text(t.label);
            tooltip.select('.tt-shape').text(`Shape: ${shapeStr}`);
            tooltip.select('.tt-dims').text(dimStr);
            tooltip.select('.tt-desc').text(t.desc || '');
            tooltip.classed('visible', true);
        });

        block.on('mousemove', (event) => {
            d3.select('#tooltip')
                .style('left', (event.clientX + 12) + 'px')
                .style('top', (event.clientY - 10) + 'px');
        });

        block.on('mouseleave', () => {
            d3.select('#tooltip').classed('visible', false);
        });
    }

    // Draw ops
    for (const op of graph.ops) {
        if (op._x == null) continue;
        drawOpNode(g, op, onOpClick);
    }
}
