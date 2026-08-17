// recurrent.js — First-class views for the operations unique to recurrent mixers.
import { detailMetrics } from './shared.js';

const BLUE = '#4a90d9';
const TEAL = '#16a085';
const ORANGE = '#e67e22';
const YELLOW = '#f1c40f';
const PURPLE = '#9b59b6';
const RED = '#e74c3c';

function title(g, x, y, text) {
    g.append('text').attr('class', 'tensor-label')
        .attr('x', x).attr('y', y).attr('font-size', '14px').text(text);
}

function label(g, x, y, text, options = {}) {
    return g.append('text')
        .attr('x', x).attr('y', y)
        .attr('text-anchor', options.anchor || 'middle')
        .attr('font-size', `${options.size || 10}px`)
        .attr('font-weight', options.weight || null)
        .attr('font-style', options.italic ? 'italic' : null)
        .attr('fill', options.color || '#aaa')
        .text(text);
}

function block(g, x, y, w, h, color, text, options = {}) {
    g.append('rect')
        .attr('x', x).attr('y', y).attr('width', w).attr('height', h)
        .attr('rx', options.rx ?? 4)
        .attr('fill', color).attr('fill-opacity', options.opacity ?? 0.62)
        .attr('stroke', color).attr('stroke-width', options.strokeWidth ?? 1);
    if (text) {
        label(g, x + w / 2, y + h / 2 + 4, text, {
            size: options.fontSize || 11,
            color: options.textColor || '#fff',
            weight: options.weight,
        });
    }
}

function arrow(g, x1, y1, x2, y2, color = '#777') {
    g.append('line')
        .attr('x1', x1).attr('y1', y1).attr('x2', x2 - 7).attr('y2', y2)
        .attr('stroke', color).attr('stroke-width', 1.5);
    g.append('path')
        .attr('d', `M ${x2 - 7} ${y2 - 4} L ${x2} ${y2} L ${x2 - 7} ${y2 + 4}`)
        .attr('fill', 'none').attr('stroke', color).attr('stroke-width', 1.5);
}

function divider(g, y, width) {
    g.append('line').attr('x1', 0).attr('y1', y).attr('x2', width).attr('y2', y)
        .attr('stroke', '#2a2d3a').attr('stroke-width', 1);
}

function wrap(text, maxChars = 78) {
    const lines = [];
    let line = '';
    for (const word of text.split(' ')) {
        if (line && line.length + word.length + 1 > maxChars) {
            lines.push(line);
            line = word;
        } else {
            line = line ? `${line} ${word}` : word;
        }
    }
    if (line) lines.push(line);
    return lines;
}

function note(g, x, y, text, color = '#808694') {
    const lines = wrap(text);
    lines.forEach((line, i) => label(g, x, y + i * 13, line, {
        anchor: 'start', size: 9, color, italic: true,
    }));
    return y + lines.length * 13;
}

function sequenceCells(width) {
    if (width <= 8) {
        return Array.from({ length: width }, (_, i) => ({
            lag: width - 1 - i,
            weight: i,
        }));
    }
    return [
        { lag: width - 1, weight: 0 },
        { lag: width - 2, weight: 1 },
        { lag: width - 3, weight: 2 },
        null,
        { lag: 2, weight: width - 3 },
        { lag: 1, weight: width - 2 },
        { lag: 0, weight: width - 1 },
    ];
}

function tokenLabel(lag) {
    if (lag === 0) return 'xₜ';
    if (lag === 1) return 'xₜ₋₁';
    return `xₜ₋${lag}`;
}

export function drawShortConvDetail(svg, op, tensorMap) {
    const { w } = detailMetrics();
    const innerW = w - 36;
    const g = svg.append('g').attr('transform', 'translate(18, 24)');
    const width = Math.max(1, op.convWidth || 1);
    const input = tensorMap[op.inputs[0]];
    const output = tensorMap[op.output];
    const state = op.inputs.map(id => tensorMap[id]).find(t => t?.state) || tensorMap.conv_state;
    const channels = input?.shape?.[input.shape.length - 1] || '?';
    const cells = sequenceCells(width);

    title(g, innerW / 2, 0, `Depthwise causal convolution · K=${width}`);
    label(g, innerW / 2, 18, 'one output position, one channel c', { size: 9, color: '#777' });

    const gap = 6;
    const cellW = Math.min(46, (innerW - gap * (cells.length - 1)) / cells.length);
    const rowW = cells.length * cellW + (cells.length - 1) * gap;
    const startX = (innerW - rowW) / 2;
    const inputY = 48;

    if (width > 1) {
        const historyW = rowW - cellW - gap;
        g.append('rect').attr('x', startX - 5).attr('y', inputY - 19)
            .attr('width', historyW + 10).attr('height', 54).attr('rx', 5)
            .attr('fill', TEAL).attr('fill-opacity', 0.08)
            .attr('stroke', TEAL).attr('stroke-opacity', 0.45)
            .attr('stroke-dasharray', '3,3');
        label(g, startX, inputY - 25, 'past only · rolling state', {
            anchor: 'start', size: 8, color: TEAL,
        });
    }

    cells.forEach((cell, i) => {
        const x = startX + i * (cellW + gap);
        if (!cell) {
            label(g, x + cellW / 2, inputY + 19, '…', { size: 18, color: '#666' });
            label(g, x + cellW / 2, inputY + 64, '…', { size: 18, color: '#666' });
            return;
        }
        const current = cell.lag === 0;
        block(g, x, inputY, cellW, 32, current ? BLUE : TEAL, tokenLabel(cell.lag), {
            opacity: current ? 0.78 : 0.5,
        });
        label(g, x + cellW / 2, inputY + 46, '×', { size: 13, color: '#777' });
        block(g, x, inputY + 54, cellW, 25, PURPLE, `w${cell.weight}`, {
            opacity: 0.48, fontSize: 9,
        });
    });

    const sumY = 151;
    cells.forEach((cell, i) => {
        if (!cell) return;
        const x = startX + i * (cellW + gap) + cellW / 2;
        g.append('line').attr('x1', x).attr('y1', inputY + 81)
            .attr('x2', innerW / 2).attr('y2', sumY - 12)
            .attr('stroke', '#4a4e5d').attr('stroke-width', 1);
    });
    block(g, innerW / 2 - 42, sumY - 12, 84, 28, ORANGE, 'sum', { opacity: 0.48 });
    arrow(g, innerW / 2 + 42, sumY + 2, innerW / 2 + 80, sumY + 2);
    block(g, innerW / 2 + 80, sumY - 12, 74, 28, YELLOW, 'SiLU', { opacity: 0.5 });

    label(g, innerW / 2, 196,
        `x̃ₜ,c = SiLU( Σⱼ₌₀…${width - 1} w_c,j · xₜ₋${width - 1}₊ⱼ,c )`,
        { size: 11, color: '#d5d8e2' });
    label(g, innerW / 2, 214,
        `depthwise: ${channels} channels use separate filters; channels never mix here`,
        { size: 9, color: '#777' });

    divider(g, 232, innerW);
    title(g, innerW / 2, 255, 'Prefill and decode are the same causal operator');
    block(g, 10, 274, 92, 34, BLUE, 'Prefill', { opacity: 0.42 });
    label(g, 116, 288, '→', { size: 15, color: '#777' });
    label(g, 130, 284, 'all new positions', { anchor: 'start', size: 9 });
    label(g, 130, 299, 'in one causal kernel', { anchor: 'start', size: 9, color: '#777' });

    block(g, 252, 274, 92, 34, TEAL, 'Decode', { opacity: 0.42 });
    label(g, 358, 288, '→', { size: 15, color: '#777' });
    label(g, 372, 284, 'read history', { anchor: 'start', size: 9 });
    label(g, 372, 299, 'then shift in xₜ', { anchor: 'start', size: 9, color: '#777' });

    const slots = state?.shape?.[state.shape.length - 1];
    let y = 326;
    y = note(g, 0, y,
        `No future token enters the window. The retained convolution state is fixed-size${slots ? ` (${slots} slots per channel in this graph)` : ''}, so it does not grow with context.`);
    svg.attr('height', y + 24);
}

function stateTile(g, x, y, w, h, color, text) {
    block(g, x, y, w, h, color, text, { opacity: 0.38 });
    for (let i = 1; i < 4; i++) {
        g.append('line').attr('x1', x + i * w / 4).attr('y1', y)
            .attr('x2', x + i * w / 4).attr('y2', y + h)
            .attr('stroke', color).attr('stroke-opacity', 0.4);
    }
    for (let i = 1; i < 3; i++) {
        g.append('line').attr('x1', x).attr('y1', y + i * h / 3)
            .attr('x2', x + w).attr('y2', y + i * h / 3)
            .attr('stroke', color).attr('stroke-opacity', 0.4);
    }
}

export function drawStateUpdateDetail(svg, op, tensorMap, params) {
    const { w } = detailMetrics();
    const innerW = w - 36;
    const g = svg.append('g').attr('transform', 'translate(18, 24)');
    const gatedDelta = op.type === 'gated_delta_update';
    const state = op.inputs.map(id => tensorMap[id]).find(t => t?.state);
    const next = tensorMap[op.output];
    const tokenCount = params?.queryLens?.reduce((a, b) => a + b, 0) || params?.S_q || 1;

    title(g, innerW / 2, 0, gatedDelta ? 'Gated delta state transition' : 'Selective SSM state transition');
    label(g, innerW / 2, 18, 'one token updates a fixed-size state', { size: 9, color: '#777' });

    const y = 48;
    stateTile(g, 8, y, 92, 70, TEAL, state?.label || (gatedDelta ? 'Sₜ₋₁' : 'hₜ₋₁'));
    arrow(g, 100, y + 35, 136, y + 35);

    if (gatedDelta) {
        block(g, 136, y + 1, 74, 30, YELLOW, 'αₜ decay', { opacity: 0.46, fontSize: 9 });
        block(g, 136, y + 40, 74, 30, RED, 'old S·k', { opacity: 0.4, fontSize: 9 });
        label(g, 229, y + 39, '+', { size: 17, color: '#aaa' });
        block(g, 248, y + 12, 96, 46, ORANGE, 'βₜ(vₜ−S·k)kᵀ', { opacity: 0.44, fontSize: 9 });
        arrow(g, 344, y + 35, 366, y + 35);
    } else {
        block(g, 136, y + 1, 74, 30, YELLOW, 'Āₜ decay', { opacity: 0.46, fontSize: 9 });
        block(g, 136, y + 40, 74, 30, BLUE, 'B̄ₜ · xₜ', { opacity: 0.46, fontSize: 9 });
        label(g, 229, y + 39, '+', { size: 17, color: '#aaa' });
        block(g, 248, y + 12, 96, 46, PURPLE, 'token-selective write', { opacity: 0.4, fontSize: 9 });
        arrow(g, 344, y + 35, 366, y + 35);
    }
    stateTile(g, 366, y, 70, 70, TEAL, next?.label || (gatedDelta ? 'Sₜ' : 'hₜ'));

    const formula = gatedDelta
        ? 'Sₜ = αₜ Sₜ₋₁ + βₜ (vₜ − Sₜ₋₁ k̂ₜ) k̂ₜᵀ'
        : 'hₜ = exp(Δₜ A) ⊙ hₜ₋₁ + B̄ₜ ⊙ xₜ';
    label(g, innerW / 2, 145, formula, { size: 11, color: '#d5d8e2' });
    label(g, innerW / 2, 164,
        gatedDelta ? 'α forgets globally; β replaces the association addressed by k̂ₜ'
            : 'Δₜ controls per-channel memory timescale; Bₜ controls what is written',
        { size: 9, color: '#777' });

    divider(g, 184, innerW);
    title(g, innerW / 2, 208, tokenCount > 1 ? `Scan across ${tokenCount.toLocaleString()} new tokens` : 'One recurrent decode step');
    const timelineY = 230;
    const xs = [32, 135, 238, 341];
    const names = ['state in', 'after t₀', 'after t₁', 'state out'];
    xs.forEach((x, i) => {
        block(g, x, timelineY, 70, 36, TEAL, names[i], { opacity: 0.34, fontSize: 8 });
        if (i < xs.length - 1) {
            arrow(g, x + 70, timelineY + 18, xs[i + 1], timelineY + 18);
            label(g, x + 86, timelineY + 10, i === 0 ? 'token 0' : i === 1 ? 'token 1' : '…', {
                size: 8, color: YELLOW,
            });
        }
    });
    label(g, innerW / 2, 286,
        tokenCount > 1 ? 'Prefill parallelizes this recurrence as a hardware-aware scan.'
            : 'Decode reads one retained state and writes its successor.',
        { size: 9, color: '#aaa' });

    const stateShape = state?.shape?.slice(1).join(' × ');
    let endY = 311;
    endY = note(g, 0, endY,
        `Only the final state is retained${stateShape ? ` (${stateShape} elements per request before dtype sizing)` : ''}. Intermediate token states live inside the scan; there is no S_q × S attention matrix.`);
    svg.attr('height', endY + 24);
}

export function drawStateReadDetail(svg, op, tensorMap) {
    const { w } = detailMetrics();
    const innerW = w - 36;
    const g = svg.append('g').attr('transform', 'translate(18, 24)');
    const state = op.inputs.map(id => tensorMap[id]).find(t => t?.state);
    const output = tensorMap[op.output];
    const matrixMemory = state?.shape?.length === 4;

    title(g, innerW / 2, 0, matrixMemory ? 'Read the associative matrix state' : 'Read the selective SSM state');
    label(g, innerW / 2, 18, 'the state is observed, not modified by this operation', { size: 9, color: '#777' });

    const y = 52;
    stateTile(g, 18, y, 120, 86, TEAL, state?.label || 'stateₜ');
    label(g, 78, y + 103, state?.shape ? `[${state.shape.slice(1).join(' × ')}] / request` : 'fixed-size state', {
        size: 8, color: '#777',
    });

    if (matrixMemory) {
        block(g, 177, y + 19, 66, 48, RED, 'q̂ₜ', { opacity: 0.5 });
        label(g, 157, y + 48, '×', { size: 17, color: '#aaa' });
        arrow(g, 243, y + 43, 286, y + 43);
        block(g, 286, y + 17, 130, 52, ORANGE, output?.label || 'yₜ = Sₜ q̂ₜ', { opacity: 0.55 });
        label(g, innerW / 2, 183, 'yₜ = Sₜ q̂ₜ', { size: 12, color: '#d5d8e2' });
        label(g, innerW / 2, 203, 'q̂ addresses a learned association; no token positions are scored', {
            size: 9, color: '#777',
        });
    } else {
        block(g, 168, y + 3, 74, 34, RED, 'Cₜ read', { opacity: 0.46, fontSize: 9 });
        block(g, 168, y + 50, 74, 34, PURPLE, 'D ⊙ xₜ', { opacity: 0.42, fontSize: 9 });
        label(g, 259, y + 48, '+', { size: 17, color: '#aaa' });
        arrow(g, 276, y + 43, 306, y + 43);
        block(g, 306, y + 17, 110, 52, ORANGE, output?.label || 'yₜ', { opacity: 0.55 });
        label(g, innerW / 2, 183, 'yₜ,c = Σₙ Cₜ,n · hₜ,c,n + D_c · xₜ,c', {
            size: 11, color: '#d5d8e2',
        });
        label(g, innerW / 2, 203, 'Cₜ selects memory content; D carries a direct local path', {
            size: 9, color: '#777',
        });
    }

    divider(g, 224, innerW);
    title(g, innerW / 2, 248, 'A read is produced at every new token');
    const xs = [28, 140, 252, 364];
    xs.forEach((x, i) => {
        block(g, x, 270, 62, 32, i === xs.length - 1 ? ORANGE : TEAL,
            i === xs.length - 1 ? 'yₜ' : i === 0 ? 'stateₜ₋₂' : i === 1 ? 'stateₜ₋₁' : 'stateₜ',
            { opacity: 0.36, fontSize: 8 });
        if (i < xs.length - 1) arrow(g, x + 62, 286, xs[i + 1], 286);
    });
    let endY = 328;
    endY = note(g, 0, endY,
        'During prefill, the scan produces the evolving state needed for each read. The diagram retains only the final state for the next request step.');
    svg.attr('height', endY + 24);
}

export function drawGateDetail(svg, op, tensorMap) {
    const { w } = detailMetrics();
    const innerW = w - 36;
    const g = svg.append('g').attr('transform', 'translate(18, 24)');
    const inputs = op.inputs.map(id => tensorMap[id]).filter(Boolean);
    const output = tensorMap[op.output];
    const normalized = op.label.toLowerCase().includes('norm');
    const value = inputs[0];
    const gate = inputs[1];

    title(g, innerW / 2, 0, normalized ? 'Normalize, then apply a SiLU gate' : 'Elementwise SiLU gate');
    label(g, innerW / 2, 18, 'one independent gate value per output element', { size: 9, color: '#777' });

    const y = 58;
    block(g, 8, y, 92, 54, ORANGE, value?.label || 'y', { opacity: 0.52 });
    if (normalized) {
        arrow(g, 100, y + 27, 128, y + 27);
        block(g, 128, y + 8, 70, 38, PURPLE, 'RMSNorm', { opacity: 0.42, fontSize: 9 });
    }
    const mulX = normalized ? 229 : 164;
    arrow(g, normalized ? 198 : 100, y + 27, mulX - 12, y + 27);
    label(g, mulX, y + 34, '⊙', { size: 21, color: YELLOW });
    block(g, mulX - 44, y + 73, 88, 40, YELLOW, `SiLU(${gate?.label || 'g'})`, {
        opacity: 0.46, fontSize: 9,
    });
    arrow(g, mulX + 14, y + 27, 300, y + 27);
    block(g, 300, y, 130, 54, ORANGE, output?.label || 'gated output', { opacity: 0.6, fontSize: 9 });
    g.append('line').attr('x1', mulX).attr('y1', y + 73)
        .attr('x2', mulX).attr('y2', y + 46)
        .attr('stroke', YELLOW).attr('stroke-width', 1.5);

    label(g, innerW / 2, 206,
        normalized ? 'out = RMSNorm(y) ⊙ SiLU(g)' : 'out = y ⊙ SiLU(z)',
        { size: 12, color: '#d5d8e2' });
    label(g, innerW / 2, 227, 'SiLU(g) = g · sigmoid(g)', { size: 10, color: YELLOW });

    divider(g, 249, innerW);
    title(g, innerW / 2, 273, 'What the gate changes');
    const samples = [
        { x: -3, s: -0.14 }, { x: -1, s: -0.27 }, { x: 0, s: 0 },
        { x: 1, s: 0.73 }, { x: 3, s: 2.86 },
    ];
    const baseX = 44, chartW = innerW - 88, chartY = 320;
    g.append('line').attr('x1', baseX).attr('y1', chartY).attr('x2', baseX + chartW).attr('y2', chartY)
        .attr('stroke', '#555');
    samples.forEach((sample, i) => {
        const x = baseX + i * chartW / (samples.length - 1);
        const barH = Math.abs(sample.s) * 13;
        g.append('rect').attr('x', x - 13)
            .attr('y', sample.s >= 0 ? chartY - barH : chartY)
            .attr('width', 26).attr('height', Math.max(1, barH))
            .attr('fill', sample.s >= 0 ? YELLOW : RED).attr('fill-opacity', 0.62);
        label(g, x, chartY + 15, `${sample.x}`, { size: 8, color: '#777' });
    });
    label(g, innerW / 2, 361, 'negative gates are softly suppressed; positive gates pass increasingly strongly', {
        size: 9, color: '#808694', italic: true,
    });
    svg.attr('height', 390);
}

export function drawNormalizeDetail(svg, op, tensorMap) {
    const { w } = detailMetrics();
    const innerW = w - 36;
    const g = svg.append('g').attr('transform', 'translate(18, 24)');
    const input = tensorMap[op.inputs[0]];
    const output = tensorMap[op.output];
    const dim = input?.shape?.[input.shape.length - 1] || '?';

    title(g, innerW / 2, 0, 'L2-normalize each head vector');
    label(g, innerW / 2, 18, 'direction is retained; magnitude is removed', { size: 9, color: '#777' });
    block(g, 22, 62, 116, 62, input?.color || BLUE, input?.label || 'x', { opacity: 0.5 });
    arrow(g, 138, 93, 182, 93);
    block(g, 182, 62, 90, 62, PURPLE, '÷ ‖x‖₂', { opacity: 0.46 });
    arrow(g, 272, 93, 316, 93);
    block(g, 316, 62, 116, 62, output?.color || BLUE, output?.label || 'x̂', { opacity: 0.62 });

    label(g, innerW / 2, 169, 'x̂ = x / √(Σᵢ xᵢ² + ε)', { size: 12, color: '#d5d8e2' });
    label(g, innerW / 2, 190, `normalization spans d=${dim} features independently for each token and head`, {
        size: 9, color: '#777',
    });
    divider(g, 215, innerW);
    title(g, innerW / 2, 242, 'Why Gated ΔNet normalizes q and k');
    let endY = 270;
    endY = note(g, 0, endY,
        'The state update and read depend on vector direction, not an arbitrary projection magnitude. Unit-length keys make the delta replacement well-scaled; unit-length queries make retrieval stable.');
    svg.attr('height', endY + 24);
}
