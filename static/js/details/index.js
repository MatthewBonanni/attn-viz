// index.js — Detail panel entry point: routing and state management
import { drawMatmulDetail } from './matmul.js';
import { drawMaskDetail, drawMaskTensorDetail, drawPagedMaskTensorDetail } from './mask.js';
import { drawPagedMaskDetail } from './paged-mask.js';
import { drawBroadcastDetail } from './broadcast.js';
import { drawGenericDetail } from './generic.js';
import { drawPagedCacheDetail } from './cache.js';
import { drawTensorShapeDetail } from './tensor-shape.js';
import { drawRopeDetail } from './rope.js';
import { computeOpCost, tensorElements, tensorBytes, fmtNum, fmtBytes, computeRooflineThreshold } from '../costs.js';

// Track currently displayed detail for live refresh
let _currentDetail = null;  // { type: 'op'|'tensor'|'group', id, graphId }

export function showDetail(op, graph, params) {
    _currentDetail = { type: 'op', id: op.id, graphId: graph.id };
    _renderOpDetail(op, graph, params);
}

function _shiftStatsOverlay(visible) {
    d3.select('#stats-overlay').style('right', visible ? '536px' : '16px');
}

function _renderOpDetail(op, graph, params) {
    const panel = d3.select('#detail-panel');
    panel.classed('visible', true);
    _shiftStatsOverlay(true);
    d3.select('#detail-title').text(op.label);

    const tensorMap = {};
    for (const t of graph.tensors) tensorMap[t.id] = t;

    // Build description + cost HTML
    let descHtml = op.desc || '';
    const cost = computeOpCost(op, tensorMap);
    if (cost && (cost.flops > 0 || cost.readBytes > 0)) {
        const threshold = computeRooflineThreshold('H100 SXM');
        const totalBytes = cost.readBytes + cost.writeBytes;
        const ai = cost.arithmeticIntensity;
        const regime = cost.flops === 0 ? 'MEMORY-ONLY'
            : ai >= threshold ? 'COMPUTE-BOUND' : 'MEMORY-BOUND';
        const regimeColor = regime === 'COMPUTE-BOUND' ? '#2ecc71'
            : regime === 'MEMORY-BOUND' ? '#e74c3c' : '#888';

        descHtml += `<div style="margin-top:12px;padding:10px 12px;background:#1a1d2a;border-radius:6px;border:1px solid #2a2d3a;font-size:11px;line-height:1.8">`;
        descHtml += `<div style="font-weight:600;color:#bbb;margin-bottom:4px;font-size:12px">Cost Analysis</div>`;

        if (cost.flops > 0) {
            descHtml += `<div><span style="color:#888">FLOPs:</span> <span style="color:#7c8cf8;font-weight:600">${fmtNum(cost.flops)}</span></div>`;
        }

        // Memory breakdown
        for (const item of cost.breakdown) {
            if (item.bytes > 0) {
                const shapeStr = item.shape ? ` [${item.shape.join('×')}]` : '';
                descHtml += `<div><span style="color:#888">${item.label}:</span> <span style="color:#aaa">${fmtBytes(item.bytes)}</span><span style="color:#555;font-size:10px">${shapeStr} × 2B</span></div>`;
            }
        }

        descHtml += `<div style="margin-top:4px;border-top:1px solid #2a2d3a;padding-top:4px">`;
        descHtml += `<span style="color:#888">Total memory:</span> <span style="color:#aaa">${fmtBytes(totalBytes)}</span>`;
        if (cost.flops > 0) {
            descHtml += ` &nbsp;|&nbsp; <span style="color:#888">Arithmetic intensity:</span> <span style="color:#7c8cf8;font-weight:600">${ai.toFixed(1)}</span> <span style="color:#555">FLOPs/byte</span>`;
        }
        descHtml += `</div>`;
        descHtml += `<div style="margin-top:2px"><span style="color:${regimeColor};font-weight:600">${regime}</span> <span style="color:#555;font-size:10px">(threshold: ${threshold.toFixed(0)} FLOPs/byte on H100)</span></div>`;
        descHtml += `</div>`;
    }
    d3.select('#detail-desc').html(descHtml);

    const svg = d3.select('#detail-svg');
    svg.selectAll('*').remove();
    svg.attr('height', 350);

    switch (op.type) {
        case 'matmul':
        case 'compress':
        case 'decompress':
            drawMatmulDetail(svg, op, tensorMap);
            break;
        case 'mask':
            if (params.pagedAttn) {
                drawPagedMaskDetail(svg, op, tensorMap, params);
            } else {
                drawMaskDetail(svg, op, tensorMap, params);
            }
            break;
        case 'broadcast':
        case 'reshape':
            drawBroadcastDetail(svg, op, tensorMap);
            break;
        case 'rope':
            drawRopeDetail(svg, op, tensorMap);
            break;
        default:
            drawGenericDetail(svg, op, tensorMap);
    }
}

export function showTensorDetail(tensor, params) {
    _currentDetail = { type: 'tensor', id: tensor.id };
    _renderTensorDetail(tensor, params);
}

function _renderTensorDetail(tensor, params) {
    const panel = d3.select('#detail-panel');
    panel.classed('visible', true);
    _shiftStatsOverlay(true);
    d3.select('#detail-title').text(tensor.label);

    let descHtml = tensor.desc || '';
    const elems = tensorElements(tensor.shape);
    const bytes = tensorBytes(tensor.shape);
    const shapeStr = tensor.shape.join(' × ');
    descHtml += `<div style="margin-top:12px;padding:10px 12px;background:#1a1d2a;border-radius:6px;border:1px solid #2a2d3a;font-size:11px;line-height:1.8">`;
    descHtml += `<div style="font-weight:600;color:#bbb;margin-bottom:4px;font-size:12px">Size</div>`;
    descHtml += `<div><span style="color:#888">Shape:</span> <span style="color:#7c8cf8">[${shapeStr}]</span></div>`;
    descHtml += `<div><span style="color:#888">Elements:</span> <span style="color:#aaa">${elems.toLocaleString()}</span> <span style="color:#555;font-size:10px">(${fmtNum(elems)})</span></div>`;
    descHtml += `<div><span style="color:#888">Size (bf16):</span> <span style="color:#aaa">${fmtBytes(bytes)}</span></div>`;
    descHtml += `</div>`;
    d3.select('#detail-desc').html(descHtml);

    const svg = d3.select('#detail-svg');
    svg.selectAll('*').remove();

    if (tensor.badge === 'PAGED' && params.pagedAttn) {
        svg.attr('height', 100);
        drawPagedCacheDetail(svg, tensor, params);
    } else if (tensor.type === 'mask' && tensor.pagedMask && params.pagedAttn) {
        svg.attr('height', 350);
        drawPagedMaskTensorDetail(svg, tensor, params);
    } else if (tensor.type === 'mask') {
        svg.attr('height', 350);
        drawMaskTensorDetail(svg, tensor, params);
    } else {
        drawTensorShapeDetail(svg, tensor, params);
    }
}

export function showGroupDetail(group) {
    _currentDetail = { type: 'group', id: group.label };
    _renderGroupDetail(group);
}

function _renderGroupDetail(group) {
    const panel = d3.select('#detail-panel');
    panel.classed('visible', true);
    _shiftStatsOverlay(true);
    d3.select('#detail-title').text(group.label);
    d3.select('#detail-desc').html(group.desc || '');

    const svg = d3.select('#detail-svg');
    svg.selectAll('*').remove();
    svg.attr('height', 0);
}

export function refreshDetail(graphs, params) {
    if (!_currentDetail) return;
    const panel = d3.select('#detail-panel');
    if (!panel.classed('visible')) return;

    // Find the matching op, tensor, or group in the fresh graphs
    for (const graph of graphs) {
        if (_currentDetail.type === 'op') {
            const op = graph.ops.find(o => o.id === _currentDetail.id);
            if (op && (!_currentDetail.graphId || graph.id === _currentDetail.graphId)) {
                _renderOpDetail(op, graph, params);
                return;
            }
        } else if (_currentDetail.type === 'group') {
            const group = (graph.groups || []).find(gr => gr.label === _currentDetail.id);
            if (group) {
                _renderGroupDetail(group);
                return;
            }
        } else {
            const tensor = graph.tensors.find(t => t.id === _currentDetail.id);
            if (tensor) {
                _renderTensorDetail(tensor, params);
                return;
            }
        }
    }
}

export function hideDetail() {
    _currentDetail = null;
    d3.select('#detail-panel').classed('visible', false);
    _shiftStatsOverlay(false);
}
