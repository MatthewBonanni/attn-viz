// index.js — Detail panel entry point: routing and state management
import { drawMatmulDetail } from './matmul.js';
import { drawMaskDetail, drawMaskTensorDetail, drawPagedMaskTensorDetail } from './mask.js';
import { drawPagedMaskDetail } from './paged-mask.js';
import { drawBroadcastDetail } from './broadcast.js';
import { drawGenericDetail } from './generic.js';
import { drawPagedCacheDetail } from './cache.js';
import { drawTensorShapeDetail } from './tensor-shape.js';
import { drawRopeDetail } from './rope.js';

// Track currently displayed detail for live refresh
let _currentDetail = null;  // { type: 'op'|'tensor'|'group', id, graphId }

export function showDetail(op, graph, params) {
    _currentDetail = { type: 'op', id: op.id, graphId: graph.id };
    _renderOpDetail(op, graph, params);
}

function _renderOpDetail(op, graph, params) {
    const panel = d3.select('#detail-panel');
    panel.classed('visible', true);
    d3.select('#detail-title').text(op.label);
    d3.select('#detail-desc').html(op.desc || '');

    const svg = d3.select('#detail-svg');
    svg.selectAll('*').remove();
    svg.attr('height', 350);

    const tensorMap = {};
    for (const t of graph.tensors) tensorMap[t.id] = t;

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
    d3.select('#detail-title').text(tensor.label);
    d3.select('#detail-desc').html(tensor.desc || '');

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
}
