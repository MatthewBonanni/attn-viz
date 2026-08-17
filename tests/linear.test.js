import test from 'node:test';
import assert from 'node:assert/strict';

import { computeOpCost } from '../static/js/costs.js';
import { OP_DETAIL_DRAWERS } from '../static/js/details/index.js';
import { gatedDeltaGraph, mambaGraph } from '../static/js/graphs.js';
import { opSymbol } from '../static/js/render.js';
import { buildHash } from '../static/js/url-state.js';

const mambaParams = {
    B: 1,
    S: 1024,
    S_q: 64,
    seqLens: [1024],
    queryLens: [64],
    d_model: 2560,
    d_state: 16,
    d_conv: 4,
    expand: 2,
};

const gatedDeltaParams = {
    B: 1,
    S: 1024,
    S_q: 64,
    seqLens: [1024],
    queryLens: [64],
    d_model: 2048,
    n_h: 16,
    d_h: 128,
    d_conv: 4,
    expand_v: 2,
};

test('linear mixers retain fixed-size state instead of token-addressed KV cache', () => {
    for (const [graphFn, params] of [
        [mambaGraph, mambaParams],
        [gatedDeltaGraph, gatedDeltaParams],
    ]) {
        const short = graphFn(params);
        const long = graphFn({
            ...params,
            S: 65536,
            seqLens: [65536],
        });
        const shortStates = short.tensors.filter(tensor => tensor.state);
        const longStates = long.tensors.filter(tensor => tensor.state);

        assert.deepEqual(
            shortStates.map(tensor => tensor.shape),
            longStates.map(tensor => tensor.shape),
        );
        assert.ok(shortStates.every(tensor => !tensor.cache));
        assert.ok(!short.tensors.some(tensor =>
            tensor.dimNames?.includes('S') && tensor.cache));
    }
});

test('Mamba and Gated DeltaNet recurrent work scales with new tokens, not context', () => {
    for (const [graphFn, params, updateId] of [
        [mambaGraph, mambaParams, 'selective_update'],
        [gatedDeltaGraph, gatedDeltaParams, 'delta_update'],
    ]) {
        const graph = graphFn(params);
        const tensorMap = Object.fromEntries(graph.tensors.map(tensor => [tensor.id, tensor]));
        const update = graph.ops.find(op => op.id === updateId);
        const one = computeOpCost(update, tensorMap, {
            ...params,
            S_q: 1,
            queryLens: [1],
            graphId: graph.id,
        });
        const eight = computeOpCost(update, tensorMap, {
            ...params,
            S_q: 8,
            queryLens: [8],
            graphId: graph.id,
        });

        assert.equal(eight.flops, 8 * one.flops);
    }
});

test('shared links retain linear-mixer state dimensions', () => {
    const baseline = { ...mambaParams };
    const params = { ...baseline, d_state: 64, d_conv: 8, expand: 3 };
    const hash = buildHash({
        variant: 'mamba',
        preset: 0,
        params,
        baseline,
    });

    assert.equal(hash, '#v=mamba&d_state=64&d_conv=8&expand=3');
});

test('every linear-mixer operation has an explicit icon and detail renderer', () => {
    // Regression: recurrent operations once silently fell through to a "?" node
    // and the four-line generic detail panel. Keep the two registries complete as
    // new linear-mixer operation types are introduced.
    const opTypes = new Set([
        ...mambaGraph(mambaParams).ops.map(op => op.type),
        ...gatedDeltaGraph(gatedDeltaParams).ops.map(op => op.type),
    ]);

    for (const type of opTypes) {
        assert.notEqual(opSymbol(type), '?', `${type} needs an explicit graph icon`);
        assert.ok(OP_DETAIL_DRAWERS[type], `${type} needs a dedicated detail renderer`);
    }
});
