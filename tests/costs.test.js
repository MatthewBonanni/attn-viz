import test from 'node:test';
import assert from 'node:assert/strict';

import {
    computeAttentionPairs,
    computeSparsePairs,
    computeOpCost,
    computePipelineTotals,
    computePipelineRoofline,
    estimateOpRoofline,
} from '../static/js/costs.js';
import { gqaGraph } from '../static/js/graphs.js';

function tensor(id, shape, dimNames, extra = {}) {
    return { id, label: id, shape, dimNames, ...extra };
}

function tensorMap(...tensors) {
    return Object.fromEntries(tensors.map(t => [t.id, t]));
}

test('batched attention excludes cross-request token pairs', () => {
    const params = {
        B: 2,
        queryLens: [1, 1],
        seqLens: [100, 100],
    };
    assert.equal(computeAttentionPairs(params), 200);

    const Q = tensor('Q', [2, 2, 4], ['n_h', 'ΣS_q', 'd_h']);
    const K = tensor('K', [2, 200, 4], ['n_h', 'ΣS', 'd_h']);
    const scores = tensor('scores', [2, 2, 200], ['n_h', 'ΣS_q', 'ΣS']);
    const op = { id: 'qkt', type: 'matmul', inputs: ['Q', 'K'], output: 'scores' };
    const cost = computeOpCost(op, tensorMap(Q, K, scores), { ...params, graphId: 'mha' });

    // 2 heads × 200 valid request-local pairs × 4-wide dot × 2 FLOPs/MAC.
    assert.equal(cost.flops, 3200);
    assert.equal(cost.writeBytes, 800);
});

test('sliding-window work counts the live causal band exactly', () => {
    const params = {
        B: 1,
        S: 8,
        S_q: 4,
        slidingWindow: true,
        window_size: 3,
    };
    assert.equal(computeAttentionPairs(params), 12);

    const Q = tensor('Q', [2, 4, 4], ['n_h', 'S_q', 'd_h']);
    const K = tensor('K', [2, 8, 4], ['n_h', 'S', 'd_h']);
    const scores = tensor('scores', [2, 4, 8], ['n_h', 'S_q', 'S']);
    const op = { id: 'qkt', type: 'matmul', inputs: ['Q', 'K'], output: 'scores' };
    const cost = computeOpCost(op, tensorMap(Q, K, scores), { ...params, graphId: 'mha' });

    assert.equal(cost.flops, 192);
    // The four trailing queries touch a six-token union when W=3.
    assert.equal(cost.breakdown.find(item => item.label === 'Read K').bytes, 96);
});

test('DSA sparse work respects causal availability before top-k fills', () => {
    assert.equal(computeSparsePairs({ B: 1, S: 4, S_q: 4, topk: 2 }), 7);
});

test('DP uses the busiest request partition and keeps weights replicated', () => {
    const X = tensor('X', [4, 8], ['ΣS_q', 'D']);
    const W = tensor('W', [8, 8], ['D', 'D'], { type: 'weight' });
    const Y = tensor('Y', [4, 8], ['ΣS_q', 'D']);
    const op = { id: 'linear', type: 'matmul', inputs: ['X', 'W'], output: 'Y' };
    const params = {
        B: 2,
        queryLens: [1, 3],
        seqLens: [10, 10],
        dp_size: 2,
        graphId: 'mha',
    };
    const cost = computeOpCost(op, tensorMap(X, W, Y), params);

    assert.equal(cost.flops, 384); // critical rank has three of four query tokens
    assert.equal(cost.breakdown.find(item => item.label === 'Read W').bytes, 128);
    assert.equal(cost.breakdown.find(item => item.label === 'Read X').bytes, 48);

    const graph = { id: 'mha', tensors: [X, W, Y], ops: [op] };
    const totals = computePipelineTotals(graph, params);
    const estimate = computePipelineRoofline(totals, 'H100 SXM');
    assert.equal(estimate.dpRank, 1);
});

test('TP reports local math plus ideal ring all-reduce traffic', () => {
    const ctx = tensor('ctx', [4, 8], ['S_q', 'D'], { tpSharded: true });
    const W = tensor('W_O', [8, 8], ['D', 'D'], { type: 'weight', tpSharded: true });
    const out = tensor('out', [4, 8], ['S_q', 'D']);
    const op = {
        id: 'out_proj', type: 'matmul', inputs: ['ctx', 'W_O'], output: 'out',
        tpAllReduce: true, tpSize: 2,
    };
    const cost = computeOpCost(op, tensorMap(ctx, W, out), {
        B: 1, S: 4, S_q: 4, tp_size: 2, graphId: 'mha',
    });

    assert.equal(cost.flops, 256);
    assert.equal(cost.communicationBytes, 64);
});

test('GQA broadcast tensors read compact KV storage', () => {
    const params = {
        B: 1, S: 100, S_q: 1,
        n_h: 8, n_kv: 2, d_h: 64,
        tp_size: 1, dp_size: 1,
    };
    const totals = computePipelineTotals(gqaGraph(params), params);
    const qkt = totals.opCosts.find(({ op }) => op.id === 'qkt').cost;
    const keyRead = qkt.breakdown.find(item => item.label === "Read K'");

    // Two stored KV heads, not eight logically broadcast query heads.
    assert.equal(keyRead.bytes, 2 * 100 * 64 * 2);
});

test('roofline timing uses operation dtype and sums per-op lower bounds', () => {
    const q = tensor('q', [1, 1, 16], ['n_i', 'S_q', 'd_i'], { bytesPerEl: 1 });
    const k = tensor('k', [1, 1, 16], ['n_i', 'S', 'd_i'], { bytesPerEl: 1 });
    const scores = tensor('scores', [1, 1, 1], ['n_i', 'S_q', 'S']);
    const fp8Cost = computeOpCost(
        { id: 'idx_qk', type: 'matmul', inputs: ['q', 'k'], output: 'scores' },
        tensorMap(q, k, scores),
        { B: 1, S: 1, S_q: 1, graphId: 'dsa' },
    );
    assert.equal(fp8Cost.computeDtype, 'fp8');
    assert.equal(estimateOpRoofline(fp8Cost, 'H100 SXM').effectiveDtype, 'fp8');
    assert.equal(estimateOpRoofline(fp8Cost, 'A100 SXM').effectiveDtype, 'bf16');

    const totals = {
        opCosts: [
            { cost: { flops: 312e12, readBytes: 0, writeBytes: 0, computeDtype: 'bf16' } },
            { cost: { flops: 0, readBytes: 2e12, writeBytes: 0, computeDtype: 'bf16' } },
        ],
    };
    const estimate = computePipelineRoofline(totals, 'A100 SXM');
    assert.equal(estimate.computeTime, 1);
    assert.equal(estimate.memoryTime, 1);
    assert.equal(estimate.lowerBoundTime, 2);
    assert.equal(estimate.bound, 'MIXED');
});
