import test from 'node:test';
import assert from 'node:assert/strict';

import { dsv4CompressedLength, dsv4Graph } from '../static/js/graphs.js';
import { buildHash } from '../static/js/url-state.js';
import { computeDsv4AttentionPairs } from '../static/js/costs.js';

const params = {
    B: 1,
    S: 1024,
    S_q: 1,
    seqLens: [1024],
    queryLens: [1],
    d_model: 4096,
    n_h: 64,
    d_h: 512,
    d_r: 64,
    topk: 512,
    n_i: 64,
    d_i: 128,
};

test('DeepSeek-V4 compression counts only completed native-token groups', () => {
    const mixed = { ...params, B: 2, seqLens: [15, 257] };
    assert.equal(dsv4CompressedLength(mixed, 4), 3 + 64);
    assert.equal(dsv4CompressedLength(mixed, 128), 0 + 2);
});

test('DeepSeek-V4 hybrid attention counts local and causally complete memory', () => {
    const shortPrefill = { B: 1, S: 8, S_q: 8, topk: 512 };
    assert.equal(computeDsv4AttentionPairs(shortPrefill, 'swa'), 36);
    assert.equal(computeDsv4AttentionPairs(shortPrefill, 'c128'), 36);
    assert.equal(computeDsv4AttentionPairs(shortPrefill, 'c4'), 42);

    const decode = { B: 1, S: 256, S_q: 1, topk: 512 };
    assert.equal(computeDsv4AttentionPairs(decode, 'c4'), 128 + 64);
    assert.equal(computeDsv4AttentionPairs(decode, 'c128'), 128 + 2);
});

test('DeepSeek-V4 layer views preserve their distinct long-range contracts', () => {
    const c4 = dsv4Graph(params, 'c4');
    const c128 = dsv4Graph(params, 'c128');
    const swa = dsv4Graph(params, 'swa');

    assert.equal(c4.tensors.find(t => t.id === 'KV_compressed').shape[0], 256);
    assert.ok(c4.ops.some(op => op.id === 'topk'));
    assert.equal(c4.ops.find(op => op.id === 'cache_compressed').inputs[0], 'KV_compressed_new');
    assert.ok(
        c4.tensors.find(t => t.id === 'KV_selected').stage
        < c4.tensors.find(t => t.id === 'KV_attended').stage,
        'the selected KV input must render left of its concatenate output',
    );

    assert.equal(c128.tensors.find(t => t.id === 'KV_compressed').shape[0], 8);
    assert.ok(!c128.ops.some(op => op.id === 'topk'));

    assert.ok(!swa.tensors.some(t => t.id === 'KV_compressed'));
    assert.deepEqual(swa.ops.find(op => op.id === 'combine_kv').inputs, ['KV_local']);
});

test('shared links retain the selected DeepSeek-V4 layer', () => {
    const hash = buildHash({
        variant: 'dsv4',
        layer: 'c128',
        preset: 0,
        params,
        baseline: params,
    });
    assert.equal(hash, '#v=dsv4&l=c128');
});
