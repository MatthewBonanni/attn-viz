import test from 'node:test';
import assert from 'node:assert/strict';

import { applyWorkloadPreset, detectWorkloadPreset } from '../static/js/workloads.js';

test('mixed workload preset creates one request of every execution type', () => {
    const params = { B: 1, S: 1024, S_q: 1024, seqLens: [1024], queryLens: [1024] };
    applyWorkloadPreset(params, 'mixed');

    assert.equal(params.B, 4);
    assert.deepEqual(params.seqLens, [1024, 768, 512, 256]);
    assert.deepEqual(params.queryLens, [1024, 256, 8, 1]);
    assert.equal(detectWorkloadPreset(params), 'mixed');
});

test('uniform workload presets preserve batch contexts and replace query lengths', () => {
    const params = {
        B: 2,
        S: 2048,
        S_q: 1,
        seqLens: [2048, 1024],
        queryLens: [1, 1],
    };

    applyWorkloadPreset(params, 'spec');
    assert.deepEqual(params.seqLens, [2048, 1024]);
    assert.deepEqual(params.queryLens, [8, 8]);
    assert.equal(detectWorkloadPreset(params), 'spec');

    applyWorkloadPreset(params, 'prefill');
    assert.deepEqual(params.queryLens, [2048, 1024]);
    assert.equal(detectWorkloadPreset(params), 'prefill');
});
