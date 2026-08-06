export const WORKLOAD_PRESETS = [
    { id: 'prefill', label: 'Prefill', title: 'All requests process their full context' },
    { id: 'extend', label: 'Extend', title: 'All requests append up to 256 tokens' },
    { id: 'spec', label: 'Spec', title: 'All requests process an 8-token speculative step' },
    { id: 'decode', label: 'Decode', title: 'All requests process one token' },
    { id: 'mixed', label: 'Mixed', title: 'Prefill, extend, speculative decode, and decode requests together' },
];

function requestType(s, q) {
    if (q === 1) return 'decode';
    if (q >= s) return 'prefill';
    if (q < 16) return 'spec';
    return 'extend';
}

function currentSeqLens(params) {
    const B = Math.max(1, params.B || 1);
    return Array.from({ length: B }, (_, i) =>
        Math.max(1, params.seqLens?.[i] || params.S || 1));
}

export function applyWorkloadPreset(params, preset) {
    if (preset === 'mixed') {
        const S = Math.max(1, params.S || Math.max(...currentSeqLens(params)));
        params.B = 4;
        params.seqLens = [S, Math.max(1, Math.floor(3 * S / 4)),
            Math.max(1, Math.floor(S / 2)), Math.max(1, Math.floor(S / 4))];
        params.queryLens = [
            params.seqLens[0],
            Math.min(256, params.seqLens[1]),
            Math.min(8, params.seqLens[2]),
            1,
        ];
    } else {
        const seqLens = currentSeqLens(params);
        params.seqLens = seqLens;
        params.queryLens = seqLens.map(s => {
            if (preset === 'prefill') return s;
            if (preset === 'extend') return Math.min(256, s);
            if (preset === 'spec') return Math.min(8, s);
            return 1;
        });
    }

    params.S = Math.max(...params.seqLens);
    params.S_q = Math.max(...params.queryLens);
}

export function detectWorkloadPreset(params) {
    const B = Math.max(1, params.B || 1);
    const seqLens = currentSeqLens(params);
    const queryLens = Array.from({ length: B }, (_, i) =>
        Math.max(1, Math.min(seqLens[i], params.queryLens?.[i] || params.S_q || 1)));
    const types = new Set(seqLens.map((s, i) => requestType(s, queryLens[i])));
    if (types.size === 1) return types.values().next().value;
    return B > 1 ? 'mixed' : null;
}
