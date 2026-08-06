// url-state.js — Serialize the current view into location.hash so it can be shared.
//
// Encoding is preset-relative: the hash carries the variant, the preset index,
// and only the values that differ from that preset's baseline. Picking a preset
// gives a two-parameter URL; tweaking sliders adds only what you tweaked.

export const URL_KEYS = [
    'B', 'S', 'S_q', 'd_model', 'n_h', 'd_h', 'n_kv', 'd_c', 'd_q', 'd_r',
    'topk', 'n_i', 'd_i', 'tp_size', 'dp_size', 'block_size', 'window_size',
];

// short URL key → params key
export const URL_FLAGS = { flash: 'flashAttn', paged: 'pagedAttn', swa: 'slidingWindow' };

function uniform(arr, n) {
    if (!arr || arr.length < n) return null;
    const first = arr[0];
    return arr.slice(0, n).every(v => v === first) ? first : null;
}

export function readUrlState() {
    const raw = window.location.hash.replace(/^#/, '');
    if (!raw) return null;
    const q = new URLSearchParams(raw);
    if (![...q.keys()].length) return null;

    const state = { values: {}, flags: {} };
    if (q.has('v')) state.variant = q.get('v');
    if (q.has('l')) state.layer = q.get('l');
    if (q.has('p')) {
        const p = parseInt(q.get('p'), 10);
        if (Number.isFinite(p)) state.preset = p;
    }
    for (const key of URL_KEYS) {
        if (!q.has(key)) continue;
        const v = parseInt(q.get(key), 10);
        if (Number.isFinite(v)) state.values[key] = v;
    }
    for (const short of Object.keys(URL_FLAGS)) {
        if (q.has(short)) state.flags[URL_FLAGS[short]] = q.get(short) === '1';
    }
    const parseList = (s) => s.split(',').map(v => parseInt(v, 10)).filter(Number.isFinite);
    if (q.has('sl')) state.seqLens = parseList(q.get('sl'));
    if (q.has('ql')) state.queryLens = parseList(q.get('ql'));
    return state;
}

// Build the hash for the current view. `baseline` is the params object that
// loading this variant + preset from scratch would produce.
export function buildHash({ variant, layer, preset, params, baseline }) {
    const q = new URLSearchParams();
    q.set('v', variant);
    if (variant === 'dsv4' && layer && layer !== 'c4') q.set('l', layer);
    if (preset != null && preset > 0) q.set('p', String(preset));

    for (const key of URL_KEYS) {
        if (params[key] != null && params[key] !== baseline[key]) q.set(key, String(params[key]));
    }
    for (const [short, pkey] of Object.entries(URL_FLAGS)) {
        if (!!params[pkey] !== !!baseline[pkey]) q.set(short, params[pkey] ? '1' : '0');
    }

    // Per-request lengths can be omitted only when they are all equal AND agree
    // with S/S_q, which is what a link without them reconstructs from.
    const B = params.B || 1;
    if (B > 1) {
        const su = uniform(params.seqLens, B);
        if (su == null || su !== params.S) q.set('sl', params.seqLens.slice(0, B).join(','));
        const qu = uniform(params.queryLens, B);
        if (qu == null || qu !== params.S_q) q.set('ql', params.queryLens.slice(0, B).join(','));
    }
    return '#' + q.toString();
}

let pending = null;

// Debounced: update() fires on every slider frame, and history writes are not free
export function writeUrlState(state) {
    if (pending) clearTimeout(pending);
    pending = setTimeout(() => {
        pending = null;
        const hash = buildHash(state);
        if (hash !== window.location.hash) {
            history.replaceState(null, '', window.location.pathname + window.location.search + hash);
        }
    }, 250);
}
