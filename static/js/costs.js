// costs.js — Compute FLOPs and memory transfer costs for ops and tensors

const BYTES_BF16 = 2;

// Format large numbers with SI suffixes
export function fmtNum(n) {
    if (n === 0) return '0';
    const abs = Math.abs(n);
    if (abs >= 1e12) return (n / 1e12).toFixed(1) + 'T';
    if (abs >= 1e9) return (n / 1e9).toFixed(1) + 'G';
    if (abs >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (abs >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return n.toString();
}

// Format bytes with binary-ish suffixes
export function fmtBytes(b) {
    if (b === 0) return '0 B';
    if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB';
    if (b >= 1e6) return (b / 1e6).toFixed(2) + ' MB';
    if (b >= 1e3) return (b / 1e3).toFixed(1) + ' KB';
    return b + ' B';
}

export function tensorElements(shape) {
    return shape.reduce((a, b) => a * b, 1);
}

export function tensorBytes(shape, bytesPerEl = BYTES_BF16) {
    return tensorElements(shape) * bytesPerEl;
}

function normalizedWorkload(params = {}) {
    const B = Math.max(1, params.B || 1);
    let queryLens;
    let seqLens;
    if (B === 1) {
        const s = Math.max(1, params.S || params.sumS || 1);
        seqLens = [s];
        queryLens = [Math.max(1, Math.min(s, params.S_q || params.sumSq || s))];
    } else {
        seqLens = (params.seqLens || Array(B).fill(params.S || 1))
            .slice(0, B).map(v => Math.max(1, v));
        queryLens = (params.queryLens || Array(B).fill(params.S_q || params.S || 1))
            .slice(0, B).map((v, i) => Math.max(1, Math.min(seqLens[i], v)));
    }
    return {
        B,
        queryLens,
        seqLens,
        windowSize: params.slidingWindow ? Math.max(1, params.window_size || 1) : null,
        topk: Math.max(1, params.topk || 2048),
        dsv4Layer: params.dsv4Layer || String(params.graphId || '').replace(/^dsv4_/, '') || null,
        tpSize: Math.max(1, params.tp_size || 1),
        dpSize: Math.max(1, Math.min(params.dp_size || 1, B)),
        dpRank: Number.isInteger(params.dpRank) ? params.dpRank : null,
    };
}

function sum(values) {
    return values.reduce((a, b) => a + b, 0);
}

function queryPairCount(s, q, limit) {
    if (limit == null) return q * s;
    let pairs = 0;
    const firstQueryPos = s - q;
    for (let i = 0; i < q; i++) {
        pairs += Math.min(limit, firstQueryPos + i + 1);
    }
    return pairs;
}

export function computeAttentionPairs(params = {}) {
    const w = normalizedWorkload(params);
    return sum(w.seqLens.map((s, i) => queryPairCount(s, w.queryLens[i], w.windowSize)));
}

export function computeDensePairs(params = {}) {
    const w = normalizedWorkload(params);
    return sum(w.seqLens.map((s, i) => s * w.queryLens[i]));
}

export function computeSparsePairs(params = {}) {
    const w = normalizedWorkload(params);
    return sum(w.seqLens.map((s, i) => queryPairCount(s, w.queryLens[i], w.topk)));
}

function dsv4Ratio(layer) {
    return layer === 'c4' ? 4 : layer === 'c128' ? 128 : 1;
}

function queryPositionWork(s, q, fn) {
    const firstQueryPos = s - q;
    let total = 0;
    for (let i = 0; i < q; i++) total += fn(firstQueryPos + i);
    return total;
}

function dsv4PerRequestWork(kind, workload) {
    const layer = workload.dsv4Layer;
    const ratio = dsv4Ratio(layer);
    return workload.seqLens.map((s, i) => {
        const q = workload.queryLens[i];
        if (kind === 'compressedS') return layer === 'swa' ? 0 : Math.floor(s / ratio);
        if (kind === 'compressedNew') {
            return layer === 'swa' ? 0 : Math.floor(s / ratio) - Math.floor((s - q) / ratio);
        }
        if (kind === 'localS') return Math.min(s, 128);
        if (kind === 'compressedDense') {
            return queryPositionWork(s, q, pos => Math.floor((pos + 1) / ratio));
        }
        if (kind === 'compressedSparse') {
            return queryPositionWork(s, q, pos =>
                Math.min(workload.topk, Math.floor((pos + 1) / ratio)));
        }
        if (kind === 'hybridAttention') {
            return queryPositionWork(s, q, pos => {
                const local = Math.min(128, pos + 1);
                if (layer === 'swa') return local;
                const compressed = Math.floor((pos + 1) / ratio);
                return local + (layer === 'c4' ? Math.min(workload.topk, compressed) : compressed);
            });
        }
        return 0;
    });
}

export function computeDsv4AttentionPairs(params = {}, layer = 'c4') {
    const workload = normalizedWorkload({ ...params, dsv4Layer: layer });
    return sum(dsv4PerRequestWork('hybridAttention', workload));
}

function perRequestWork(kind, params) {
    const w = normalizedWorkload(params);
    if (['compressedS', 'compressedNew', 'localS', 'compressedDense',
        'compressedSparse', 'hybridAttention'].includes(kind)) {
        return dsv4PerRequestWork(kind, w);
    }
    if (kind === 'q') return w.queryLens;
    if (kind === 's') return w.seqLens;
    if (kind === 'dense') return w.seqLens.map((s, i) => s * w.queryLens[i]);
    if (kind === 'sparse') {
        return w.seqLens.map((s, i) => queryPairCount(s, w.queryLens[i], w.topk));
    }
    if (kind === 'attention') {
        return w.seqLens.map((s, i) => queryPairCount(s, w.queryLens[i], w.windowSize));
    }
    return null;
}

function rankLoads(values, dpSize) {
    const loads = Array(dpSize).fill(0);
    for (let i = 0; i < values.length; i++) {
        const rank = Math.min(dpSize - 1, Math.floor(i * dpSize / values.length));
        loads[rank] += values[i];
    }
    return loads;
}

function selectedRankWork(values, workload) {
    if (!values || workload.dpSize <= 1) return sum(values || []);
    const loads = rankLoads(values, workload.dpSize);
    return workload.dpRank == null
        ? Math.max(...loads)
        : loads[Math.min(workload.dpSize - 1, workload.dpRank)];
}

function normalizedDimName(name) {
    return String(name || '').replace(/^\u03a3/, '');
}

function tensorWorkKind(tensor, context) {
    const names = (tensor.dimNames || []).map(normalizedDimName);
    const hasQ = names.includes('S_q');
    const hasS = names.includes('S');
    const hasK = names.includes('k');
    const hasCompressed = names.includes('C');
    const hasCompressedNew = names.includes('C_new');
    const hasCompressedK = names.includes('K_c');
    const hasLocal = names.includes('L');
    const hasHybrid = names.includes('A');
    if (hasQ && hasHybrid) return 'hybridAttention';
    if (hasQ && hasCompressedK) return 'compressedSparse';
    if (hasQ && hasCompressed) return 'compressedDense';
    if (hasCompressedNew) return 'compressedNew';
    if (hasCompressed) return 'compressedS';
    if (hasLocal) return 'localS';
    if (hasQ && hasS) return context.graphId === 'dsa' ? 'dense' : 'attention';
    if (hasQ && hasK) return 'sparse';
    if (hasQ) return 'q';
    if (hasS) return 's';
    return null;
}

function tensorOtherDimProduct(tensor, kind) {
    const names = (tensor.dimNames || []).map(normalizedDimName);
    const excluded = kind === 'hybridAttention' ? new Set(['S_q', 'A'])
        : kind === 'compressedSparse' ? new Set(['S_q', 'K_c'])
        : kind === 'compressedDense' ? new Set(['S_q', 'C'])
        : kind === 'compressedNew' ? new Set(['C_new'])
        : kind === 'compressedS' ? new Set(['C'])
        : kind === 'localS' ? new Set(['L'])
        : kind === 'sparse' ? new Set(['S_q', 'k'])
        : kind === 'dense' || kind === 'attention' ? new Set(['S_q', 'S'])
        : kind === 'q' ? new Set(['S_q'])
        : kind === 's' ? new Set(['S']) : new Set();
    return tensor.shape.reduce((product, dim, i) =>
        product * (excluded.has(names[i]) ? 1 : dim), 1);
}

function hasWorkloadContext(context) {
    return context && (context.S != null || context.sumS != null || context.seqLens != null);
}

function tensorElementInfo(tensor, context = {}, { windowedCache = false } = {}) {
    const kind = hasWorkloadContext(context) ? tensorWorkKind(tensor, context) : null;
    const w = normalizedWorkload(context);
    let values = kind ? perRequestWork(kind, context) : null;
    const otherDims = kind ? tensorOtherDimProduct(tensor, kind) : 1;

    if (windowedCache && w.windowSize && kind === 's') {
        // The union of keys touched by the last q queries spans at most W+q-1 keys.
        values = w.seqLens.map((s, i) => Math.min(s, w.windowSize + w.queryLens[i] - 1));
    }

    let elements;
    if (values) {
        elements = selectedRankWork(values, w) * otherDims;
    } else {
        elements = tensorElements(tensor.shape);
    }

    if (tensor.tpSharded && w.tpSize > 1) {
        elements /= Math.max(1, Math.min(w.tpSize, tensor.tpSize || w.tpSize));
    }
    return { elements, kind };
}

function tensorByteWidth(tensor) {
    return tensor.bytesPerEl || BYTES_BF16;
}

function tensorTransfer(tensor, context, options = {}) {
    const { elements } = tensorElementInfo(tensor, context, options);
    const bytesPerEl = options.bytesPerEl || tensorByteWidth(tensor);
    return { elements, bytesPerEl, bytes: elements * bytesPerEl };
}

function transferItem(label, tensor, context, options = {}) {
    const transfer = tensorTransfer(tensor, context, options);
    return { label, shape: tensor.shape, ...transfer };
}

function hbmTransferItem(label, tensor, context, options = {}) {
    const item = transferItem(label, tensor, context, options);
    if (tensor.sramOnly) item.bytes = 0;
    return item;
}

function opTpFactor(op, inputs, output, context) {
    const globalTp = normalizedWorkload(context).tpSize;
    if (globalTp <= 1) return 1;
    return [...inputs, output]
        .filter(t => t && t.tpSharded)
        .reduce((factor, t) => Math.max(
            factor,
            Math.min(globalTp, t.tpSize || globalTp),
        ), 1);
}

function opDpFraction(op, inputs, output, context) {
    const w = normalizedWorkload(context);
    if (w.dpSize <= 1) return 1;
    const tensors = [output, ...inputs].filter(t => t && t.type !== 'weight');
    const kind = tensors.map(t => tensorWorkKind(t, context)).find(Boolean) || 'q';
    const values = perRequestWork(kind, context);
    const total = sum(values);
    return total > 0 ? selectedRankWork(values, w) / total : 1;
}

function opWorkKind(inputs, output, context) {
    const kinds = [output, ...inputs]
        .filter(t => t && t.type !== 'weight')
        .map(t => tensorWorkKind(t, context))
        .filter(Boolean);
    return ['hybridAttention', 'compressedDense', 'compressedSparse',
        'attention', 'dense', 'sparse', 'compressedNew', 'compressedS',
        'localS', 'q', 's'].find(kind => kinds.includes(kind)) || 'q';
}

function criticalDpRank(kind, context) {
    const w = normalizedWorkload(context);
    if (w.dpSize <= 1) return 0;
    const loads = rankLoads(perRequestWork(kind, context), w.dpSize);
    return loads.indexOf(Math.max(...loads));
}

function inferComputeDtype(op, inputs) {
    if (['matmul', 'compress', 'decompress'].includes(op.type) &&
        inputs.length >= 2 && inputs[0].bytesPerEl === 1 && inputs[1].bytesPerEl === 1) {
        return 'fp8';
    }
    return 'bf16';
}

// Compute critical-rank cost for an operation. Tensor shapes remain the visual
// packed-batch shapes; context supplies the exact per-request workload.
export function computeOpCost(op, tensorMap, context = {}) {
    const output = tensorMap[op.output];
    if (!output) return null;

    const inputs = op.inputs.map(id => tensorMap[id]).filter(Boolean);
    if (normalizedWorkload(context).dpSize > 1 && normalizedWorkload(context).dpRank == null) {
        context = { ...context, dpRank: criticalDpRank(opWorkKind(inputs, output, context), context) };
    }
    const computeDtype = inferComputeDtype(op, inputs);

    function finish(flops, breakdown, communicationBytes = 0) {
        const readBytes = breakdown
            .filter(item => item.direction !== 'write')
            .reduce((total, item) => total + item.bytes, 0);
        const writeBytes = breakdown
            .filter(item => item.direction === 'write')
            .reduce((total, item) => total + item.bytes, 0);
        const totalIO = readBytes + writeBytes;
        return {
            flops,
            readBytes,
            writeBytes,
            communicationBytes,
            arithmeticIntensity: totalIO > 0 ? flops / totalIO : (flops > 0 ? Infinity : 0),
            computeDtype,
            breakdown,
        };
    }

    function readItem(tensor, options = {}) {
        const storageTensor = tensor.storageAliasId
            ? tensorMap[tensor.storageAliasId] || tensor : tensor;
        return {
            ...hbmTransferItem(`Read ${storageTensor.label}`, storageTensor, context, options),
            direction: 'read',
        };
    }

    function writeItem(tensor, options = {}) {
        return { ...hbmTransferItem(`Write ${tensor.label}`, tensor, context, options), direction: 'write' };
    }

    switch (op.type) {
        case 'matmul':
        case 'compress':
        case 'decompress': {
            const A = inputs[0];
            const B = inputs[1];
            if (!A || !B) return null;

            const shA = A.shape;
            const shC = output.shape;

            const K = shA[shA.length - 1];
            const M = shA.length >= 2 ? shA[shA.length - 2] : 1;
            const N = shC[shC.length - 1];
            const batchDims = shC.length > 2 ? shC.slice(0, -2) : [];
            const batchSize = batchDims.reduce((a, b) => a * b, 1);
            const outputKind = tensorWorkKind(output, context);
            const aKind = tensorWorkKind(A, context);
            let flops;
            const pairKinds = ['dense', 'attention', 'sparse', 'hybridAttention',
                'compressedDense', 'compressedSparse'];
            if (pairKinds.includes(outputKind)) {
                flops = 2 * tensorElementInfo(output, context).elements * K;
            } else if (pairKinds.includes(aKind)) {
                flops = 2 * tensorElementInfo(A, context).elements * N;
            } else {
                flops = 2 * batchSize * M * K * N;
                flops *= opDpFraction(op, inputs, output, context);
                flops /= opTpFactor(op, inputs, output, context);
            }

            const windowedAttention = context.slidingWindow &&
                (outputKind === 'attention' || aKind === 'attention');
            const breakdown = [
                readItem(A, { windowedCache: windowedAttention && tensorWorkKind(A, context) === 's' }),
                readItem(B, { windowedCache: windowedAttention && tensorWorkKind(B, context) === 's' }),
                writeItem(output),
            ];
            let communicationBytes = 0;
            if (op.tpAllReduce && normalizedWorkload(context).tpSize > 1) {
                const tp = normalizedWorkload(context).tpSize;
                const messageBytes = tensorTransfer(output, context).bytes;
                communicationBytes = 2 * (tp - 1) / tp * messageBytes;
            }
            return finish(flops, breakdown, communicationBytes);
        }

        case 'mask': {
            // Mask + softmax: ~5 FLOPs per element (mask compare, exp, sum, div, multiply)
            const elements = tensorElementInfo(output, context).elements;
            const flops = 5 * elements;
            const breakdown = [];
            if (inputs[0]) breakdown.push(readItem(inputs[0]));
            // The causal/window mask is generated from positions in the kernel.
            breakdown.push(writeItem(output));
            return finish(flops, breakdown);
        }

        case 'rope': {
            // RoPE: 6 FLOPs per pair of dimensions (sin, cos, 2 multiplies, 1 add, 1 sub)
            // = 3 FLOPs per element (each pair covers 2 elements)
            const elements = tensorElementInfo(output, context).elements;
            const flops = 3 * elements;
            return finish(flops, [readItem(inputs[0]), writeItem(output)]);
        }

        case 'pool': {
            const poolWidth = Math.max(1, op.poolWidth || 1);
            const elements = tensorElementInfo(output, context).elements;
            const flops = 2 * poolWidth * elements;
            return finish(flops, [...inputs.map(t => readItem(t)), writeItem(output)]);
        }

        case 'add': {
            const elements = tensorElementInfo(output, context).elements;
            const flops = elements;
            return finish(flops, [...inputs.map(t => readItem(t)), writeItem(output)]);
        }

        case 'broadcast':
        case 'reshape': {
            // Logical operation, no FLOPs or real memory transfer
            return finish(0, []);
        }

        case 'flash_attn': {
            // Fused FlashAttention kernel: QK^T + mask+softmax + Attn@V
            // Only Q, K, V are read from HBM; only O is written. No intermediate HBM traffic.
            const Q = inputs.find(t => t && (t.id.startsWith('Q') || t.id === 'q_lat'));
            const K = inputs.find(t => t && (t.id.startsWith('K') || t.id === 'k_r' || t.id === 'c_KV'));
            const V = inputs.find(t => t && (t.id.startsWith('V') || t.id === 'c_KV'));
            if (!Q || !output) return null;

            // Shapes: Q is [n_h, S_q, d] (3D) or [S_q, d] (2D), K is [n_h, S, d] or [S, d]
            const shQ = Q.shape;
            const n_h = shQ.length >= 3 ? shQ[0] : 1;
            const shK = K ? K.shape : shQ;
            const d_k = shK[shK.length - 1];

            // V may be different tensor (or same as K for absorbed MLA)
            const actualV = (V && V.id !== K.id) ? V : K;
            const d_v = actualV.shape[actualV.shape.length - 1];

            const qHeads = Q.tpSharded && normalizedWorkload(context).tpSize > 1
                ? n_h / normalizedWorkload(context).tpSize : n_h;
            const pairValues = perRequestWork('attention', context);
            const localPairs = selectedRankWork(pairValues, normalizedWorkload(context));
            const flops = qHeads * localPairs * (2 * d_k + 5 + 2 * d_v);

            // Memory: read all non-mask inputs from HBM; write O to HBM
            const breakdown = [];
            const counted = new Set();
            for (const t of inputs) {
                if (!t || t.type === 'mask' || counted.has(t.id)) continue;
                counted.add(t.id);
                breakdown.push(readItem(t, {
                    windowedCache: context.slidingWindow && tensorWorkKind(t, context) === 's',
                }));
            }
            // Causal/window masks are generated in-kernel and never read from HBM.
            breakdown.push(writeItem(output));
            return finish(flops, breakdown);
        }

        case 'cache': {
            // Append to cache: copy S_q new tokens (at the cache's dtype, e.g. FP8 indexer keys)
            if (!inputs[0]) return null;
            const source = transferItem('Read new tokens', inputs[0], context, { bytesPerEl: tensorByteWidth(output) });
            const target = { ...source, label: 'Write to cache', direction: 'write' };
            source.direction = 'read';
            return finish(0, [source, target]);
        }

        case 'softmax': {
            // Standalone row-wise softmax (no mask): exp, sum, div, max-subtract ≈ 4 FLOPs/element
            const elements = tensorElementInfo(output, context).elements;
            const flops = 4 * elements;
            return finish(flops, [readItem(inputs[0]), writeItem(output)]);
        }

        case 'relu_wsum': {
            // ReLU + per-head weighted sum over indexer heads:
            // logits[t,s] = Σ_h w[t,h] · ReLU(scores[h,t,s]) → ReLU + mul + add ≈ 3 FLOPs per head-element
            const scores = inputs[0];
            const weights = inputs[1];
            if (!scores) return null;
            const flops = 3 * tensorElementInfo(scores, context).elements;
            return finish(flops, [readItem(scores), ...(weights ? [readItem(weights)] : []), writeItem(output)]);
        }

        case 'topk': {
            // Top-k selection per query row: one comparison pass over all logits
            const logits = inputs[0];
            if (!logits) return null;
            const flops = tensorElementInfo(logits, context).elements;
            return finish(flops, [
                readItem(logits),
                writeItem(output),
            ]);
        }

        case 'gather': {
            // Gather selected rows from cache by index: pure memory movement.
            // Each query token reads its own k rows — S_q·k·rowBytes is the no-reuse
            // upper bound; the gathered tiles stay in SRAM (output is sramOnly).
            const idx = inputs[0];
            const src = inputs[1];
            if (!src) return null;
            const gathered = tensorTransfer(output, context, { bytesPerEl: tensorByteWidth(src) });
            return finish(0, [
                ...(idx ? [readItem(idx)] : []),
                { label: `Gather from ${src.label}`, shape: output.shape, ...gathered, direction: 'read' },
                writeItem(output),
            ]);
        }

        default:
            return null;
    }
}

// Dense tensor-core peaks and per-GPU interconnect bandwidth. These are ideal
// hardware limits, not achieved application throughput. Source URLs are shown
// in the UI so the assumptions stay auditable.
export const GPU_SPEC_AS_OF = '2026-07-30';
export const GPU_SPECS = {
    'A100 SXM': {
        peakTFLOPS: { bf16: 312, fp8: null }, bandwidthTBs: 2.0, interconnectTBs: 0.6,
        label: 'A100 SXM (80GB)', sourceUrl: 'https://www.nvidia.com/en-us/data-center/a100/',
    },
    'H100 SXM': {
        peakTFLOPS: { bf16: 990, fp8: 1979 }, bandwidthTBs: 3.35, interconnectTBs: 0.9,
        label: 'H100 SXM (80GB)', sourceUrl: 'https://www.nvidia.com/en-us/data-center/h100/',
    },
    'B200': {
        peakTFLOPS: { bf16: 2250, fp8: 4500 }, bandwidthTBs: 7.7, interconnectTBs: 1.8,
        label: 'B200 (180GB)', sourceUrl: 'https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/',
    },
    'B300': {
        peakTFLOPS: { bf16: 3500, fp8: 7000 }, bandwidthTBs: 8.0, interconnectTBs: 1.8,
        label: 'B300 (288GB)', sourceUrl: 'https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/',
    },
};

// Compute critical-rank pipeline totals for a graph. At DP>1, each op uses the
// busiest rank's request partition; at TP>1, sharded tensors and math are local.
export function computePipelineTotals(graph, params = {}) {
    const tensorMap = {};
    for (const t of graph.tensors) tensorMap[t.id] = t;
    for (const op of graph.ops) {
        if (op.type === 'broadcast' && op.inputs.length === 1 && tensorMap[op.output]) {
            tensorMap[op.output].storageAliasId = op.inputs[0];
        }
    }
    function totalsForContext(context) {
        let totalFlops = 0;
        let totalReadBytes = 0;
        let totalWriteBytes = 0;
        let totalCommunicationBytes = 0;
        const opCosts = [];

        for (const op of graph.ops) {
            const cost = computeOpCost(op, tensorMap, context);
            if (cost) {
                totalFlops += cost.flops;
                totalReadBytes += cost.readBytes;
                totalWriteBytes += cost.writeBytes;
                totalCommunicationBytes += cost.communicationBytes || 0;
                opCosts.push({ op, cost });
            }
        }

        const totalBytes = totalReadBytes + totalWriteBytes;
        return {
            totalFlops,
            totalReadBytes,
            totalWriteBytes,
            totalBytes,
            totalCommunicationBytes,
            arithmeticIntensity: totalBytes > 0 ? totalFlops / totalBytes : 0,
            opCosts,
        };
    }

    const w = normalizedWorkload(params);
    const totals = totalsForContext({ ...params, graphId: graph.id });
    const rankTotals = w.dpSize > 1
        ? Array.from({ length: w.dpSize }, (_, dpRank) => ({
            dpRank,
            ...totalsForContext({ ...params, graphId: graph.id, dpRank }),
        }))
        : null;
    return {
        ...totals,
        rankTotals,
        tpSize: w.tpSize,
        dpSize: w.dpSize,
        scope: w.tpSize > 1 || w.dpSize > 1 ? 'per-op critical rank' : 'single GPU',
    };
}

// Arithmetic intensity threshold: ops/byte above which we're compute-bound
export function computeRooflineThreshold(gpuKey, dtype = 'bf16') {
    const gpu = GPU_SPECS[gpuKey];
    if (!gpu) return 156; // A100 default
    const peak = gpu.peakTFLOPS[dtype] || gpu.peakTFLOPS.bf16;
    return peak / gpu.bandwidthTBs;
}

export function estimateOpRoofline(cost, gpuKey) {
    const gpu = GPU_SPECS[gpuKey];
    if (!gpu) throw new Error(`Unknown GPU: ${gpuKey}`);
    const requestedDtype = cost.computeDtype || 'bf16';
    const nativePeak = gpu.peakTFLOPS[requestedDtype];
    const effectiveDtype = nativePeak ? requestedDtype : 'bf16';
    const peak = gpu.peakTFLOPS[effectiveDtype];
    const computeTime = cost.flops > 0 ? cost.flops / (peak * 1e12) : 0;
    const memoryTime = (cost.readBytes + cost.writeBytes) / (gpu.bandwidthTBs * 1e12);
    const communicationTime = (cost.communicationBytes || 0) / (gpu.interconnectTBs * 1e12);
    const kernelTime = Math.max(computeTime, memoryTime);
    return {
        computeTime,
        memoryTime,
        communicationTime,
        lowerBoundTime: kernelTime + communicationTime,
        bound: cost.flops === 0 ? 'MEM' : computeTime >= memoryTime ? 'COMPUTE' : 'MEM',
        requestedDtype,
        effectiveDtype,
        usedFallback: requestedDtype !== effectiveDtype,
    };
}

// Sequential kernels cannot share one global max(compute, memory). Sum each
// operation's own roofline lower bound, then add serialized collective time.
export function computePipelineRoofline(totals, gpuKey) {
    if (totals.rankTotals?.length) {
        const rankEstimates = totals.rankTotals.map(rank => ({
            dpRank: rank.dpRank,
            ...computePipelineRoofline({ ...rank, rankTotals: null }, gpuKey),
        }));
        return rankEstimates.reduce((critical, estimate) =>
            estimate.lowerBoundTime > critical.lowerBoundTime ? estimate : critical);
    }
    const estimates = totals.opCosts.map(({ cost }) => estimateOpRoofline(cost, gpuKey));
    const computeTime = sum(estimates.map(e => e.computeTime));
    const memoryTime = sum(estimates.map(e => e.memoryTime));
    const communicationTime = sum(estimates.map(e => e.communicationTime));
    const lowerBoundTime = sum(estimates.map(e => e.lowerBoundTime));
    const hasComputeBound = estimates.some(e => e.bound === 'COMPUTE');
    const hasMemoryBound = estimates.some(e => e.bound === 'MEM');
    const bound = hasComputeBound && hasMemoryBound ? 'MIXED'
        : hasComputeBound ? 'COMPUTE' : 'MEM';
    return {
        computeTime,
        memoryTime,
        communicationTime,
        lowerBoundTime,
        bound,
        usedFallback: estimates.some(e => e.usedFallback),
    };
}
