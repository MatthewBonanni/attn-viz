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

// Compute cost for an operation
// Returns { flops, readBytes, writeBytes, arithmeticIntensity }
export function computeOpCost(op, tensorMap) {
    const output = tensorMap[op.output];
    if (!output) return null;

    const inputs = op.inputs.map(id => tensorMap[id]).filter(Boolean);
    // Bytes for a tensor, honoring per-tensor dtype (e.g. FP8 indexer, int32 indices)
    const tBytes = (t) => tensorBytes(t.shape, t.bytesPerEl);
    // SRAM-only tensors (FlashAttention) have zero HBM transfer
    const outBytes = output.sramOnly ? 0 : tBytes(output);

    switch (op.type) {
        case 'matmul':
        case 'compress':
        case 'decompress': {
            const A = inputs[0];
            const B = inputs[1];
            if (!A || !B) return null;

            const shA = A.shape;
            const shB = B.shape;
            const shC = output.shape;

            // Inner dimension (contraction)
            const K = shA[shA.length - 1];
            // Output dimensions
            const M = shA.length >= 2 ? shA[shA.length - 2] : 1;
            const N = shC[shC.length - 1];
            // Batch dimensions from output
            const batchDims = shC.length > 2 ? shC.slice(0, -2) : [];
            const batchSize = batchDims.reduce((a, b) => a * b, 1);

            const flops = 2 * batchSize * M * K * N;
            const readA = A.sramOnly ? 0 : tBytes(A);
            const readB = B.sramOnly ? 0 : tBytes(B);
            const readBytes = readA + readB;
            const writeBytes = outBytes;

            const totalIO = readBytes + writeBytes;
            return {
                flops,
                readBytes,
                writeBytes,
                arithmeticIntensity: totalIO > 0 ? flops / totalIO : Infinity,
                breakdown: [
                    { label: `Read ${A.label}`, shape: shA, bytes: readA },
                    { label: `Read ${B.label}`, shape: shB, bytes: readB },
                    { label: `Write ${output.label}`, shape: output.shape, bytes: writeBytes },
                ],
            };
        }

        case 'mask': {
            // Mask + softmax: ~5 FLOPs per element (mask compare, exp, sum, div, multiply)
            const elements = tensorElements(output.shape);
            const flops = 5 * elements;
            // Read scores + mask, write attention weights
            const scoreBytes = (inputs[0] && !inputs[0].sramOnly) ? tBytes(inputs[0]) : 0;
            const maskBytes = (inputs[1] && !inputs[1].sramOnly) ? tensorElements(inputs[1].shape) * 1 : 0;
            const readBytes = scoreBytes + maskBytes;
            const writeBytes = outBytes;

            const totalIO = readBytes + writeBytes;
            return {
                flops,
                readBytes,
                writeBytes,
                arithmeticIntensity: totalIO > 0 ? flops / totalIO : Infinity,
                breakdown: [
                    { label: `Read ${inputs[0]?.label || 'scores'}`, shape: inputs[0]?.shape, bytes: scoreBytes },
                    { label: `Read ${inputs[1]?.label || 'mask'}`, shape: inputs[1]?.shape, bytes: maskBytes },
                    { label: `Write ${output.label}`, shape: output.shape, bytes: writeBytes },
                ],
            };
        }

        case 'rope': {
            // RoPE: 6 FLOPs per pair of dimensions (sin, cos, 2 multiplies, 1 add, 1 sub)
            // = 3 FLOPs per element (each pair covers 2 elements)
            const elements = tensorElements(output.shape);
            const flops = 3 * elements;
            const readBytes = (inputs[0] && !inputs[0].sramOnly) ? tBytes(inputs[0]) : 0;
            const writeBytes = outBytes;

            return {
                flops,
                readBytes,
                writeBytes,
                arithmeticIntensity: (readBytes + writeBytes) > 0 ? flops / (readBytes + writeBytes) : Infinity,
                breakdown: [
                    { label: `Read ${inputs[0]?.label || 'input'}`, shape: inputs[0]?.shape, bytes: readBytes },
                    { label: `Write ${output.label}`, shape: output.shape, bytes: writeBytes },
                ],
            };
        }

        case 'add': {
            const elements = tensorElements(output.shape);
            const flops = elements;
            const readBytes = inputs.reduce((sum, t) => sum + (t.sramOnly ? 0 : tBytes(t)), 0);
            const writeBytes = outBytes;
            const totalIO = readBytes + writeBytes;

            return {
                flops,
                readBytes,
                writeBytes,
                arithmeticIntensity: totalIO > 0 ? flops / totalIO : Infinity,
                breakdown: [
                    ...inputs.map(t => ({ label: `Read ${t.label}`, shape: t.shape, bytes: t.sramOnly ? 0 : tBytes(t) })),
                    { label: `Write ${output.label}`, shape: output.shape, bytes: writeBytes },
                ],
            };
        }

        case 'broadcast':
        case 'reshape': {
            // Logical operation, no FLOPs or real memory transfer
            return { flops: 0, readBytes: 0, writeBytes: 0, arithmeticIntensity: 0, breakdown: [] };
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
            const S_q = shQ.length >= 3 ? shQ[1] : shQ[0];
            const d = shQ[shQ.length - 1];

            // K determines S and inner dim for QK^T
            const shK = K ? K.shape : shQ;
            const S = shK.length >= 3 ? shK[1] : shK[0];
            const d_k = shK[shK.length - 1];

            // V may be different tensor (or same as K for absorbed MLA)
            const actualV = (V && V.id !== K.id) ? V : K;
            const d_v = actualV.shape[actualV.shape.length - 1];

            // FLOPs: 2*n_h*S_q*S*d_k (QK^T) + 5*n_h*S_q*S (mask+softmax) + 2*n_h*S_q*S*d_v (Attn@V)
            // Note: S_q and S are sumSq/sumS when B > 1, so total work across all requests is captured
            const flops = n_h * S_q * S * (2 * d_k + 5 + 2 * d_v);

            // Memory: read all non-mask inputs from HBM; write O to HBM
            const maskInput = inputs.find(t => t && t.type === 'mask');
            const breakdown = [];
            let readBytes = 0;
            const counted = new Set();
            for (const t of inputs) {
                if (!t || t.type === 'mask' || counted.has(t.id)) continue;
                counted.add(t.id);
                const bytes = tBytes(t);
                readBytes += bytes;
                breakdown.push({ label: `Read ${t.label}`, shape: t.shape, bytes });
            }
            if (maskInput) {
                const maskBytes = tensorElements(maskInput.shape) * 1;
                readBytes += maskBytes;
                breakdown.push({ label: `Read ${maskInput.label}`, shape: maskInput.shape, bytes: maskBytes });
            }
            const writeBytes = outBytes;
            breakdown.push({ label: `Write ${output.label}`, shape: output.shape, bytes: writeBytes });

            const totalIO = readBytes + writeBytes;
            return {
                flops,
                readBytes,
                writeBytes,
                arithmeticIntensity: totalIO > 0 ? flops / totalIO : Infinity,
                breakdown,
            };
        }

        case 'cache': {
            // Append to cache: copy S_q new tokens (at the cache's dtype, e.g. FP8 indexer keys)
            const inputBytes = inputs[0] ? tensorBytes(inputs[0].shape, output.bytesPerEl) : 0;
            return {
                flops: 0,
                readBytes: inputBytes,
                writeBytes: inputBytes,
                arithmeticIntensity: 0,
                breakdown: [
                    { label: `Read new tokens`, shape: inputs[0]?.shape, bytes: inputBytes },
                    { label: `Write to cache`, shape: inputs[0]?.shape, bytes: inputBytes },
                ],
            };
        }

        case 'softmax': {
            // Standalone row-wise softmax (no mask): exp, sum, div, max-subtract ≈ 4 FLOPs/element
            const elements = tensorElements(output.shape);
            const flops = 4 * elements;
            const readBytes = (inputs[0] && !inputs[0].sramOnly) ? tBytes(inputs[0]) : 0;
            const writeBytes = outBytes;
            const totalIO = readBytes + writeBytes;

            return {
                flops,
                readBytes,
                writeBytes,
                arithmeticIntensity: totalIO > 0 ? flops / totalIO : Infinity,
                breakdown: [
                    { label: `Read ${inputs[0]?.label || 'scores'}`, shape: inputs[0]?.shape, bytes: readBytes },
                    { label: `Write ${output.label}`, shape: output.shape, bytes: writeBytes },
                ],
            };
        }

        case 'relu_wsum': {
            // ReLU + per-head weighted sum over indexer heads:
            // logits[t,s] = Σ_h w[t,h] · ReLU(scores[h,t,s]) → ReLU + mul + add ≈ 3 FLOPs per head-element
            const scores = inputs[0];
            const weights = inputs[1];
            if (!scores) return null;
            const flops = 3 * tensorElements(scores.shape);
            const readScores = scores.sramOnly ? 0 : tBytes(scores);
            const readWeights = (weights && !weights.sramOnly) ? tBytes(weights) : 0;
            const readBytes = readScores + readWeights;
            const writeBytes = outBytes;
            const totalIO = readBytes + writeBytes;

            return {
                flops,
                readBytes,
                writeBytes,
                arithmeticIntensity: totalIO > 0 ? flops / totalIO : Infinity,
                breakdown: [
                    { label: `Read ${scores.label}`, shape: scores.shape, bytes: readScores },
                    ...(weights ? [{ label: `Read ${weights.label}`, shape: weights.shape, bytes: readWeights }] : []),
                    { label: `Write ${output.label}`, shape: output.shape, bytes: writeBytes },
                ],
            };
        }

        case 'topk': {
            // Top-k selection per query row: one comparison pass over all logits
            const logits = inputs[0];
            const maskInput = inputs[1];
            if (!logits) return null;
            const flops = tensorElements(logits.shape);
            const readLogits = logits.sramOnly ? 0 : tBytes(logits);
            const readMask = maskInput ? tensorElements(maskInput.shape) * 1 : 0;
            const readBytes = readLogits + readMask;
            const writeBytes = outBytes; // int32 indices via bytesPerEl
            const totalIO = readBytes + writeBytes;

            return {
                flops,
                readBytes,
                writeBytes,
                arithmeticIntensity: totalIO > 0 ? flops / totalIO : Infinity,
                breakdown: [
                    { label: `Read ${logits.label}`, shape: logits.shape, bytes: readLogits },
                    ...(maskInput ? [{ label: `Read ${maskInput.label}`, shape: maskInput.shape, bytes: readMask }] : []),
                    { label: `Write ${output.label}`, shape: output.shape, bytes: writeBytes },
                ],
            };
        }

        case 'gather': {
            // Gather selected rows from cache by index: pure memory movement.
            // Each query token reads its own k rows — S_q·k·rowBytes is the no-reuse
            // upper bound; the gathered tiles stay in SRAM (output is sramOnly).
            const idx = inputs[0];
            const src = inputs[1];
            if (!src) return null;
            const readIdx = idx ? tBytes(idx) : 0;
            const readSrc = tensorBytes(output.shape, src.bytesPerEl);
            const readBytes = readIdx + readSrc;
            const writeBytes = outBytes; // 0 when sramOnly

            return {
                flops: 0,
                readBytes,
                writeBytes,
                arithmeticIntensity: 0,
                breakdown: [
                    ...(idx ? [{ label: `Read ${idx.label}`, shape: idx.shape, bytes: readIdx }] : []),
                    { label: `Gather from ${src.label}`, shape: output.shape, bytes: readSrc },
                    { label: `Write ${output.label}`, shape: output.shape, bytes: writeBytes },
                ],
            };
        }

        default:
            return null;
    }
}

// GPU specs for roofline reference
export const GPU_SPECS = {
    'A100 SXM': { peakTFLOPS_bf16: 312,  bandwidthTBs: 2.0,  label: 'A100 SXM (80GB)' },
    'H100 SXM': { peakTFLOPS_bf16: 990,  bandwidthTBs: 3.35, label: 'H100 SXM (80GB)' },
    'B200':     { peakTFLOPS_bf16: 2250, bandwidthTBs: 7.7,  label: 'B200 (180GB)' },
    'B300':     { peakTFLOPS_bf16: 3500, bandwidthTBs: 8.0,  label: 'B300 (288GB)' },
};

// Compute pipeline totals for a graph
export function computePipelineTotals(graph) {
    const tensorMap = {};
    for (const t of graph.tensors) tensorMap[t.id] = t;

    let totalFlops = 0;
    let totalReadBytes = 0;
    let totalWriteBytes = 0;
    const opCosts = [];

    for (const op of graph.ops) {
        const cost = computeOpCost(op, tensorMap);
        if (cost) {
            totalFlops += cost.flops;
            totalReadBytes += cost.readBytes;
            totalWriteBytes += cost.writeBytes;
            opCosts.push({ op, cost });
        }
    }

    const totalBytes = totalReadBytes + totalWriteBytes;
    const arithmeticIntensity = totalBytes > 0 ? totalFlops / totalBytes : 0;

    return { totalFlops, totalReadBytes, totalWriteBytes, totalBytes, arithmeticIntensity, opCosts };
}

// Arithmetic intensity threshold: ops/byte above which we're compute-bound
export function computeRooflineThreshold(gpuKey) {
    const gpu = GPU_SPECS[gpuKey];
    if (!gpu) return 156; // A100 default
    return (gpu.peakTFLOPS_bf16 * 1e12) / (gpu.bandwidthTBs * 1e12);
}
