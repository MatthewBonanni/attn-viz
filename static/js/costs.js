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

export function tensorBytes(shape) {
    return tensorElements(shape) * BYTES_BF16;
}

// Compute cost for an operation
// Returns { flops, readBytes, writeBytes, arithmeticIntensity }
export function computeOpCost(op, tensorMap) {
    const output = tensorMap[op.output];
    if (!output) return null;

    const inputs = op.inputs.map(id => tensorMap[id]).filter(Boolean);
    // SRAM-only tensors (FlashAttention) have zero HBM transfer
    const outBytes = output.sramOnly ? 0 : tensorBytes(output.shape);

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
            const readA = A.sramOnly ? 0 : tensorBytes(shA);
            const readB = B.sramOnly ? 0 : tensorBytes(shB);
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
            const scoreBytes = (inputs[0] && !inputs[0].sramOnly) ? tensorBytes(inputs[0].shape) : 0;
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
            const readBytes = (inputs[0] && !inputs[0].sramOnly) ? tensorBytes(inputs[0].shape) : 0;
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
            const readBytes = inputs.reduce((sum, t) => sum + (t.sramOnly ? 0 : tensorBytes(t.shape)), 0);
            const writeBytes = outBytes;
            const totalIO = readBytes + writeBytes;

            return {
                flops,
                readBytes,
                writeBytes,
                arithmeticIntensity: totalIO > 0 ? flops / totalIO : Infinity,
                breakdown: [
                    ...inputs.map(t => ({ label: `Read ${t.label}`, shape: t.shape, bytes: t.sramOnly ? 0 : tensorBytes(t.shape) })),
                    { label: `Write ${output.label}`, shape: output.shape, bytes: writeBytes },
                ],
            };
        }

        case 'broadcast': {
            // Logical operation, no FLOPs or real memory transfer
            return { flops: 0, readBytes: 0, writeBytes: 0, arithmeticIntensity: 0, breakdown: [] };
        }

        case 'cache': {
            // Append to cache: copy S_q new tokens
            const inputBytes = inputs[0] ? tensorBytes(inputs[0].shape) : 0;
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
