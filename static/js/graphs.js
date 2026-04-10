// graphs.js — Declarative graph definitions for MHA, GQA, MQA, MLA (up-projected & absorbed)

// --- Variant descriptions ---

export const VARIANT_DESCS = {
    mha: `<b>Multi-Head Attention</b> (Vaswani et al., 2017) — The standard mechanism. Each of n_h heads independently projects Q, K, V with separate weight matrices of size [D, d_h]. KV cache scales as O(2 · n_h · d_h · S) per layer. Used in GPT-2, BERT, and original Transformers.`,
    gqa: `<b>Grouped-Query Attention</b> (Ainslie et al., 2023) — A middle ground between MHA and MQA. Uses n_kv KV-head groups, where each group is shared by n_h/n_kv query heads. Reduces KV cache by n_h/n_kv× while retaining most of MHA's quality. Used in Llama 2/3, Mistral, Gemma, Qwen.`,
    mqa: `<b>Multi-Query Attention</b> (Shazeer, 2019) — All n_h query heads share a single K and V head. Reduces KV cache to O(2 · d_h · S) per layer — an n_h× reduction vs MHA. Trades some quality for dramatically faster inference. Used in PaLM, Falcon, StarCoder.`,
    mla: `<b>Multi-Head Latent Attention</b> (DeepSeek-V2, 2024) — Compresses KV representations via a low-rank bottleneck. Instead of caching full K, V tensors (size n_h · d_h per token), caches a compressed latent c_kv of size d_c ≪ n_h · d_h. Toggle between the <em>up-projected</em> view (explicit Q,K,V) and the <em>absorbed</em> view (attention computed directly in latent space via pre-computed W_QK = W_UQ @ W_UK^T).`,
};

export const MLA_MODE_DESCS = {
    upproj: `<b>Up-projected (training view)</b> — Explicitly decompresses c_kv into full K, V tensors via up-projection matrices, then computes standard multi-head attention. Conceptually clear but requires expanding to full n_h · d_h dimension.`,
    absorbed: `<b>Absorbed / latent (inference view)</b> — Pre-computes W_QK = W_UQ @ W_UK^T, enabling attention directly between compressed latent vectors c_q and c_kv without ever forming explicit K tensors. Only decompresses the context vector at the end via W_UV. This is what makes MLA efficient at inference — the KV cache stores only d_c values per token.`,
};

// --- MHA ---

export function mhaGraph(p) {
    const { B, S, n_h, d_h } = p;
    const D = n_h * d_h;
    return {
        id: 'mha', label: 'Multi-Head Attention (MHA)',
        tensors: [
            { id: 'X',      shape: [B, S, D],           label: 'X',     stage: 0, row: 1, color: '#4a90d9', dimNames: ['B','S','D'],
              desc: 'Input activation tensor. Each of B sequences has S tokens, each represented as a D-dimensional vector.' },
            { id: 'W_Q',    shape: [D, D],               label: 'Wq',   stage: 1, row: 0, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: 'Query projection weight. Maps D-dim input to D-dim query space (= n_h heads × d_h per head).' },
            { id: 'W_K',    shape: [D, D],               label: 'Wk',   stage: 1, row: 1, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: 'Key projection weight. Each head gets its own [D, d_h] slice.' },
            { id: 'W_V',    shape: [D, D],               label: 'Wv',   stage: 1, row: 2, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: 'Value projection weight. Each head gets its own [D, d_h] slice.' },
            { id: 'Q',      shape: [B, n_h, S, d_h],     label: 'Q',    stage: 2, row: 0, color: '#e74c3c', dimNames: ['B','n_h','S','d_h'],
              desc: 'Query tensor after projection and reshape. Each head attends independently.' },
            { id: 'K',      shape: [B, n_h, S, d_h],     label: 'K',    stage: 2, row: 1, color: '#2ecc71', dimNames: ['B','n_h','S','d_h'],
              desc: 'Key tensor. Cached during autoregressive generation (part of KV cache).' },
            { id: 'V',      shape: [B, n_h, S, d_h],     label: 'V',    stage: 2, row: 2, color: '#f39c12', dimNames: ['B','n_h','S','d_h'],
              desc: 'Value tensor. Cached during autoregressive generation (part of KV cache).' },
            { id: 'scores', shape: [B, n_h, S, S],       label: 'QKᵀ',  stage: 4, row: 0, color: '#9b59b6', dimNames: ['B','n_h','S','S'],
              desc: 'Raw attention scores. Each [S, S] matrix shows how much each query position attends to each key position. Scaled by 1/√d_h.' },
            { id: 'mask',   shape: [S, S],                label: 'Mask', stage: 4, row: 1, color: '#1abc9c', dimNames: ['S','S'], type: 'mask',
              desc: 'Causal mask. Lower-triangular: position i can only attend to positions ≤ i. Upper triangle is set to -∞ before softmax.' },
            { id: 'attn',   shape: [B, n_h, S, S],       label: 'Attn', stage: 6, row: 0, color: '#9b59b6', dimNames: ['B','n_h','S','S'],
              desc: 'Attention weights after masking and softmax. Each row sums to 1 — a probability distribution over key positions.' },
            { id: 'ctx',    shape: [B, n_h, S, d_h],     label: 'Ctx',  stage: 8, row: 0, color: '#e67e22', dimNames: ['B','n_h','S','d_h'],
              desc: 'Context vectors — weighted sum of values. Each position\'s output is a mixture of value vectors weighted by attention.' },
            { id: 'W_O',    shape: [D, D],                label: 'Wo',   stage: 9, row: 0, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: 'Output projection weight. Maps concatenated head outputs back to model dimension D.' },
            { id: 'out',    shape: [B, S, D],             label: 'Out',  stage: 10, row: 0, color: '#3498db', dimNames: ['B','S','D'],
              desc: 'Final attention output, added to the residual stream.' },
        ],
        ops: [
            { id: 'proj_q',  type: 'matmul',  inputs: ['X','W_Q'],       output: 'Q',      label: 'Linear',
              desc: `Project input to queries: X @ Wq, then reshape [B, S, D] → [B, n_h, S, d_h]. Each head\'s query is a d_h-dimensional vector.` },
            { id: 'proj_k',  type: 'matmul',  inputs: ['X','W_K'],       output: 'K',      label: 'Linear',
              desc: `Project input to keys: X @ Wk → [B, n_h, S, d_h]. During autoregressive generation, new keys are appended to the KV cache.` },
            { id: 'proj_v',  type: 'matmul',  inputs: ['X','W_V'],       output: 'V',      label: 'Linear',
              desc: `Project input to values: X @ Wv → [B, n_h, S, d_h]. Values are cached alongside keys for subsequent decode steps.` },
            { id: 'qkt',     type: 'matmul',  inputs: ['Q','K'],         output: 'scores', label: 'Q @ Kᵀ',
              desc: `Compute attention scores per head: Q @ K^T / √${d_h}. For each head: [S, ${d_h}] @ [${d_h}, S] → [S, S]. Scaling by 1/√d_h prevents dot products from growing too large with increasing d_h.` },
            { id: 'masking', type: 'mask',    inputs: ['scores','mask'], output: 'attn',   label: 'Mask+Softmax',
              desc: 'Apply causal mask: set scores[i,j] = -∞ where j > i (future positions), then row-wise softmax to get attention weights. Each row becomes a probability distribution over attended positions.' },
            { id: 'attn_v',  type: 'matmul',  inputs: ['attn','V'],      output: 'ctx',    label: 'Attn @ V',
              desc: `Weighted sum of values: Attn @ V. For each head: [S, S] @ [S, ${d_h}] → [S, ${d_h}]. Each output position is a convex combination of value vectors.` },
            { id: 'out_proj',type: 'matmul',  inputs: ['ctx','W_O'],     output: 'out',    label: 'Linear',
              desc: `Output projection: concat all heads [B, S, n_h·d_h] → [B, S, D] via Wo. This mixes information across heads.` },
        ]
    };
}

// --- GQA ---

export function gqaGraph(p) {
    const { B, S, n_h, d_h, n_kv } = p;
    const D = n_h * d_h;
    const d_kv = n_kv * d_h;
    const gpc = Math.floor(n_h / n_kv);
    return {
        id: 'gqa', label: 'Grouped-Query Attention (GQA)',
        tensors: [
            { id: 'X',      shape: [B, S, D],             label: 'X',     stage: 0, row: 1, color: '#4a90d9', dimNames: ['B','S','D'],
              desc: 'Input activation tensor.' },
            { id: 'W_Q',    shape: [D, D],                 label: 'Wq',   stage: 1, row: 0, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: `Query projection: full n_h=${n_h} heads, each with d_h=${d_h} dims.` },
            { id: 'W_K',    shape: [D, d_kv],              label: 'Wk',   stage: 1, row: 1, color: '#7b68ee', dimNames: ['D',`${n_kv}·d_h`], type: 'weight',
              desc: `Key projection: only n_kv=${n_kv} KV heads (not n_h=${n_h}). Output dim = n_kv × d_h = ${d_kv}. This is ${n_h/n_kv}× smaller than MHA's key projection.` },
            { id: 'W_V',    shape: [D, d_kv],              label: 'Wv',   stage: 1, row: 2, color: '#7b68ee', dimNames: ['D',`${n_kv}·d_h`], type: 'weight',
              desc: `Value projection: only n_kv=${n_kv} KV heads. Same reduction as Wk.` },
            { id: 'Q',      shape: [B, n_h, S, d_h],       label: 'Q',    stage: 2, row: 0, color: '#e74c3c', dimNames: ['B','n_h','S','d_h'],
              desc: `Full query tensor with all n_h=${n_h} heads.` },
            { id: 'K_g',    shape: [B, n_kv, S, d_h],      label: 'K',    stage: 2, row: 1, color: '#2ecc71', dimNames: ['B','n_kv','S','d_h'],
              desc: `Key tensor with only n_kv=${n_kv} heads. Notice the reduced depth compared to Q.` },
            { id: 'V_g',    shape: [B, n_kv, S, d_h],      label: 'V',    stage: 2, row: 2, color: '#f39c12', dimNames: ['B','n_kv','S','d_h'],
              desc: `Value tensor with only n_kv=${n_kv} heads.` },
            { id: 'K',      shape: [B, n_h, S, d_h],       label: "K'",   stage: 3, row: 1, color: '#2ecc71', dimNames: ['B','n_h','S','d_h'],
              desc: `Broadcast keys: each KV head is repeated ${gpc}× to match n_h=${n_h} query heads.` },
            { id: 'V',      shape: [B, n_h, S, d_h],       label: "V'",   stage: 3, row: 2, color: '#f39c12', dimNames: ['B','n_h','S','d_h'],
              desc: `Broadcast values: same expansion as keys.` },
            { id: 'scores', shape: [B, n_h, S, S],         label: 'QKᵀ',  stage: 5, row: 0, color: '#9b59b6', dimNames: ['B','n_h','S','S'],
              desc: 'Attention scores (same as MHA from this point on).' },
            { id: 'mask',   shape: [S, S],                  label: 'Mask', stage: 5, row: 1, color: '#1abc9c', dimNames: ['S','S'], type: 'mask',
              desc: 'Causal mask.' },
            { id: 'attn',   shape: [B, n_h, S, S],         label: 'Attn', stage: 7, row: 0, color: '#9b59b6', dimNames: ['B','n_h','S','S'],
              desc: 'Attention weights after mask + softmax.' },
            { id: 'ctx',    shape: [B, n_h, S, d_h],       label: 'Ctx',  stage: 9, row: 0, color: '#e67e22', dimNames: ['B','n_h','S','d_h'],
              desc: 'Context vectors.' },
            { id: 'W_O',    shape: [D, D],                  label: 'Wo',   stage: 10, row: 0, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: 'Output projection weight.' },
            { id: 'out',    shape: [B, S, D],               label: 'Out',  stage: 11, row: 0, color: '#3498db', dimNames: ['B','S','D'],
              desc: 'Attention output.' },
        ],
        ops: [
            { id: 'proj_q',    type: 'matmul',    inputs: ['X','W_Q'],       output: 'Q',      label: 'Linear',
              desc: `Project to full n_h=${n_h} query heads.` },
            { id: 'proj_k',    type: 'matmul',    inputs: ['X','W_K'],       output: 'K_g',    label: 'Linear',
              desc: `Project to n_kv=${n_kv} key heads (${n_h/n_kv}× fewer than queries).` },
            { id: 'proj_v',    type: 'matmul',    inputs: ['X','W_V'],       output: 'V_g',    label: 'Linear',
              desc: `Project to n_kv=${n_kv} value heads.` },
            { id: 'bcast_k',   type: 'broadcast', inputs: ['K_g'],           output: 'K',      label: 'Broadcast',
              desc: `Repeat each KV head ${gpc}× to match n_h=${n_h} query heads. Groups of ${gpc} query heads share one KV head. This is logically equivalent — in practice, we use grouped GEMM or index tricks instead of materializing the broadcast.` },
            { id: 'bcast_v',   type: 'broadcast', inputs: ['V_g'],           output: 'V',      label: 'Broadcast',
              desc: `Repeat each value head ${gpc}× to match query heads.` },
            { id: 'qkt',       type: 'matmul',    inputs: ['Q','K'],         output: 'scores', label: 'Q @ Kᵀ',
              desc: `Attention scores: Q @ K^T / √${d_h} → [B, n_h, S, S].` },
            { id: 'masking',   type: 'mask',      inputs: ['scores','mask'], output: 'attn',   label: 'Mask+Softmax',
              desc: 'Apply causal mask then row-wise softmax.' },
            { id: 'attn_v',    type: 'matmul',    inputs: ['attn','V'],      output: 'ctx',    label: 'Attn @ V',
              desc: 'Weighted sum of values.' },
            { id: 'out_proj',  type: 'matmul',    inputs: ['ctx','W_O'],     output: 'out',    label: 'Linear',
              desc: 'Output projection: concat heads → D.' },
        ]
    };
}

// --- MQA ---

export function mqaGraph(p) {
    const { B, S, n_h, d_h } = p;
    const D = n_h * d_h;
    return {
        id: 'mqa', label: 'Multi-Query Attention (MQA)',
        tensors: [
            { id: 'X',      shape: [B, S, D],           label: 'X',     stage: 0, row: 1, color: '#4a90d9', dimNames: ['B','S','D'],
              desc: 'Input activation tensor.' },
            { id: 'W_Q',    shape: [D, D],               label: 'Wq',   stage: 1, row: 0, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: `Query projection: full n_h=${n_h} heads.` },
            { id: 'W_K',    shape: [D, d_h],             label: 'Wk',   stage: 1, row: 1, color: '#7b68ee', dimNames: ['D','d_h'], type: 'weight',
              desc: `Key projection: single head only. Output dim = d_h=${d_h}, which is ${n_h}× smaller than MHA.` },
            { id: 'W_V',    shape: [D, d_h],             label: 'Wv',   stage: 1, row: 2, color: '#7b68ee', dimNames: ['D','d_h'], type: 'weight',
              desc: `Value projection: single head only.` },
            { id: 'Q',      shape: [B, n_h, S, d_h],     label: 'Q',    stage: 2, row: 0, color: '#e74c3c', dimNames: ['B','n_h','S','d_h'],
              desc: `Full query tensor with n_h=${n_h} heads.` },
            { id: 'K_1',    shape: [B, 1, S, d_h],       label: 'K',    stage: 2, row: 1, color: '#2ecc71', dimNames: ['B','1','S','d_h'],
              desc: `Single key head. Notice the minimal depth — only 1 head vs Q's ${n_h}. This is what makes MQA's KV cache so small.` },
            { id: 'V_1',    shape: [B, 1, S, d_h],       label: 'V',    stage: 2, row: 2, color: '#f39c12', dimNames: ['B','1','S','d_h'],
              desc: 'Single value head.' },
            { id: 'K',      shape: [B, n_h, S, d_h],     label: "K'",   stage: 3, row: 1, color: '#2ecc71', dimNames: ['B','n_h','S','d_h'],
              desc: `Broadcast key to all ${n_h} query heads.` },
            { id: 'V',      shape: [B, n_h, S, d_h],     label: "V'",   stage: 3, row: 2, color: '#f39c12', dimNames: ['B','n_h','S','d_h'],
              desc: `Broadcast value to all ${n_h} query heads.` },
            { id: 'scores', shape: [B, n_h, S, S],       label: 'QKᵀ',  stage: 5, row: 0, color: '#9b59b6', dimNames: ['B','n_h','S','S'],
              desc: 'Attention scores.' },
            { id: 'mask',   shape: [S, S],                label: 'Mask', stage: 5, row: 1, color: '#1abc9c', dimNames: ['S','S'], type: 'mask',
              desc: 'Causal mask.' },
            { id: 'attn',   shape: [B, n_h, S, S],       label: 'Attn', stage: 7, row: 0, color: '#9b59b6', dimNames: ['B','n_h','S','S'],
              desc: 'Attention weights after mask + softmax.' },
            { id: 'ctx',    shape: [B, n_h, S, d_h],     label: 'Ctx',  stage: 9, row: 0, color: '#e67e22', dimNames: ['B','n_h','S','d_h'],
              desc: 'Context vectors.' },
            { id: 'W_O',    shape: [D, D],                label: 'Wo',   stage: 10, row: 0, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: 'Output projection.' },
            { id: 'out',    shape: [B, S, D],             label: 'Out',  stage: 11, row: 0, color: '#3498db', dimNames: ['B','S','D'],
              desc: 'Attention output.' },
        ],
        ops: [
            { id: 'proj_q',    type: 'matmul',    inputs: ['X','W_Q'],       output: 'Q',      label: 'Linear',
              desc: 'Project to full query heads.' },
            { id: 'proj_k',    type: 'matmul',    inputs: ['X','W_K'],       output: 'K_1',    label: 'Linear',
              desc: `Project to a single key head: [B, S, D] @ [D, ${d_h}] → [B, 1, S, ${d_h}]. This is the core MQA insight — one K,V pair shared across all ${n_h} query heads.` },
            { id: 'proj_v',    type: 'matmul',    inputs: ['X','W_V'],       output: 'V_1',    label: 'Linear',
              desc: 'Project to a single value head.' },
            { id: 'bcast_k',   type: 'broadcast', inputs: ['K_1'],           output: 'K',      label: 'Broadcast',
              desc: `Broadcast single K head to all ${n_h} query heads. All heads see identical keys.` },
            { id: 'bcast_v',   type: 'broadcast', inputs: ['V_1'],           output: 'V',      label: 'Broadcast',
              desc: `Broadcast single V head to all ${n_h} query heads.` },
            { id: 'qkt',       type: 'matmul',    inputs: ['Q','K'],         output: 'scores', label: 'Q @ Kᵀ',
              desc: `Attention scores: Q @ K^T / √${d_h}.` },
            { id: 'masking',   type: 'mask',      inputs: ['scores','mask'], output: 'attn',   label: 'Mask+Softmax',
              desc: 'Apply causal mask then softmax.' },
            { id: 'attn_v',    type: 'matmul',    inputs: ['attn','V'],      output: 'ctx',    label: 'Attn @ V',
              desc: 'Weighted sum of values.' },
            { id: 'out_proj',  type: 'matmul',    inputs: ['ctx','W_O'],     output: 'out',    label: 'Linear',
              desc: 'Output projection.' },
        ]
    };
}

// --- MLA (Up-projected / training view) ---

export function mlaUpprojGraph(p) {
    const { B, S, n_h, d_h, d_c } = p;
    const D = n_h * d_h;
    return {
        id: 'mla_upproj', label: 'MLA — Up-projected (Training View)',
        tensors: [
            { id: 'X',      shape: [B, S, D],           label: 'X',       stage: 0, row: 1, color: '#4a90d9', dimNames: ['B','S','D'],
              desc: 'Input activation tensor.' },
            // Q compression path
            { id: 'W_DQ',   shape: [D, d_c],             label: 'W↓q',    stage: 1, row: 0, color: '#7b68ee', dimNames: ['D','d_c'], type: 'weight',
              desc: `Query down-projection: D=${D} → d_c=${d_c}. Compresses queries into the latent space.` },
            // KV compression path
            { id: 'W_DKV',  shape: [D, d_c],             label: 'W↓kv',   stage: 1, row: 1, color: '#7b68ee', dimNames: ['D','d_c'], type: 'weight',
              desc: `KV down-projection: D=${D} → d_c=${d_c}. This ${(D/d_c).toFixed(1)}× compression is the key to MLA's cache efficiency.` },
            { id: 'c_Q',    shape: [B, S, d_c],          label: 'c_q',    stage: 2, row: 0, color: '#c0392b', dimNames: ['B','S','d_c'],
              desc: `Compressed query latent. Dimension reduced from D=${D} to d_c=${d_c}.` },
            { id: 'c_KV',   shape: [B, S, d_c],          label: 'c_kv',   stage: 2, row: 1, color: '#e67e22', dimNames: ['B','S','d_c'], badge: 'KV CACHE',
              desc: `Compressed KV latent — this is what gets cached! Only d_c=${d_c} values per token instead of 2·n_h·d_h = ${2*n_h*d_h}. A ${(2*n_h*d_h/d_c).toFixed(1)}× cache reduction.` },
            { id: 'W_UQ',   shape: [d_c, D],             label: 'W↑q',    stage: 3, row: -1, color: '#7b68ee', dimNames: ['d_c','D'], type: 'weight',
              desc: `Query up-projection: d_c=${d_c} → D=${D}. Decompresses queries back to full attention dimension.` },
            { id: 'W_UK',   shape: [d_c, D],             label: 'W↑k',    stage: 3, row: 1, color: '#7b68ee', dimNames: ['d_c','D'], type: 'weight',
              desc: `Key up-projection: d_c=${d_c} → D=${D}.` },
            { id: 'W_UV',   shape: [d_c, D],             label: 'W↑v',    stage: 3, row: 2, color: '#7b68ee', dimNames: ['d_c','D'], type: 'weight',
              desc: `Value up-projection: d_c=${d_c} → D=${D}.` },
            { id: 'Q',      shape: [B, n_h, S, d_h],     label: 'Q',      stage: 4, row: 0, color: '#e74c3c', dimNames: ['B','n_h','S','d_h'],
              desc: 'Decompressed queries, reshaped to per-head form.' },
            { id: 'K',      shape: [B, n_h, S, d_h],     label: 'K',      stage: 4, row: 1, color: '#2ecc71', dimNames: ['B','n_h','S','d_h'],
              desc: 'Decompressed keys.' },
            { id: 'V',      shape: [B, n_h, S, d_h],     label: 'V',      stage: 4, row: 2, color: '#f39c12', dimNames: ['B','n_h','S','d_h'],
              desc: 'Decompressed values.' },
            { id: 'scores', shape: [B, n_h, S, S],       label: 'QKᵀ',    stage: 6, row: 0, color: '#9b59b6', dimNames: ['B','n_h','S','S'],
              desc: 'Attention scores (standard from here).' },
            { id: 'mask',   shape: [S, S],                label: 'Mask',   stage: 6, row: 1, color: '#1abc9c', dimNames: ['S','S'], type: 'mask',
              desc: 'Causal mask.' },
            { id: 'attn',   shape: [B, n_h, S, S],       label: 'Attn',   stage: 8, row: 0, color: '#9b59b6', dimNames: ['B','n_h','S','S'],
              desc: 'Attention weights.' },
            { id: 'ctx',    shape: [B, n_h, S, d_h],     label: 'Ctx',    stage: 10, row: 0, color: '#e67e22', dimNames: ['B','n_h','S','d_h'],
              desc: 'Context vectors.' },
            { id: 'W_O',    shape: [D, D],                label: 'Wo',     stage: 11, row: 0, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: 'Output projection.' },
            { id: 'out',    shape: [B, S, D],             label: 'Out',    stage: 12, row: 0, color: '#3498db', dimNames: ['B','S','D'],
              desc: 'Attention output.' },
        ],
        ops: [
            { id: 'compress_q',  type: 'compress',   inputs: ['X','W_DQ'],        output: 'c_Q',    label: 'Down-proj',
              desc: `Compress queries: X @ W↓q → c_q [B, S, ${d_c}].` },
            { id: 'compress_kv', type: 'compress',   inputs: ['X','W_DKV'],       output: 'c_KV',   label: 'Down-proj',
              desc: `Compress KV: X @ W↓kv → c_kv [B, S, ${d_c}]. This low-rank bottleneck (D=${D} → d_c=${d_c}) is the key to MLA's KV cache efficiency. Only c_kv is cached, not the full K and V.` },
            { id: 'decomp_q',   type: 'decompress', inputs: ['c_Q','W_UQ'],      output: 'Q',      label: 'Up-proj Q',
              desc: `Decompress queries: c_q @ W↑q → Q, reshape to [B, n_h, S, d_h].` },
            { id: 'decomp_k',   type: 'decompress', inputs: ['c_KV','W_UK'],     output: 'K',      label: 'Up-proj K',
              desc: `Decompress keys: c_kv @ W↑k → K [B, n_h, S, d_h].` },
            { id: 'decomp_v',   type: 'decompress', inputs: ['c_KV','W_UV'],     output: 'V',      label: 'Up-proj V',
              desc: `Decompress values: c_kv @ W↑v → V [B, n_h, S, d_h].` },
            { id: 'qkt',        type: 'matmul',     inputs: ['Q','K'],           output: 'scores', label: 'Q @ Kᵀ',
              desc: `Attention scores: Q @ K^T / √${d_h}.` },
            { id: 'masking',    type: 'mask',       inputs: ['scores','mask'],   output: 'attn',   label: 'Mask+Softmax',
              desc: 'Apply causal mask then softmax.' },
            { id: 'attn_v',     type: 'matmul',     inputs: ['attn','V'],        output: 'ctx',    label: 'Attn @ V',
              desc: 'Weighted sum of values.' },
            { id: 'out_proj',   type: 'matmul',     inputs: ['ctx','W_O'],       output: 'out',    label: 'Linear',
              desc: 'Output projection → [B, S, D].' },
        ]
    };
}

// --- MLA (Absorbed / inference view) ---

export function mlaAbsorbedGraph(p) {
    const { B, S, n_h, d_h, d_c } = p;
    const D = n_h * d_h;
    return {
        id: 'mla_absorbed', label: 'MLA — Absorbed (Inference View)',
        tensors: [
            { id: 'X',        shape: [B, S, D],           label: 'X',       stage: 0, row: 1, color: '#4a90d9', dimNames: ['B','S','D'],
              desc: 'Input activation tensor.' },
            // Compression
            { id: 'W_DQ',     shape: [D, d_c],             label: 'W↓q',    stage: 1, row: 0, color: '#7b68ee', dimNames: ['D','d_c'], type: 'weight',
              desc: `Query down-projection: D=${D} → d_c=${d_c}.` },
            { id: 'W_DKV',    shape: [D, d_c],             label: 'W↓kv',   stage: 1, row: 1, color: '#7b68ee', dimNames: ['D','d_c'], type: 'weight',
              desc: `KV down-projection: D=${D} → d_c=${d_c}.` },
            { id: 'c_Q',      shape: [B, S, d_c],          label: 'c_q',    stage: 2, row: 0, color: '#c0392b', dimNames: ['B','S','d_c'],
              desc: `Compressed query latent.` },
            { id: 'c_KV',     shape: [B, S, d_c],          label: 'c_kv',   stage: 2, row: 1, color: '#e67e22', dimNames: ['B','S','d_c'], badge: 'KV CACHE',
              desc: `Compressed KV latent — the only thing cached. d_c=${d_c} values per token vs 2·n_h·d_h=${2*n_h*d_h} in MHA.` },
            // Absorbed weight: W_QK = W_UQ @ W_UK^T, per head [d_c, d_c]
            { id: 'W_QK',     shape: [n_h, d_c, d_c],      label: 'W_QK',   stage: 3, row: 0, color: '#7b68ee', dimNames: ['n_h','d_c','d_c'], type: 'weight', badge: 'ABSORBED',
              desc: `Absorbed QK weight: W_QK = W_UQ @ W_UK^T, pre-computed per head. Shape [n_h, d_c, d_c]. This lets us compute attention scores directly between c_q and c_kv without ever forming explicit Q, K tensors. Each head has a [${d_c}, ${d_c}] matrix.` },
            // Projected query in latent space
            { id: 'q_lat',    shape: [B, n_h, S, d_c],     label: "q'",     stage: 4, row: 0, color: '#e74c3c', dimNames: ['B','n_h','S','d_c'],
              desc: `Absorbed query: c_q projected through W_QK per head. Still in the d_c=${d_c} latent space — never expanded to d_h=${d_h}.` },
            // Scores
            { id: 'scores',   shape: [B, n_h, S, S],       label: 'QKᵀ',    stage: 6, row: 0, color: '#9b59b6', dimNames: ['B','n_h','S','S'],
              desc: `Latent attention scores: q' @ c_kv^T. Computed in d_c-dimensional latent space (vs d_h in standard attention).` },
            { id: 'mask',     shape: [S, S],                label: 'Mask',   stage: 6, row: 1, color: '#1abc9c', dimNames: ['S','S'], type: 'mask',
              desc: 'Causal mask.' },
            { id: 'attn',     shape: [B, n_h, S, S],       label: 'Attn',   stage: 8, row: 0, color: '#9b59b6', dimNames: ['B','n_h','S','S'],
              desc: 'Attention weights.' },
            // Latent context: attn @ c_kv (still in latent space)
            { id: 'c_ctx',    shape: [B, n_h, S, d_c],     label: 'c_ctx',  stage: 10, row: 0, color: '#e67e22', dimNames: ['B','n_h','S','d_c'], badge: 'LATENT',
              desc: `Latent context: Attn @ c_kv. Attention-weighted sum of compressed values — still in d_c=${d_c} latent space. No decompression of values has happened yet.` },
            // Value decompression at the end
            { id: 'W_UV',     shape: [d_c, D],              label: 'W↑v',    stage: 11, row: 0, color: '#7b68ee', dimNames: ['d_c','D'], type: 'weight',
              desc: `Value up-projection: d_c=${d_c} → D=${D}. Only applied once to the context (not to every cached V token). This is why absorbed MLA is efficient — decompression happens after attention, not before.` },
            { id: 'ctx',      shape: [B, n_h, S, d_h],     label: 'Ctx',    stage: 12, row: 0, color: '#e67e22', dimNames: ['B','n_h','S','d_h'],
              desc: 'Decompressed context vectors.' },
            { id: 'W_O',      shape: [D, D],                label: 'Wo',     stage: 13, row: 0, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: 'Output projection.' },
            { id: 'out',      shape: [B, S, D],             label: 'Out',    stage: 14, row: 0, color: '#3498db', dimNames: ['B','S','D'],
              desc: 'Attention output.' },
        ],
        ops: [
            { id: 'compress_q',    type: 'compress',   inputs: ['X','W_DQ'],          output: 'c_Q',    label: 'Down-proj',
              desc: `Compress queries: X @ W↓q → c_q [B, S, ${d_c}].` },
            { id: 'compress_kv',   type: 'compress',   inputs: ['X','W_DKV'],         output: 'c_KV',   label: 'Down-proj',
              desc: `Compress KV: X @ W↓kv → c_kv [B, S, ${d_c}]. Only c_kv is cached.` },
            { id: 'absorbed_proj', type: 'matmul',     inputs: ['c_Q','W_QK'],        output: 'q_lat',  label: 'Absorbed proj',
              desc: `Apply absorbed QK weight: c_q @ W_QK → q' [B, n_h, S, d_c]. This replaces two operations (Q up-projection + K up-projection) with one, staying in the d_c-dimensional latent space.` },
            { id: 'latent_qk',    type: 'matmul',     inputs: ['q_lat','c_KV'],      output: 'scores', label: "q' @ c_kv^T",
              desc: `Latent dot product: q' @ c_kv^T → [B, n_h, S, S]. Attention scores computed entirely in the d_c=${d_c} latent space, without ever expanding to d_h=${d_h}.` },
            { id: 'masking',      type: 'mask',       inputs: ['scores','mask'],     output: 'attn',   label: 'Mask+Softmax',
              desc: 'Apply causal mask then softmax.' },
            { id: 'latent_attn_v',type: 'matmul',     inputs: ['attn','c_KV'],       output: 'c_ctx',  label: 'Attn @ c_kv',
              desc: `Latent attention: Attn @ c_kv → c_ctx [B, n_h, S, d_c]. Weighted sum of compressed values — still in latent space. c_kv is reused from cache (no decompression needed).` },
            { id: 'decomp_ctx',   type: 'decompress', inputs: ['c_ctx','W_UV'],      output: 'ctx',    label: 'Up-proj V',
              desc: `Decompress context: c_ctx @ W↑v → ctx [B, n_h, S, d_h]. Decompression only happens once at the end, not for every cached token.` },
            { id: 'out_proj',     type: 'matmul',     inputs: ['ctx','W_O'],         output: 'out',    label: 'Linear',
              desc: 'Output projection → [B, S, D].' },
        ]
    };
}
