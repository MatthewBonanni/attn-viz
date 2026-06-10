// graphs.js — Declarative graph definitions for MHA, GQA, MQA, MLA (up-projected & absorbed)

// --- Variant descriptions ---

export const VARIANT_DESCS = {
    mha: `<b>Multi-Head Attention</b> (Vaswani et al., 2017) — The standard mechanism. Each of n_h heads independently projects Q, K, V with separate weight matrices of size [D, d_h]. KV cache scales as O(2 · n_h · d_h · S) per layer. Used in GPT-2, BERT, and original Transformers.`,
    gqa: `<b>Grouped-Query Attention</b> (Ainslie et al., 2023) — A middle ground between MHA and MQA. Uses n_kv KV-head groups, where each group is shared by n_h/n_kv query heads. Reduces KV cache by n_h/n_kv× while retaining most of MHA's quality. Used in Llama 2/3, Mistral, Gemma, Qwen.`,
    mqa: `<b>Multi-Query Attention</b> (Shazeer, 2019) — All n_h query heads share a single K and V head. Reduces KV cache to O(2 · d_h · S) per layer — an n_h× reduction vs MHA. Trades some quality for dramatically faster inference. Used in PaLM, Falcon, StarCoder.`,
    mla: `<b>Multi-Head Latent Attention</b> (DeepSeek-V2, 2024) — Compresses KV representations via a low-rank bottleneck. Instead of caching full K, V tensors (size n_h · d_h per token), caches a compressed latent c_kv of size d_c ≪ n_h · d_h. During inference, the <b>up-projected</b> path is used for compute-bound <em>prefills</em>, while the <b>absorbed</b> path is used for memory-bandwidth-bound <em>decodes</em>. Both are shown below.`,
    dsa: `<b>DeepSeek Sparse Attention</b> (DeepSeek-V3.2, 2025) — MLA plus a learned <b>lightning indexer</b>: a small FP8 scorer (n_i heads of d_i dims, MQA-style shared key) that rates every cached token per query, then keeps only the <b>top-k</b> (k=2048). Core attention runs in MLA's absorbed (MQA-style) form over just those k tokens — for both prefill and decode — cutting attention cost from O(S) to O(k) per query at long context.`,
};

// Dim label helpers: when B > 1, S_q/S labels become ΣS_q/ΣS
function sqLabel(B) { return B > 1 ? '\u03a3S_q' : 'S_q'; }
function sLabel(B)  { return B > 1 ? '\u03a3S' : 'S'; }

// --- MHA ---

export function mhaGraph(p) {
    const { n_h, d_h, B } = p;
    const S_q = p.sumSq || p.S_q || p.S;
    const S   = p.sumS  || p.S;
    const D = n_h * d_h;
    const lq = sqLabel(B), ls = sLabel(B);
    return {
        id: 'mha', label: 'Multi-Head Attention (MHA)',
        tensors: [
            { id: 'X',        shape: [S_q, D],           label: 'X',       stage: 0, row: 1, color: '#4a90d9', dimNames: [lq,'D'],
              desc: 'Input activation tensor. S_q query tokens, each represented as a D-dimensional vector.' },
            { id: 'W_Q',      shape: [D, D],               label: 'Wq',     stage: 1, row: 0, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: 'Query projection weight. Maps D-dim input to D-dim query space (= n_h heads × d_h per head).' },
            { id: 'W_K',      shape: [D, D],               label: 'Wk',     stage: 1, row: 1, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: 'Key projection weight. Each head gets its own [D, d_h] slice.' },
            { id: 'W_V',      shape: [D, D],               label: 'Wv',     stage: 1, row: 2, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: 'Value projection weight. Each head gets its own [D, d_h] slice.' },
            { id: 'Q_flat',   shape: [S_q, D],           label: 'Q_flat', stage: 2, row: 0, color: '#e74c3c', dimNames: [lq,'D'],
              desc: `Raw query projection result before splitting into heads.` },
            { id: 'K_flat',   shape: [S_q, D],           label: 'K_flat', stage: 2, row: 1, color: '#2ecc71', dimNames: [lq,'D'],
              desc: `Raw key projection result before splitting into heads.` },
            { id: 'V_flat',   shape: [S_q, D],           label: 'V_flat', stage: 2, row: 2, color: '#f39c12', dimNames: [lq,'D'],
              desc: `Raw value projection result before splitting into heads.` },
            { id: 'Q',        shape: [n_h, S_q, d_h],   label: 'Q',      stage: 3, row: 0, color: '#e74c3c', dimNames: ['n_h',lq,'d_h'],
              desc: `Query tensor reshaped into per-head form: D split into n_h=${n_h} heads of d_h=${d_h} dims. Each head attends independently.` },
            { id: 'K_new',    shape: [n_h, S_q, d_h],   label: 'K_new',  stage: 3, row: 1, color: '#2ecc71', dimNames: ['n_h',lq,'d_h'],
              desc: `Newly projected keys reshaped into per-head form for the current S_q tokens.` },
            { id: 'V_new',    shape: [n_h, S_q, d_h],   label: 'V_new',  stage: 3, row: 2, color: '#f39c12', dimNames: ['n_h',lq,'d_h'],
              desc: `Newly projected values reshaped into per-head form for the current S_q tokens.` },
            { id: 'Q_r',      shape: [n_h, S_q, d_h],   label: "Q'",     stage: 4, row: 0, color: '#e74c3c', dimNames: ['n_h',lq,'d_h'],
              desc: 'Query with RoPE (Rotary Position Embedding) applied. Pairs of dimensions are rotated by position-dependent angles, encoding absolute position so that dot products depend on relative position.' },
            { id: 'K_new_r',  shape: [n_h, S_q, d_h],   label: "K_new'", stage: 4, row: 1, color: '#2ecc71', dimNames: ['n_h',lq,'d_h'],
              desc: `Keys with RoPE applied. RoPE is applied before storing in the KV cache so that cached keys already carry positional information.` },
            { id: 'K_r',      shape: [n_h, S, d_h],     label: "K'",     stage: 5, row: 4, color: '#2ecc71', dimNames: ['n_h',ls,'d_h'], cache: true,
              desc: `Full key cache: new RoPE'd keys appended to cached keys → S total. During prefill (S_q=S) the cache starts empty.` },
            { id: 'V_cache',  shape: [n_h, S, d_h],     label: 'V',      stage: 5, row: 5, color: '#f39c12', dimNames: ['n_h',ls,'d_h'], cache: true,
              desc: `Full value cache: new values appended to cached values → S total.` },
            { id: 'scores',   shape: [n_h, S_q, S],     label: 'QKᵀ',    stage: 6, row: 0, color: '#9b59b6', dimNames: ['n_h',lq,ls],
              desc: 'Raw attention scores. Each [S_q, S] matrix shows how much each query position attends to each key position. Scaled by 1/√d_h.' },
            { id: 'mask',     shape: [S_q, S],              label: 'Mask',   stage: 6, row: 1, color: '#1abc9c', dimNames: [lq,ls], type: 'mask',
              desc: 'Causal mask. Lower-triangular: position i can only attend to positions ≤ i. Upper triangle is set to -∞ before softmax.' },
            { id: 'attn',     shape: [n_h, S_q, S],     label: 'Attn',   stage: 8, row: 0, color: '#9b59b6', dimNames: ['n_h',lq,ls],
              desc: 'Attention weights after masking and softmax. Each row sums to 1 — a probability distribution over key positions.' },
            { id: 'attn_head',shape: [n_h, S_q, d_h],   label: 'AV_head',stage: 10, row: 0, color: '#e67e22', dimNames: ['n_h',lq,'d_h'],
              desc: `Per-head context vectors: weighted sum of values, before merging across heads.` },
            { id: 'ctx',      shape: [S_q, D],           label: 'Ctx',    stage: 11, row: 1, color: '#e67e22', dimNames: [lq,'D'],
              desc: `Context vectors — n_h=${n_h} heads merged via view: n_h × d_h = ${D} = D.` },
            { id: 'W_O',      shape: [D, D],               label: 'Wo',     stage: 11, row: 2, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: 'Output projection weight. Maps merged head outputs back to model dimension D.' },
            { id: 'out',      shape: [S_q, D],           label: 'Out',    stage: 12, row: 1, color: '#3498db', dimNames: [lq,'D'],
              desc: 'Final attention output, added to the residual stream.' },
        ],
        ops: [
            { id: 'proj_q',  type: 'matmul',  inputs: ['X','W_Q'],           output: 'Q_flat',   label: 'Linear',
              desc: `Project input to queries: X @ Wq.` },
            { id: 'proj_k',  type: 'matmul',  inputs: ['X','W_K'],           output: 'K_flat',   label: 'Linear',
              desc: `Project input to keys: X @ Wk. Only the S_q new tokens are projected; previous keys are already in the cache.` },
            { id: 'proj_v',  type: 'matmul',  inputs: ['X','W_V'],           output: 'V_flat',   label: 'Linear',
              desc: `Project input to values: X @ Wv. Only new tokens are projected.` },
            { id: 'view_q',  type: 'reshape', inputs: ['Q_flat'],            output: 'Q',        label: 'View',
              desc: `Reshape query: [S_q, D] → [n_h, S_q, d_h]. Splits the D=${D} dimension into n_h=${n_h} heads of d_h=${d_h}. This is a zero-cost metadata operation — no data is moved.` },
            { id: 'view_k',  type: 'reshape', inputs: ['K_flat'],            output: 'K_new',    label: 'View',
              desc: `Reshape key: [S_q, D] → [n_h, S_q, d_h]. Zero-cost metadata operation.` },
            { id: 'view_v',  type: 'reshape', inputs: ['V_flat'],            output: 'V_new',    label: 'View',
              desc: `Reshape value: [S_q, D] → [n_h, S_q, d_h]. Zero-cost metadata operation.` },
            { id: 'rope_q',  type: 'rope',    inputs: ['Q'],                 output: 'Q_r',      label: 'RoPE',
              desc: `Apply Rotary Position Embedding to queries. Each pair of dimensions is rotated by an angle proportional to the token position.` },
            { id: 'rope_k',  type: 'rope',    inputs: ['K_new'],             output: 'K_new_r',  label: 'RoPE',
              desc: `Apply RoPE to new keys before caching. Cached keys already carry their positional encoding from when they were first computed.` },
            { id: 'cache_k', type: 'cache',   inputs: ['K_new_r'],           output: 'K_r',      label: 'KV Cache', alignX: 'kv_cache',
              desc: `Append new RoPE'd keys to the cache: [n_h, S_q, d_h] → [n_h, S, d_h]. During prefill (S_q=S), the cache starts empty.` },
            { id: 'cache_v', type: 'cache',   inputs: ['V_new'],             output: 'V_cache',  label: 'KV Cache', alignX: 'kv_cache',
              desc: `Append new values to the cache: [n_h, S_q, d_h] → [n_h, S, d_h].` },
            { id: 'qkt',     type: 'matmul',  inputs: ['Q_r','K_r'],         output: 'scores',   label: 'Q @ Kᵀ',
              desc: `Compute attention scores per head: Q' @ K'^T / √${d_h}. Queries attend over all S cached keys.` },
            { id: 'masking', type: 'mask',    inputs: ['scores','mask'],     output: 'attn',     label: 'Mask+Softmax',
              desc: 'Apply causal mask: set scores[i,j] = -∞ where j > i (future positions), then row-wise softmax to get attention weights.' },
            { id: 'attn_v',  type: 'matmul',  inputs: ['attn','V_cache'],    output: 'attn_head', label: 'Attn @ V',
              desc: `Weighted sum of values per head: Attn @ V → [n_h, S_q, ${d_h}].` },
            { id: 'view_out',type: 'reshape', inputs: ['attn_head'],         output: 'ctx',      label: 'View',
              desc: `View heads as flat: [n_h, S_q, d_h] → [S_q, D]. Merges n_h=${n_h} × d_h=${d_h} = D=${D}. Zero-cost metadata operation.` },
            { id: 'out_proj',type: 'matmul',  inputs: ['ctx','W_O'],         output: 'out',      label: 'Linear',
              desc: `Output projection: [S_q, D] @ [D, D] → [S_q, D] via Wo. This mixes information across heads.` },
        ],
        groups: [
            { label: 'KV PROJECTION & CACHE', color: '#16a085',
              tensors: ['W_K','W_V','K_flat','V_flat','K_new','V_new','K_new_r','K_r','V_cache'],
              ops: ['proj_k','proj_v','view_k','view_v','rope_k','cache_k','cache_v'],
              desc: `Keys and values are projected from the input (X @ Wk, X @ Wv), reshaped into per-head form, RoPE is applied to keys, and both are appended to the KV cache. The cache grows by S_q tokens each step. In MHA, each of n_h=${n_h} heads has its own K and V, so total cache per token = 2 × n_h × d_h = ${2*n_h*d_h}.` },
            { label: 'ATTENTION', color: '#9b59b6', padTop: 40,
              tensors: ['scores','mask','attn','attn_head','ctx'],
              ops: ['qkt','masking','attn_v','view_out'],
              desc: `Attention scores are computed as Q' @ K'^T / √d_h, giving a [S_q, S] matrix per head. A causal mask sets future positions to -∞, then softmax normalizes each row. The attention-weighted sum of values (Attn @ V) produces per-head context vectors, then viewed as a flat tensor across heads.` },
            { label: 'OUTPUT PROJECTION', color: '#e67e22', padTop: 40,
              tensors: ['W_O','out'],
              ops: ['out_proj'],
              desc: `The merged context vectors (D=${D}) are projected through Wo back to model dimension.` },
        ]
    };
}

// --- GQA ---

export function gqaGraph(p) {
    const { n_h, d_h, n_kv, B } = p;
    const S_q = p.sumSq || p.S_q || p.S;
    const S   = p.sumS  || p.S;
    const D = n_h * d_h;
    const d_kv = n_kv * d_h;
    const gpc = Math.floor(n_h / n_kv);
    const lq = sqLabel(B), ls = sLabel(B);
    return {
        id: 'gqa', label: 'Grouped-Query Attention (GQA)',
        tensors: [
            { id: 'X',        shape: [S_q, D],             label: 'X',       stage: 0, row: 1, color: '#4a90d9', dimNames: [lq,'D'],
              desc: 'Input activation tensor.' },
            { id: 'W_Q',      shape: [D, D],                 label: 'Wq',     stage: 1, row: 0, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: `Query projection: full n_h=${n_h} heads, each with d_h=${d_h} dims.` },
            { id: 'W_K',      shape: [D, d_kv],              label: 'Wk',     stage: 1, row: 1, color: '#7b68ee', dimNames: ['D',`${n_kv}·d_h`], type: 'weight',
              desc: `Key projection: only n_kv=${n_kv} KV heads (not n_h=${n_h}). Output dim = n_kv × d_h = ${d_kv}. This is ${n_h/n_kv}× smaller than MHA's key projection.` },
            { id: 'W_V',      shape: [D, d_kv],              label: 'Wv',     stage: 1, row: 2, color: '#7b68ee', dimNames: ['D',`${n_kv}·d_h`], type: 'weight',
              desc: `Value projection: only n_kv=${n_kv} KV heads. Same reduction as Wk.` },
            { id: 'Q_flat',   shape: [S_q, D],             label: 'Q_flat', stage: 2, row: 0, color: '#e74c3c', dimNames: [lq,'D'],
              desc: `Raw query projection result before splitting into heads.` },
            { id: 'K_flat',   shape: [S_q, d_kv],          label: 'K_flat', stage: 2, row: 1, color: '#2ecc71', dimNames: [lq,`${n_kv}·d_h`],
              desc: `Raw key projection result before splitting into n_kv=${n_kv} heads.` },
            { id: 'V_flat',   shape: [S_q, d_kv],          label: 'V_flat', stage: 2, row: 2, color: '#f39c12', dimNames: [lq,`${n_kv}·d_h`],
              desc: `Raw value projection result before splitting into n_kv=${n_kv} heads.` },
            { id: 'Q',        shape: [n_h, S_q, d_h],     label: 'Q',      stage: 3, row: 0, color: '#e74c3c', dimNames: ['n_h',lq,'d_h'],
              desc: `Full query tensor reshaped into n_h=${n_h} heads.` },
            { id: 'K_g',      shape: [n_kv, S_q, d_h],    label: 'K_new',  stage: 3, row: 1, color: '#2ecc71', dimNames: ['n_kv',lq,'d_h'],
              desc: `Newly projected keys reshaped into n_kv=${n_kv} heads for the current S_q tokens.` },
            { id: 'V_g',      shape: [n_kv, S_q, d_h],    label: 'V_new',  stage: 3, row: 2, color: '#f39c12', dimNames: ['n_kv',lq,'d_h'],
              desc: `Newly projected values reshaped into n_kv=${n_kv} heads for the current S_q tokens.` },
            { id: 'Q_r',      shape: [n_h, S_q, d_h],     label: "Q'",     stage: 4, row: 0, color: '#e74c3c', dimNames: ['n_h',lq,'d_h'],
              desc: 'Query with RoPE applied. Position encoded before broadcast/attention.' },
            { id: 'K_gr',     shape: [n_kv, S_q, d_h],    label: "K_new'", stage: 4, row: 1, color: '#2ecc71', dimNames: ['n_kv',lq,'d_h'],
              desc: `Keys with RoPE applied. RoPE is applied before caching.` },
            { id: 'K_cache',  shape: [n_kv, S, d_h],      label: "K'",     stage: 5, row: 4, color: '#2ecc71', dimNames: ['n_kv',ls,'d_h'], cache: true,
              desc: `Full key cache with n_kv=${n_kv} heads: new RoPE'd keys appended to cached keys → S total.` },
            { id: 'V_cache',  shape: [n_kv, S, d_h],      label: 'V',      stage: 5, row: 5, color: '#f39c12', dimNames: ['n_kv',ls,'d_h'], cache: true,
              desc: `Full value cache with n_kv=${n_kv} heads: new values appended → S total.` },
            { id: 'K',        shape: [n_h, S, d_h],       label: 'K↑',     stage: 6, row: 1, color: '#2ecc71', dimNames: ['n_h',ls,'d_h'],
              desc: `Broadcast keys: each KV head is repeated ${gpc}× to match n_h=${n_h} query heads.` },
            { id: 'V_bcast',  shape: [n_h, S, d_h],       label: 'V↑',     stage: 6, row: 2, color: '#f39c12', dimNames: ['n_h',ls,'d_h'],
              desc: `Broadcast values: same expansion as keys.` },
            { id: 'scores',   shape: [n_h, S_q, S],       label: 'QKᵀ',    stage: 8, row: 0, color: '#9b59b6', dimNames: ['n_h',lq,ls],
              desc: 'Attention scores (same as MHA from this point on).' },
            { id: 'mask',     shape: [S_q, S],                label: 'Mask',   stage: 8, row: 1, color: '#1abc9c', dimNames: [lq,ls], type: 'mask',
              desc: 'Causal mask.' },
            { id: 'attn',     shape: [n_h, S_q, S],       label: 'Attn',   stage: 10, row: 0, color: '#9b59b6', dimNames: ['n_h',lq,ls],
              desc: 'Attention weights after mask + softmax.' },
            { id: 'attn_head',shape: [n_h, S_q, d_h],     label: 'AV_head',stage: 12, row: 0, color: '#e67e22', dimNames: ['n_h',lq,'d_h'],
              desc: `Per-head context vectors before merging across heads.` },
            { id: 'ctx',      shape: [S_q, D],             label: 'Ctx',    stage: 13, row: 1, color: '#e67e22', dimNames: [lq,'D'],
              desc: 'Context vectors, merged across heads via view.' },
            { id: 'W_O',      shape: [D, D],                 label: 'Wo',     stage: 13, row: 2, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: 'Output projection weight.' },
            { id: 'out',      shape: [S_q, D],             label: 'Out',    stage: 14, row: 1, color: '#3498db', dimNames: [lq,'D'],
              desc: 'Attention output.' },
        ],
        ops: [
            { id: 'proj_q',    type: 'matmul',    inputs: ['X','W_Q'],            output: 'Q_flat',   label: 'Linear',
              desc: `Project to full query space: X @ Wq.` },
            { id: 'proj_k',    type: 'matmul',    inputs: ['X','W_K'],            output: 'K_flat',   label: 'Linear',
              desc: `Project S_q new tokens to KV space: X @ Wk → [S_q, ${d_kv}].` },
            { id: 'proj_v',    type: 'matmul',    inputs: ['X','W_V'],            output: 'V_flat',   label: 'Linear',
              desc: `Project S_q new tokens to value space: X @ Wv → [S_q, ${d_kv}].` },
            { id: 'view_q',    type: 'reshape',   inputs: ['Q_flat'],             output: 'Q',        label: 'View',
              desc: `Reshape query: [S_q, D] → [n_h, S_q, d_h]. Splits D=${D} into n_h=${n_h} heads of d_h=${d_h}. Zero-cost metadata operation.` },
            { id: 'view_k',    type: 'reshape',   inputs: ['K_flat'],             output: 'K_g',      label: 'View',
              desc: `Reshape key: [S_q, ${d_kv}] → [n_kv, S_q, d_h]. Splits into n_kv=${n_kv} heads. Zero-cost metadata operation.` },
            { id: 'view_v',    type: 'reshape',   inputs: ['V_flat'],             output: 'V_g',      label: 'View',
              desc: `Reshape value: [S_q, ${d_kv}] → [n_kv, S_q, d_h]. Zero-cost metadata operation.` },
            { id: 'rope_q',    type: 'rope',      inputs: ['Q'],                  output: 'Q_r',      label: 'RoPE',
              desc: 'Apply RoPE to queries before attention computation.' },
            { id: 'rope_k',    type: 'rope',      inputs: ['K_g'],                output: 'K_gr',     label: 'RoPE',
              desc: `Apply RoPE to the n_kv=${n_kv} new key heads before caching.` },
            { id: 'cache_k',   type: 'cache',     inputs: ['K_gr'],               output: 'K_cache',  label: 'KV Cache', alignX: 'kv_cache',
              desc: `Append new RoPE'd keys to the cache → [n_kv, S, d_h].` },
            { id: 'cache_v',   type: 'cache',     inputs: ['V_g'],                output: 'V_cache',  label: 'KV Cache', alignX: 'kv_cache',
              desc: `Append new values to the cache → [n_kv, S, d_h].` },
            { id: 'bcast_k',   type: 'broadcast', inputs: ['K_cache'],            output: 'K',        label: 'Broadcast',
              desc: `Repeat each cached KV head ${gpc}× to match n_h=${n_h} query heads. In practice, grouped GEMM or index tricks avoid materializing this.` },
            { id: 'bcast_v',   type: 'broadcast', inputs: ['V_cache'],            output: 'V_bcast',  label: 'Broadcast',
              desc: `Repeat each cached value head ${gpc}× to match query heads.` },
            { id: 'qkt',       type: 'matmul',    inputs: ['Q_r','K'],            output: 'scores',   label: 'Q @ Kᵀ',
              desc: `Attention scores: Q' @ K'^T / √${d_h} → [n_h, S_q, S].` },
            { id: 'masking',   type: 'mask',      inputs: ['scores','mask'],      output: 'attn',     label: 'Mask+Softmax',
              desc: 'Apply causal mask then row-wise softmax.' },
            { id: 'attn_v',    type: 'matmul',    inputs: ['attn','V_bcast'],     output: 'attn_head', label: 'Attn @ V',
              desc: `Weighted sum of values per head → [n_h, S_q, d_h].` },
            { id: 'view_out',  type: 'reshape',   inputs: ['attn_head'],          output: 'ctx',      label: 'View',
              desc: `View heads as flat: [n_h, S_q, d_h] → [S_q, D]. Merges n_h=${n_h} × d_h=${d_h} = D=${D}. Zero-cost metadata operation.` },
            { id: 'out_proj',  type: 'matmul',    inputs: ['ctx','W_O'],          output: 'out',      label: 'Linear',
              desc: `Output projection: [S_q, D] @ [D, D] → [S_q, D] via Wo. This mixes information across heads.` },
        ],
        groups: [
            { label: 'KV PROJECTION & CACHE', color: '#16a085',
              tensors: ['W_K','W_V','K_flat','V_flat','K_g','V_g','K_gr','K_cache','V_cache','K','V_bcast'],
              ops: ['proj_k','proj_v','view_k','view_v','rope_k','cache_k','cache_v','bcast_k','bcast_v'],
              desc: `GQA uses n_kv=${n_kv} KV heads instead of n_h=${n_h}, reducing cache to 2 × n_kv × d_h = ${2*n_kv*d_h} per token (${n_h/n_kv}× smaller than MHA). After caching, each KV head is broadcast ${gpc}× to match the query head count. In practice, grouped GEMM avoids materializing the broadcast.` },
            { label: 'ATTENTION', color: '#9b59b6', padTop: 40,
              tensors: ['scores','mask','attn','attn_head','ctx'],
              ops: ['qkt','masking','attn_v','view_out'],
              desc: `Attention scores are computed as Q' @ K'^T / √d_h, giving a [S_q, S] matrix per head. A causal mask sets future positions to -∞, then softmax normalizes each row. The attention-weighted sum of values (Attn @ V) produces per-head context vectors, then viewed as a flat tensor across heads.` },
            { label: 'OUTPUT PROJECTION', color: '#e67e22', padTop: 40,
              tensors: ['W_O','out'],
              ops: ['out_proj'],
              desc: `The merged context vectors are projected through Wo back to model dimension D.` },
        ]
    };
}

// --- MQA ---

export function mqaGraph(p) {
    const { n_h, d_h, B } = p;
    const S_q = p.sumSq || p.S_q || p.S;
    const S   = p.sumS  || p.S;
    const D = n_h * d_h;
    const lq = sqLabel(B), ls = sLabel(B);
    return {
        id: 'mqa', label: 'Multi-Query Attention (MQA)',
        tensors: [
            { id: 'X',         shape: [S_q, D],           label: 'X',       stage: 0, row: 1, color: '#4a90d9', dimNames: [lq,'D'],
              desc: 'Input activation tensor.' },
            { id: 'W_Q',       shape: [D, D],               label: 'Wq',     stage: 1, row: 0, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: `Query projection: full n_h=${n_h} heads.` },
            { id: 'W_K',       shape: [D, d_h],             label: 'Wk',     stage: 1, row: 1, color: '#7b68ee', dimNames: ['D','d_h'], type: 'weight',
              desc: `Key projection: single head only. Output dim = d_h=${d_h}, which is ${n_h}× smaller than MHA.` },
            { id: 'W_V',       shape: [D, d_h],             label: 'Wv',     stage: 1, row: 2, color: '#7b68ee', dimNames: ['D','d_h'], type: 'weight',
              desc: `Value projection: single head only.` },
            { id: 'Q_flat',    shape: [S_q, D],           label: 'Q_flat', stage: 2, row: 0, color: '#e74c3c', dimNames: [lq,'D'],
              desc: `Raw query projection result before splitting into heads.` },
            { id: 'K_1_flat',  shape: [S_q, d_h],         label: 'K_flat', stage: 2, row: 1, color: '#2ecc71', dimNames: [lq,'d_h'],
              desc: `Raw single-head key projection result.` },
            { id: 'V_1_flat',  shape: [S_q, d_h],         label: 'V_flat', stage: 2, row: 2, color: '#f39c12', dimNames: [lq,'d_h'],
              desc: `Raw single-head value projection result.` },
            { id: 'Q',         shape: [n_h, S_q, d_h],   label: 'Q',      stage: 3, row: 0, color: '#e74c3c', dimNames: ['n_h',lq,'d_h'],
              desc: `Full query tensor reshaped into n_h=${n_h} heads.` },
            { id: 'K_1',       shape: [1, S_q, d_h],     label: 'K_new',  stage: 3, row: 1, color: '#2ecc71', dimNames: ['1',lq,'d_h'],
              desc: `Single key head reshaped with explicit head dimension. MQA's core insight — one K,V pair shared across all ${n_h} query heads.` },
            { id: 'V_1',       shape: [1, S_q, d_h],     label: 'V_new',  stage: 3, row: 2, color: '#f39c12', dimNames: ['1',lq,'d_h'],
              desc: 'Single value head reshaped with explicit head dimension.' },
            { id: 'Q_r',       shape: [n_h, S_q, d_h],   label: "Q'",     stage: 4, row: 0, color: '#e74c3c', dimNames: ['n_h',lq,'d_h'],
              desc: 'Query with RoPE applied.' },
            { id: 'K_1r',      shape: [1, S_q, d_h],     label: "K_new'", stage: 4, row: 1, color: '#2ecc71', dimNames: ['1',lq,'d_h'],
              desc: 'Single key head with RoPE applied before caching.' },
            { id: 'K_1_cache', shape: [1, S, d_h],       label: "K'",     stage: 5, row: 4, color: '#2ecc71', dimNames: ['1',ls,'d_h'], cache: true,
              desc: `Full single-head key cache: new RoPE'd keys appended → S total.` },
            { id: 'V_1_cache', shape: [1, S, d_h],       label: 'V',      stage: 5, row: 5, color: '#f39c12', dimNames: ['1',ls,'d_h'], cache: true,
              desc: `Full single-head value cache: new values appended → S total.` },
            { id: 'K',         shape: [n_h, S, d_h],     label: 'K↑',     stage: 6, row: 1, color: '#2ecc71', dimNames: ['n_h',ls,'d_h'],
              desc: `Broadcast cached key to all ${n_h} query heads.` },
            { id: 'V_bcast',   shape: [n_h, S, d_h],     label: 'V↑',     stage: 6, row: 2, color: '#f39c12', dimNames: ['n_h',ls,'d_h'],
              desc: `Broadcast cached value to all ${n_h} query heads.` },
            { id: 'scores',    shape: [n_h, S_q, S],     label: 'QKᵀ',    stage: 8, row: 0, color: '#9b59b6', dimNames: ['n_h',lq,ls],
              desc: 'Attention scores.' },
            { id: 'mask',      shape: [S_q, S],              label: 'Mask',   stage: 8, row: 1, color: '#1abc9c', dimNames: [lq,ls], type: 'mask',
              desc: 'Causal mask.' },
            { id: 'attn',      shape: [n_h, S_q, S],     label: 'Attn',   stage: 10, row: 0, color: '#9b59b6', dimNames: ['n_h',lq,ls],
              desc: 'Attention weights after mask + softmax.' },
            { id: 'attn_head', shape: [n_h, S_q, d_h],   label: 'AV_head',stage: 12, row: 0, color: '#e67e22', dimNames: ['n_h',lq,'d_h'],
              desc: `Per-head context vectors before merging across heads.` },
            { id: 'ctx',       shape: [S_q, D],           label: 'Ctx',    stage: 13, row: 1, color: '#e67e22', dimNames: [lq,'D'],
              desc: 'Context vectors, merged across heads via view.' },
            { id: 'W_O',       shape: [D, D],               label: 'Wo',     stage: 13, row: 2, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: 'Output projection.' },
            { id: 'out',       shape: [S_q, D],           label: 'Out',    stage: 14, row: 1, color: '#3498db', dimNames: [lq,'D'],
              desc: 'Attention output.' },
        ],
        ops: [
            { id: 'proj_q',    type: 'matmul',    inputs: ['X','W_Q'],            output: 'Q_flat',     label: 'Linear',
              desc: `Project to full query space: X @ Wq.` },
            { id: 'proj_k',    type: 'matmul',    inputs: ['X','W_K'],            output: 'K_1_flat',   label: 'Linear',
              desc: `Project S_q new tokens to a single key head: X @ Wk → [S_q, ${d_h}].` },
            { id: 'proj_v',    type: 'matmul',    inputs: ['X','W_V'],            output: 'V_1_flat',   label: 'Linear',
              desc: `Project S_q new tokens to a single value head: X @ Wv → [S_q, ${d_h}].` },
            { id: 'view_q',    type: 'reshape',   inputs: ['Q_flat'],             output: 'Q',          label: 'View',
              desc: `Reshape query: [S_q, D] → [n_h, S_q, d_h]. Splits D=${D} into n_h=${n_h} heads. Zero-cost metadata operation.` },
            { id: 'view_k',    type: 'reshape',   inputs: ['K_1_flat'],           output: 'K_1',        label: 'View',
              desc: `Reshape key: [S_q, d_h] → [1, S_q, d_h]. Adds explicit head dimension. Zero-cost metadata operation.` },
            { id: 'view_v',    type: 'reshape',   inputs: ['V_1_flat'],           output: 'V_1',        label: 'View',
              desc: `Reshape value: [S_q, d_h] → [1, S_q, d_h]. Zero-cost metadata operation.` },
            { id: 'rope_q',    type: 'rope',      inputs: ['Q'],                  output: 'Q_r',        label: 'RoPE',
              desc: 'Apply RoPE to queries.' },
            { id: 'rope_k',    type: 'rope',      inputs: ['K_1'],                output: 'K_1r',       label: 'RoPE',
              desc: 'Apply RoPE to the single new key head before caching.' },
            { id: 'cache_k',   type: 'cache',     inputs: ['K_1r'],               output: 'K_1_cache',  label: 'KV Cache', alignX: 'kv_cache',
              desc: `Append new RoPE'd keys to the single-head cache → [1, S, d_h].` },
            { id: 'cache_v',   type: 'cache',     inputs: ['V_1'],                output: 'V_1_cache',  label: 'KV Cache', alignX: 'kv_cache',
              desc: `Append new values to the single-head cache → [1, S, d_h].` },
            { id: 'bcast_k',   type: 'broadcast', inputs: ['K_1_cache'],          output: 'K',          label: 'Broadcast',
              desc: `Broadcast cached single K head to all ${n_h} query heads. All heads see identical keys.` },
            { id: 'bcast_v',   type: 'broadcast', inputs: ['V_1_cache'],          output: 'V_bcast',    label: 'Broadcast',
              desc: `Broadcast cached single V head to all ${n_h} query heads.` },
            { id: 'qkt',       type: 'matmul',    inputs: ['Q_r','K'],            output: 'scores',     label: 'Q @ Kᵀ',
              desc: `Attention scores: Q' @ K'^T / √${d_h}.` },
            { id: 'masking',   type: 'mask',      inputs: ['scores','mask'],      output: 'attn',       label: 'Mask+Softmax',
              desc: 'Apply causal mask then softmax.' },
            { id: 'attn_v',    type: 'matmul',    inputs: ['attn','V_bcast'],     output: 'attn_head',  label: 'Attn @ V',
              desc: `Weighted sum of values per head → [n_h, S_q, d_h].` },
            { id: 'view_out',  type: 'reshape',   inputs: ['attn_head'],          output: 'ctx',        label: 'View',
              desc: `View heads as flat: [n_h, S_q, d_h] → [S_q, D]. Merges n_h=${n_h} × d_h=${d_h} = D=${D}. Zero-cost metadata operation.` },
            { id: 'out_proj',  type: 'matmul',    inputs: ['ctx','W_O'],          output: 'out',        label: 'Linear',
              desc: `Output projection: [S_q, D] @ [D, D] → [S_q, D] via Wo. This mixes information across heads.` },
        ],
        groups: [
            { label: 'KV PROJECTION & CACHE', color: '#16a085',
              tensors: ['W_K','W_V','K_1_flat','V_1_flat','K_1','V_1','K_1r','K_1_cache','V_1_cache','K','V_bcast'],
              ops: ['proj_k','proj_v','view_k','view_v','rope_k','cache_k','cache_v','bcast_k','bcast_v'],
              desc: `MQA projects to a single KV head (n_kv=1), so cache per token = 2 × d_h = ${2*d_h} — an ${n_h}× reduction vs MHA. The single K,V pair is broadcast to all ${n_h} query heads. This maximizes memory bandwidth efficiency at the cost of some quality.` },
            { label: 'ATTENTION', color: '#9b59b6', padTop: 40,
              tensors: ['scores','mask','attn','attn_head','ctx'],
              ops: ['qkt','masking','attn_v','view_out'],
              desc: `Attention scores are computed as Q' @ K'^T / √d_h, giving a [S_q, S] matrix per head. A causal mask sets future positions to -∞, then softmax normalizes each row. All heads see identical K,V but different Q projections, so Attn @ V produces diverse per-head context vectors.` },
            { label: 'OUTPUT PROJECTION', color: '#e67e22', padTop: 40,
              tensors: ['W_O','out'],
              ops: ['out_proj'],
              desc: `The merged context vectors are projected through Wo. Despite sharing K,V, each head's different Q creates different attention patterns, so the D=${D}-dim output is still expressive.` },
        ]
    };
}

// --- MLA (Up-projected / training view) ---

export function mlaUpprojGraph(p) {
    const { n_h, d_h, d_c, d_r, B } = p;
    const d_q = p.d_q || d_c;   // Q latent dim (Lq); distinct from KV latent dim (Lkv = d_c)
    const S_q = p.sumSq || p.S_q || p.S;
    const S   = p.sumS  || p.S;
    const D = n_h * d_h;
    const dr = d_r || 64;
    const NP = n_h * d_h;       // total nope dimension (N * P, where P = d_h)
    const NR = n_h * dr;        // total rope dimension (N * R)
    const lq = sqLabel(B), ls = sLabel(B);
    return {
        id: 'mla_upproj', label: 'MLA — MHA-style (Up-projected)',
        tensors: [
            { id: 'X',      shape: [S_q, D],           label: 'X',        stage: 0, row: 1, color: '#4a90d9', dimNames: [lq,'D'],
              desc: 'Input activation tensor.' },
            // Q compression path
            { id: 'W_DQ',   shape: [D, d_q],             label: 'W↓q',     stage: 1, row: 0, color: '#7b68ee', dimNames: ['D','d_q'], type: 'weight',
              desc: `Query down-projection: D=${D} → d_q=${d_q}. Compresses queries into the query latent space.` },
            // KV compression path
            { id: 'W_DKV',  shape: [D, d_c],             label: 'W↓kv',    stage: 1, row: 1, color: '#7b68ee', dimNames: ['D','d_c'], type: 'weight',
              desc: `KV down-projection: D=${D} → d_c=${d_c}. This ${(D/d_c).toFixed(1)}× compression is the key to MLA's cache efficiency.` },
            // RoPE key projection (decoupled from KV compression)
            { id: 'W_KR',   shape: [D, dr],              label: 'W_kr',    stage: 1, row: 2, color: '#7b68ee', dimNames: ['D','d_r'], type: 'weight',
              desc: `RoPE key projection: D=${D} → d_r=${dr}. Decoupled from the KV compression path because RoPE doesn't commute with low-rank compression.` },
            { id: 'c_Q',      shape: [S_q, d_q],        label: 'c_q',     stage: 2, row: 0, color: '#c0392b', dimNames: [lq,'d_q'],
              desc: `Compressed query latent. Dimension reduced from D=${D} to d_q=${d_q}.` },
            { id: 'c_KV_new', shape: [S_q, d_c],        label: 'c_kv_new', stage: 2, row: 1, color: '#e67e22', dimNames: [lq,'d_c'],
              desc: `Newly compressed KV latent for S_q tokens. Only d_c=${d_c} values per token instead of 2·n_h·d_h = ${2*n_h*d_h}. A ${(2*n_h*d_h/d_c).toFixed(1)}× compression.` },
            { id: 'k_rp_new', shape: [S_q, dr],         label: 'k_r_new', stage: 2, row: 2, color: '#ff7043', dimNames: [lq,'d_r'],
              desc: `Newly projected pre-RoPE key embedding for S_q tokens. Projected from X independently of the KV compression path.` },
            { id: 'k_r_new',  shape: [S_q, dr],         label: "k_r_new'", stage: 3, row: 2, color: '#ff7043', dimNames: [lq,'d_r'],
              desc: `RoPE applied to new decoupled keys before caching.` },
            { id: 'c_KV',   shape: [S, d_c],          label: 'c_kv',    stage: 4, row: 5, color: '#e67e22', dimNames: [ls,'d_c'], cache: true,
              desc: `Full compressed KV cache: new latents appended → S total. Only d_c=${d_c} values per token. Total cache per token = d_c + d_r = ${d_c + dr}.` },
            { id: 'k_r',    shape: [S, dr],           label: "k_r'",    stage: 4, row: 6, color: '#ff7043', dimNames: [ls,'d_r'], cache: true,
              desc: `Full RoPE key cache: new RoPE'd keys appended → S total. Cached alongside c_kv because RoPE doesn't commute with low-rank compression.` },
            // Q nope path: c_q → q_nope
            { id: 'W_UQ',   shape: [d_q, NP],            label: 'W↑q',     stage: 4, row: 0, color: '#7b68ee', dimNames: ['d_q','N·P'], type: 'weight',
              desc: `Query nope up-projection: d_q=${d_q} → N·P=${NP}. Decompresses the content (non-positional) part of queries.` },
            // Q rope path: c_q → q_pe
            { id: 'W_QR',   shape: [d_q, NR],            label: 'W_qr',    stage: 4, row: 1, color: '#7b68ee', dimNames: ['d_q','N·R'], type: 'weight',
              desc: `Query RoPE projection: d_q=${d_q} → N·R=${NR}. Separate weight from W↑q because the RoPE dimensions require their own projection from the query latent.` },
            // KV up-projection
            { id: 'W_UK',   shape: [d_c, NP],            label: 'W↑k',     stage: 4, row: 2, color: '#7b68ee', dimNames: ['d_c','N·P'], type: 'weight',
              desc: `Key up-projection: d_c=${d_c} → N·P=${NP}. Decompresses content keys from the KV latent.` },
            { id: 'W_UV',   shape: [d_c, D],             label: 'W↑v',     stage: 4, row: 3, color: '#7b68ee', dimNames: ['d_c','N·V'], type: 'weight',
              desc: `Value up-projection: d_c=${d_c} → N·V=${D}. V head dim = d_h=${d_h}.` },
            // Decompressed Q nope
            { id: 'q_nope', shape: [n_h, S_q, d_h],   label: 'q_nope',  stage: 6, row: 0, color: '#e74c3c', dimNames: ['n_h',lq,'P'],
              desc: `Content (no-RoPE) queries: c_q @ W↑q, viewed as [n_h, S_q, P]. P = d_h = ${d_h}.` },
            // Decompressed Q rope (pre-RoPE)
            { id: 'q_pe_pre', shape: [n_h, S_q, dr],  label: 'q_pe',    stage: 6, row: 1, color: '#ff7043', dimNames: ['n_h',lq,'R'],
              desc: `Positional (pre-RoPE) queries: c_q @ W_qr, viewed as [n_h, S_q, R]. R = d_r = ${dr}.` },
            // K nope
            { id: 'k_nope', shape: [n_h, S, d_h],     label: 'k_nope',  stage: 6, row: 2, color: '#2ecc71', dimNames: ['n_h',ls,'P'],
              desc: 'Decompressed content keys in per-head form (no RoPE — positional info comes from k_r).' },
            // V
            { id: 'V',      shape: [n_h, S, d_h],     label: 'V',       stage: 6, row: 3, color: '#f39c12', dimNames: ['n_h',ls,'V'],
              desc: 'Decompressed values in per-head form.' },
            // RoPE'd Q rope
            { id: 'q_pe',   shape: [n_h, S_q, dr],    label: "q_pe'",   stage: 7, row: 1, color: '#ff7043', dimNames: ['n_h',lq,'R'],
              desc: `Query RoPE embeddings after applying rotary position encoding.` },
            // Attention scores
            { id: 's_content', shape: [n_h, S_q, S],  label: "q·kᵀ",    stage: 9, row: 1, color: '#9b59b6', dimNames: ['n_h',lq,ls],
              desc: `Content attention scores: q_nope @ k_nope^T. P=${d_h} inner dimension.` },
            { id: 's_rope', shape: [n_h, S_q, S],     label: "q_r'k_r'ᵀ", stage: 9, row: 2, color: '#9b59b6', dimNames: ['n_h',lq,ls],
              desc: `RoPE attention scores: q_pe' @ k_r'^T. R=${dr} inner dimension.` },
            { id: 'scores', shape: [n_h, S_q, S],     label: 'Scores',  stage: 10, row: 1, color: '#9b59b6', dimNames: ['n_h',lq,ls],
              desc: `Combined attention scores = content + positional. Equivalent to SDPA with QK headdim = P + R = ${d_h + dr}.` },
            { id: 'mask',   shape: [S_q, S],              label: 'Mask',    stage: 10, row: 2, color: '#1abc9c', dimNames: [lq,ls], type: 'mask',
              desc: 'Causal mask.' },
            { id: 'attn',   shape: [n_h, S_q, S],     label: 'Attn',    stage: 11, row: 1, color: '#9b59b6', dimNames: ['n_h',lq,ls],
              desc: 'Attention weights.' },
            { id: 'attn_head', shape: [n_h, S_q, d_h], label: 'AV_head', stage: 14, row: 1, color: '#e67e22', dimNames: ['n_h',lq,'V'],
              desc: `Per-head context vectors: Attn @ V → [n_h, S_q, V]. V headdim = d_h = ${d_h}.` },
            { id: 'ctx',    shape: [S_q, D],           label: 'Ctx',     stage: 15, row: 1, color: '#e67e22', dimNames: [lq,'D'],
              desc: `Context vectors — n_h=${n_h} heads merged via view: n_h × d_h = ${D} = D.` },
            { id: 'W_O',    shape: [D, D],               label: 'Wo',      stage: 15, row: 2, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: 'Output projection.' },
            { id: 'out',    shape: [S_q, D],           label: 'Out',     stage: 16, row: 1, color: '#3498db', dimNames: [lq,'D'],
              desc: 'Attention output.' },
        ],
        ops: [
            { id: 'compress_q',  type: 'compress',   inputs: ['X','W_DQ'],        output: 'c_Q',      label: 'Down-proj',
              desc: `Compress queries: X @ W↓q → c_q [S_q, ${d_q}].` },
            { id: 'compress_kv', type: 'compress',   inputs: ['X','W_DKV'],       output: 'c_KV_new', label: 'Down-proj',
              desc: `Compress KV for S_q new tokens: X @ W↓kv → c_kv_new [S_q, ${d_c}]. This low-rank bottleneck (D=${D} → d_c=${d_c}) is the key to MLA's KV cache efficiency.` },
            { id: 'rope_k_proj', type: 'matmul',     inputs: ['X','W_KR'],        output: 'k_rp_new', label: 'Linear',
              desc: `Project S_q new tokens to RoPE key space: X @ W_kr → k_r_new [S_q, ${dr}]. Decoupled from KV compression because RoPE doesn't commute with low-rank compression.` },
            { id: 'rope_k',      type: 'rope',       inputs: ['k_rp_new'],        output: 'k_r_new',  label: 'RoPE',
              desc: `Apply RoPE to new decoupled keys before caching.` },
            { id: 'cache_kv',    type: 'cache',      inputs: ['c_KV_new'],        output: 'c_KV',     label: 'KV Cache', alignX: 'kv_cache',
              desc: `Append new compressed KV latents to the cache → [S, ${d_c}].` },
            { id: 'cache_kr',    type: 'cache',      inputs: ['k_r_new'],         output: 'k_r',      label: 'KV Cache', alignX: 'kv_cache',
              desc: `Append new RoPE'd keys to the cache → [S, ${dr}]. Total cache per token = d_c + d_r = ${d_c} + ${dr} = ${d_c + dr}.` },
            { id: 'decomp_q',   type: 'decompress', inputs: ['c_Q','W_UQ'],      output: 'q_nope',   label: 'Up-proj Q',
              desc: `Decompress content queries: c_q @ W↑q → [S_q, ${NP}], viewed as [n_h, S_q, P=${d_h}].` },
            { id: 'rope_q_proj', type: 'matmul',     inputs: ['c_Q','W_QR'],     output: 'q_pe_pre', label: 'Linear',
              desc: `Project query latent to RoPE space: c_q @ W_qr → [S_q, ${NR}], viewed as [n_h, S_q, R=${dr}].` },
            { id: 'decomp_k',   type: 'decompress', inputs: ['c_KV','W_UK'],     output: 'k_nope',   label: 'Up-proj K',
              desc: `Decompress content keys from full cache: c_kv @ W↑k → [S, ${NP}], viewed as [n_h, S, P=${d_h}].` },
            { id: 'decomp_v',   type: 'decompress', inputs: ['c_KV','W_UV'],     output: 'V',        label: 'Up-proj V',
              desc: `Decompress values from full cache: c_kv @ W↑v → [S, ${D}], viewed as [n_h, S, V=${d_h}].` },
            { id: 'rope_q',     type: 'rope',       inputs: ['q_pe_pre'],        output: 'q_pe',     label: 'RoPE',
              desc: `Apply RoPE to pre-RoPE query embeddings → [n_h, S_q, R=${dr}].` },
            { id: 'content_qk', type: 'matmul',     inputs: ['q_nope','k_nope'], output: 's_content', label: "q·kᵀ", stage: 8,
              desc: `Content attention: q_nope @ k_nope^T. Inner dimension P=${d_h}.` },
            { id: 'rope_qk',    type: 'matmul',     inputs: ['q_pe','k_r'],      output: 's_rope',   label: "q_r' @ k_r'ᵀ", stage: 8,
              desc: `Positional attention: q_pe' @ k_r'^T. Inner dimension R=${dr}. k_r' is broadcast across n_h heads.` },
            { id: 'add_scores', type: 'add',        inputs: ['s_content','s_rope'], output: 'scores', label: '+',
              desc: `Sum content and positional scores. Equivalent to cat([q_nope, q_pe'], dim=-1) @ cat([k_nope, k_r'], dim=-1)^T with QK headdim = P + R = ${d_h + dr}.` },
            { id: 'masking',    type: 'mask',       inputs: ['scores','mask'],   output: 'attn',     label: 'Mask+Softmax',
              desc: 'Apply causal mask then softmax.' },
            { id: 'attn_v',     type: 'matmul',     inputs: ['attn','V'],        output: 'attn_head', label: 'Attn @ V', stage: 12,
              desc: `Weighted sum of values per head → [n_h, S_q, V=${d_h}]. V headdim = d_h = ${d_h}.` },
            { id: 'view_out',   type: 'reshape',    inputs: ['attn_head'],       output: 'ctx',      label: 'View',
              desc: `View heads as flat: [n_h, S_q, V] → [S_q, D]. Merges n_h=${n_h} × d_h=${d_h} = D=${D}. Zero-cost metadata operation.` },
            { id: 'out_proj',   type: 'matmul',     inputs: ['ctx','W_O'],       output: 'out',      label: 'Linear',
              desc: `Output projection: [S_q, D] @ [D, D] → [S_q, D] via Wo. This mixes information across heads.` },
        ],
        groups: [
            { label: 'KV PROJECTION & CACHE', color: '#16a085',
              tensors: ['W_DKV','W_KR','c_KV_new','k_rp_new','k_r_new','c_KV','k_r','W_UK','W_UV','k_nope','V'],
              ops: ['compress_kv','rope_k_proj','rope_k','cache_kv','cache_kr','decomp_k','decomp_v'],
              desc: `MLA compresses KV via a low-rank bottleneck: X @ W↓kv → c_kv (d_c=${d_c}). Only c_kv and a decoupled RoPE key k_r (d_r=${dr}) are cached — total ${d_c + dr} per token vs ${2*n_h*d_h} for MHA. During prefill, c_kv is decompressed back to full K,V via W↑k, W↑v, then reshaped into per-head form. RoPE is decoupled because it doesn't commute with low-rank compression.` },
            { label: 'ATTENTION', color: '#9b59b6', padTop: 40,
              tensors: ['s_content','s_rope','scores','mask','attn'],
              ops: ['content_qk','rope_qk','add_scores','masking','attn_v'],
              desc: `The attention backend receives q_nope, q_pe', k_nope, k_r', V. Scores = q_nope @ k_nope^T + q_pe' @ k_r'^T, equivalent to SDPA with QK headdim = P + R = ${d_h + dr}, V headdim = ${d_h}. Causal mask + softmax, then Attn @ V → per-head context vectors.` },
            { label: 'OUTPUT PROJECTION', color: '#e67e22', padTop: 40,
              tensors: ['attn_head','ctx','W_O','out'],
              ops: ['view_out','out_proj'],
              desc: `Context vectors merged across heads via view and projected through Wo back to model dimension D.` },
        ]
    };
}

// --- MLA (Absorbed / inference view) ---

export function mlaAbsorbedGraph(p) {
    const { n_h, d_h, d_c, d_r, B } = p;
    const d_q = p.d_q || d_c;   // Q latent dim (Lq); distinct from KV latent dim (Lkv = d_c)
    const S_q = p.sumSq || p.S_q || p.S;
    const S   = p.sumS  || p.S;
    const D = n_h * d_h;
    const dr = d_r || 64;
    const lq = sqLabel(B), ls = sLabel(B);
    return {
        id: 'mla_absorbed', label: 'MLA — MQA-style (Absorbed)',
        tensors: [
            { id: 'X',        shape: [S_q, D],           label: 'X',        stage: 0, row: 1, color: '#4a90d9', dimNames: [lq,'D'],
              desc: 'Input activation tensor.' },
            // Compression
            { id: 'W_DQ',     shape: [D, d_q],             label: 'W↓q',     stage: 1, row: 0, color: '#7b68ee', dimNames: ['D','d_q'], type: 'weight',
              desc: `Query down-projection: D=${D} → d_q=${d_q}.` },
            { id: 'W_DKV',    shape: [D, d_c],             label: 'W↓kv',    stage: 1, row: 1, color: '#7b68ee', dimNames: ['D','d_c'], type: 'weight',
              desc: `KV down-projection: D=${D} → d_c=${d_c}.` },
            // Decoupled RoPE key projection
            { id: 'W_KR',     shape: [D, dr],               label: 'W_kr',    stage: 1, row: 2, color: '#7b68ee', dimNames: ['D','d_r'], type: 'weight',
              desc: `RoPE key projection: D=${D} → d_r=${dr}. Decoupled from KV compression because RoPE doesn't commute with low-rank compression.` },
            { id: 'c_Q',       shape: [S_q, d_q],         label: 'c_q',     stage: 2, row: 0, color: '#c0392b', dimNames: [lq,'d_q'],
              desc: `Compressed query latent. d_q=${d_q} (separate from KV latent d_c=${d_c}).` },
            { id: 'c_KV_new',  shape: [S_q, d_c],         label: 'c_kv_new', stage: 2, row: 1, color: '#e67e22', dimNames: [lq,'d_c'],
              desc: `Newly compressed KV latent for S_q tokens.` },
            { id: 'k_rp_new',  shape: [S_q, dr],          label: 'k_r_new', stage: 2, row: 2, color: '#ff7043', dimNames: [lq,'d_r'],
              desc: `Newly projected pre-RoPE key embedding for S_q tokens.` },
            { id: 'k_r_new',   shape: [S_q, dr],          label: "k_r_new'", stage: 3, row: 2, color: '#ff7043', dimNames: [lq,'d_r'],
              desc: `RoPE applied to new decoupled keys before caching.` },
            // Absorbed weights
            { id: 'W_QK',     shape: [n_h, d_q, d_c],      label: 'W_QK',    stage: 4, row: 1, color: '#7b68ee', dimNames: ['n_h','d_q','d_c'], type: 'weight', badge: 'ABSORBED',
              desc: `Absorbed QK content weight: W_QK_h = W_UQ_h @ W_UK_h^T per head, shape [d_q, d_c] = [${d_q}, ${d_c}]. Maps query latent to KV latent space for content-based attention.` },
            { id: 'W_QR',     shape: [n_h, d_q, dr],        label: 'W_qr',    stage: 4, row: 0, color: '#7b68ee', dimNames: ['n_h','d_q','d_r'], type: 'weight', badge: 'ABSORBED',
              desc: `Absorbed RoPE query weight: extracts the RoPE query component from c_q, per head. Shape [d_q, d_r] = [${d_q}, ${dr}] per head.` },
            { id: 'c_KV',     shape: [S, d_c],           label: 'c_kv',    stage: 4, row: 4, color: '#e67e22', dimNames: [ls,'d_c'], cache: true,
              desc: `Full compressed KV cache: new latents appended → S total. Total cache per token = d_c + d_r = ${d_c} + ${dr} = ${d_c + dr}.` },
            { id: 'k_r',      shape: [S, dr],            label: "k_r'",    stage: 4, row: 5, color: '#ff7043', dimNames: [ls,'d_r'], cache: true,
              desc: `Full RoPE key cache: new RoPE'd keys appended → S total. Provides positional information that c_kv cannot carry.` },
            // Projected queries
            { id: 'q_rp',     shape: [n_h, S_q, dr],     label: 'q_r',     stage: 6, row: 0, color: '#ff7043', dimNames: ['n_h',lq,'d_r'],
              desc: `Pre-RoPE query component: c_q @ W_qr per head → [n_h, S_q, d_r=${dr}].` },
            { id: 'q_lat',    shape: [n_h, S_q, d_c],    label: "q'",      stage: 7, row: 1, color: '#e74c3c', dimNames: ['n_h',lq,'d_c'],
              desc: `Absorbed content query: c_q @ W_QK per head → [n_h, S_q, d_c=${d_c}]. Maps from d_q=${d_q} query latent to d_c=${d_c} KV latent space.` },
            { id: 'q_r',      shape: [n_h, S_q, dr],     label: "q_r'",    stage: 7, row: 0, color: '#ff7043', dimNames: ['n_h',lq,'d_r'],
              desc: `RoPE query — after applying rotary position embedding to q_r.` },
            // Scores (split into content + RoPE + add)
            { id: 's_content', shape: [n_h, S_q, S],     label: "q'c_kvᵀ", stage: 9, row: 1, color: '#9b59b6', dimNames: ['n_h',lq,ls],
              desc: `Content attention scores: q' @ c_kv^T. Computed entirely in d_c=${d_c} latent space — no expansion to d_h needed.` },
            { id: 's_rope',   shape: [n_h, S_q, S],      label: "q_r'k_r'ᵀ", stage: 9, row: 2, color: '#9b59b6', dimNames: ['n_h',lq,ls],
              desc: `RoPE attention scores: q_r' @ k_r'^T. Position-based attention via decoupled RoPE in d_r=${dr} space.` },
            { id: 'scores',   shape: [n_h, S_q, S],      label: 'Scores',  stage: 10, row: 1, color: '#9b59b6', dimNames: ['n_h',lq,ls],
              desc: `Combined scores = (q' @ c_kv^T) + (q_r' @ k_r'^T). Equivalent to MQA with QK headdim = d_c + d_r = ${d_c + dr}, V headdim = d_c = ${d_c}.` },
            { id: 'mask',     shape: [S_q, S],               label: 'Mask',    stage: 10, row: 2, color: '#1abc9c', dimNames: [lq,ls], type: 'mask',
              desc: 'Causal mask.' },
            { id: 'attn',     shape: [n_h, S_q, S],      label: 'Attn',    stage: 11, row: 1, color: '#9b59b6', dimNames: ['n_h',lq,ls],
              desc: 'Attention weights.' },
            // Latent context: attn @ c_kv (still in latent space)
            { id: 'c_ctx',    shape: [n_h, S_q, d_c],    label: 'c_ctx',   stage: 13, row: 1, color: '#e67e22', dimNames: ['n_h',lq,'d_c'], badge: 'LATENT',
              desc: `Latent context: Attn @ c_kv. Attention-weighted sum of compressed values — still in d_c=${d_c} latent space.` },
            // Value decompression at the end
            { id: 'W_UV',     shape: [d_c, d_h],            label: 'W↑v',     stage: 13, row: 2, color: '#7b68ee', dimNames: ['d_c','d_h'], type: 'weight',
              desc: `Value up-projection (per head): d_c=${d_c} → d_h=${d_h}. Applied once per head to c_ctx — not per cached token. This is why absorbed MLA is efficient.` },
            { id: 'ctx_head', shape: [n_h, S_q, d_h],    label: 'AV_head', stage: 14, row: 1, color: '#e67e22', dimNames: ['n_h',lq,'d_h'],
              desc: `Per-head decompressed context: c_ctx @ W↑v → [n_h, S_q, d_h].` },
            { id: 'ctx',      shape: [S_q, D],            label: 'Ctx',     stage: 15, row: 1, color: '#e67e22', dimNames: [lq,'D'],
              desc: 'Context vectors merged across heads via view.' },
            { id: 'W_O',      shape: [D, D],                label: 'Wo',      stage: 15, row: 2, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: 'Output projection.' },
            { id: 'out',      shape: [S_q, D],            label: 'Out',     stage: 16, row: 1, color: '#3498db', dimNames: [lq,'D'],
              desc: 'Attention output.' },
        ],
        ops: [
            { id: 'compress_q',    type: 'compress',   inputs: ['X','W_DQ'],          output: 'c_Q',      label: 'Down-proj',
              desc: `Compress queries: X @ W↓q → c_q [S_q, ${d_q}].` },
            { id: 'compress_kv',   type: 'compress',   inputs: ['X','W_DKV'],         output: 'c_KV_new', label: 'Down-proj',
              desc: `Compress KV for S_q new tokens: X @ W↓kv → c_kv_new [S_q, ${d_c}].` },
            { id: 'rope_k_proj',   type: 'matmul',     inputs: ['X','W_KR'],          output: 'k_rp_new', label: 'Linear',
              desc: `Project S_q new tokens to decoupled RoPE key space: X @ W_kr → k_r_new [S_q, ${dr}].` },
            { id: 'rope_k',        type: 'rope',       inputs: ['k_rp_new'],          output: 'k_r_new',  label: 'RoPE',
              desc: `Apply RoPE to new decoupled keys before caching.` },
            { id: 'cache_kv',      type: 'cache',      inputs: ['c_KV_new'],          output: 'c_KV',     label: 'KV Cache', alignX: 'kv_cache',
              desc: `Append new compressed KV latents to the cache → [S, ${d_c}].` },
            { id: 'cache_kr',      type: 'cache',      inputs: ['k_r_new'],           output: 'k_r',      label: 'KV Cache', alignX: 'kv_cache',
              desc: `Append new RoPE'd keys to the cache → [S, ${dr}]. Total cache per token = d_c + d_r = ${d_c + dr}.` },
            { id: 'absorbed_proj', type: 'matmul',     inputs: ['c_Q','W_QK'],        output: 'q_lat',    label: 'Absorbed proj',
              desc: `Content query in latent space: c_q @ W_QK per head → q' [n_h, S_q, d_c=${d_c}]. W_QK maps d_q=${d_q} → d_c=${d_c}, replacing separate Q/K up-projections.` },
            { id: 'rope_q_proj',   type: 'matmul',     inputs: ['c_Q','W_QR'],        output: 'q_rp',     label: 'Linear',
              desc: `Project c_q to RoPE query space per head: c_q @ W_qr → q_r [n_h, S_q, ${dr}]. W_qr maps d_q=${d_q} → d_r=${dr} per head.` },
            { id: 'rope_q',        type: 'rope',       inputs: ['q_rp'],              output: 'q_r',      label: 'RoPE',
              desc: `Apply RoPE to query. Together with k_r', enables position-dependent attention scores.` },
            { id: 'content_qk',   type: 'matmul',     inputs: ['q_lat','c_KV'],      output: 's_content', label: "q' @ c_kvᵀ", stage: 8,
              routeBelow: ['c_KV'],
              desc: `Content attention scores: q' @ c_kv^T → [n_h, S_q, S]. Computed in d_c=${d_c} latent space.` },
            { id: 'rope_qk',      type: 'matmul',     inputs: ['q_r','k_r'],         output: 's_rope',   label: "q_r' @ k_r'ᵀ", stage: 8,
              desc: `Positional attention scores: q_r' @ k_r'^T → [n_h, S_q, S]. Decoupled RoPE in d_r=${dr} space.` },
            { id: 'add_scores',   type: 'add',        inputs: ['s_content','s_rope'], output: 'scores',   label: '+',
              desc: `Sum content and positional scores.` },
            { id: 'masking',      type: 'mask',       inputs: ['scores','mask'],     output: 'attn',     label: 'Mask+Softmax',
              desc: 'Apply causal mask then softmax.' },
            { id: 'latent_attn_v',type: 'matmul',     inputs: ['attn','c_KV'],       output: 'c_ctx',    label: 'Attn @ c_kv', stage: 12,
              routeBelow: ['c_KV'],
              desc: `Latent attention: Attn @ c_kv → c_ctx [n_h, S_q, d_c]. Weighted sum in latent space — V headdim = d_c = ${d_c}.` },
            { id: 'decomp_ctx',   type: 'decompress', inputs: ['c_ctx','W_UV'],      output: 'ctx_head', label: 'Up-proj V',
              desc: `Decompress context: c_ctx @ W↑v → ctx_head [n_h, S_q, d_h]. Decompression only at the end.` },
            { id: 'view_out',     type: 'reshape',    inputs: ['ctx_head'],          output: 'ctx',      label: 'View',
              desc: `View heads as flat: [n_h, S_q, d_h] → [S_q, D]. Zero-cost metadata operation.` },
            { id: 'out_proj',     type: 'matmul',     inputs: ['ctx','W_O'],         output: 'out',      label: 'Linear',
              desc: `Output projection: [S_q, D] @ [D, D] → [S_q, D] via Wo. This mixes information across heads.` },
        ],
        groups: [
            { label: 'KV PROJECTION & CACHE', color: '#16a085',
              tensors: ['W_DKV','W_KR','c_KV_new','k_rp_new','k_r_new','c_KV','k_r'],
              ops: ['compress_kv','rope_k_proj','rope_k','cache_kv','cache_kr'],
              desc: `Same compression as the prefill path: X @ W↓kv → c_kv (d_c=${d_c}), plus decoupled RoPE keys k_r (d_r=${dr}). Cache per token = ${d_c + dr}. The key difference in the absorbed path is that c_kv is NOT decompressed — attention operates directly in latent space.` },
            { label: 'ATTENTION', color: '#9b59b6', padTop: 40,
              tensors: ['s_content','s_rope','scores','mask','attn'],
              ops: ['content_qk','rope_qk','add_scores','masking','latent_attn_v'],
              desc: `The attention backend receives q' (in d_c=${d_c} latent space), q_r', c_kv, k_r'. Scores = q' @ c_kv^T + q_r' @ k_r'^T, equivalent to MQA with QK headdim = d_c + d_r = ${d_c + dr}, V headdim = d_c = ${d_c}. Causal mask + softmax, then Attn @ c_kv → c_ctx (still in latent space).` },
            { label: 'OUTPUT PROJECTION', color: '#e67e22', padTop: 40,
              tensors: ['c_ctx','W_UV','ctx_head','ctx','W_O','out'],
              ops: ['decomp_ctx','view_out','out_proj'],
              desc: `Latent context c_ctx is decompressed via W↑v (d_c=${d_c} → d_h=${d_h}) once per query position — not per cached token. Then merged across heads and projected through Wo back to model dimension D.` },
        ]
    };
}

// --- DSA (DeepSeek Sparse Attention, V3.2-style top-k) ---

export function dsaGraph(p) {
    const { n_h, d_h, d_c, d_r, B } = p;
    const d_q = p.d_q || d_c;   // Q latent dim (Lq); distinct from KV latent dim (Lkv = d_c)
    const n_i = p.n_i || 64;    // indexer heads
    const d_i = p.d_i || 128;   // indexer head dim
    const topk = p.topk || 2048;
    const S_q = p.sumSq || p.S_q || p.S;
    const S   = p.sumS  || p.S;
    const D = n_h * d_h;
    const dr = d_r || 64;
    const lq = sqLabel(B), ls = sLabel(B);

    // Effective k: each query attends to min(topk, S_i) tokens. With B > 1 this
    // varies per request, so use the query-weighted mean so volumes stay exact.
    let k;
    if (B > 1 && p.queryLens && p.seqLens) {
        const sq = p.queryLens.slice(0, B), s = p.seqLens.slice(0, B);
        const totQ = sq.reduce((a, b) => a + b, 0);
        k = Math.round(sq.reduce((acc, q, i) => acc + q * Math.min(topk, s[i]), 0) / Math.max(1, totQ));
    } else {
        k = Math.min(topk, p.S);
    }
    const dense = k >= S;
    const denseNote = dense
        ? ` Here k ≥ S, so every causal position is selected — dense fallback (no savings until S > topk).`
        : ``;
    const kNote = B > 1 ? ` (query-weighted mean of per-request min(topk, S_i))` : ``;

    return {
        id: 'dsa', label: 'DeepSeek Sparse Attention (DSA)',
        tensors: [
            { id: 'X',        shape: [S_q, D],           label: 'X',        stage: 0, row: 1, color: '#4a90d9', dimNames: [lq,'D'],
              desc: 'Input activation tensor.' },
            // Compression (identical to absorbed MLA)
            { id: 'W_DQ',     shape: [D, d_q],             label: 'W↓q',     stage: 1, row: 0, color: '#7b68ee', dimNames: ['D','d_q'], type: 'weight',
              desc: `Query down-projection: D=${D} → d_q=${d_q}.` },
            { id: 'W_DKV',    shape: [D, d_c],             label: 'W↓kv',    stage: 1, row: 1, color: '#7b68ee', dimNames: ['D','d_c'], type: 'weight',
              desc: `KV down-projection: D=${D} → d_c=${d_c}.` },
            { id: 'W_KR',     shape: [D, dr],               label: 'W_kr',    stage: 1, row: 2, color: '#7b68ee', dimNames: ['D','d_r'], type: 'weight',
              desc: `RoPE key projection: D=${D} → d_r=${dr}. Decoupled from KV compression because RoPE doesn't commute with low-rank compression.` },
            { id: 'c_Q',       shape: [S_q, d_q],         label: 'c_q',     stage: 2, row: 0, color: '#c0392b', dimNames: [lq,'d_q'],
              desc: `Compressed query latent. d_q=${d_q}. Feeds both the absorbed attention path and the lightning indexer.` },
            { id: 'c_KV_new',  shape: [S_q, d_c],         label: 'c_kv_new', stage: 2, row: 1, color: '#e67e22', dimNames: [lq,'d_c'],
              desc: `Newly compressed KV latent for S_q tokens.` },
            { id: 'k_rp_new',  shape: [S_q, dr],          label: 'k_r_new', stage: 2, row: 2, color: '#ff7043', dimNames: [lq,'d_r'],
              desc: `Newly projected pre-RoPE key embedding for S_q tokens.` },
            { id: 'k_r_new',   shape: [S_q, dr],          label: "k_r_new'", stage: 3, row: 2, color: '#ff7043', dimNames: [lq,'d_r'],
              desc: `RoPE applied to new decoupled keys before caching.` },
            // Absorbed weights (identical to absorbed MLA)
            { id: 'W_QR',     shape: [n_h, d_q, dr],        label: 'W_qr',    stage: 4, row: 0, color: '#7b68ee', dimNames: ['n_h','d_q','d_r'], type: 'weight', badge: 'ABSORBED',
              desc: `Absorbed RoPE query weight: extracts the RoPE query component from c_q, per head. Shape [d_q, d_r] = [${d_q}, ${dr}] per head.` },
            { id: 'W_QK',     shape: [n_h, d_q, d_c],      label: 'W_QK',    stage: 4, row: 1, color: '#7b68ee', dimNames: ['n_h','d_q','d_c'], type: 'weight', badge: 'ABSORBED',
              desc: `Absorbed QK content weight: W_QK_h = W_UQ_h @ W_UK_h^T per head, shape [d_q, d_c] = [${d_q}, ${d_c}]. Maps query latent to KV latent space for content-based attention.` },
            { id: 'c_KV',     shape: [S, d_c],           label: 'c_kv',    stage: 4, row: 4, color: '#e67e22', dimNames: [ls,'d_c'], cache: true,
              desc: `Full compressed KV cache: new latents appended → S total. Cache per token = d_c + d_r + d_i (FP8 indexer key) = ${d_c}·2B + ${dr}·2B + ${d_i}·1B.` },
            { id: 'k_r',      shape: [S, dr],            label: "k_r'",    stage: 4, row: 5, color: '#ff7043', dimNames: [ls,'d_r'], cache: true,
              desc: `Full RoPE key cache: new RoPE'd keys appended → S total. Provides positional information that c_kv cannot carry.` },
            // Projected queries (identical to absorbed MLA, stages 6-7)
            { id: 'q_rp',     shape: [n_h, S_q, dr],     label: 'q_r',     stage: 6, row: 0, color: '#ff7043', dimNames: ['n_h',lq,'d_r'],
              desc: `Pre-RoPE query component: c_q @ W_qr per head → [n_h, S_q, d_r=${dr}].` },
            { id: 'q_r',      shape: [n_h, S_q, dr],     label: "q_r'",    stage: 7, row: 0, color: '#ff7043', dimNames: ['n_h',lq,'d_r'],
              desc: `RoPE query — after applying rotary position embedding to q_r.` },
            { id: 'q_lat',    shape: [n_h, S_q, d_c],    label: "q'",      stage: 7, row: 1, color: '#e74c3c', dimNames: ['n_h',lq,'d_c'],
              desc: `Absorbed content query: c_q @ W_QK per head → [n_h, S_q, d_c=${d_c}].` },
            // --- Lightning indexer (rows 6-8) ---
            { id: 'W_KI',     shape: [D, d_i],             label: 'W_kI',    stage: 1, row: 7, color: '#7b68ee', dimNames: ['D','d_i'], type: 'weight',
              desc: `Indexer key projection: D=${D} → d_i=${d_i}. A single shared indexer key per token (MQA-style — no head dimension). In vLLM this is fused with W_w into one GEMM, with a LayerNorm on the output.` },
            { id: 'W_W',      shape: [D, n_i],             label: 'W_w',     stage: 1, row: 8, color: '#7b68ee', dimNames: ['D','n_i'], type: 'weight',
              desc: `Indexer head-weight projection: D=${D} → n_i=${n_i}. Produces per-token weights that say how much each indexer head's opinion counts.` },
            { id: 'k_I_new',  shape: [S_q, d_i],         label: 'k_I_new', stage: 2, row: 7, color: '#2ecc71', dimNames: [lq,'d_i'],
              desc: `New indexer keys for S_q tokens: X @ W_kI → [S_q, ${d_i}]. One shared key per token — all n_i indexer heads score against the same key (MQA-style).` },
            { id: 'w_I',      shape: [S_q, n_i],         label: 'w',       stage: 2, row: 8, color: '#f1c40f', dimNames: [lq,'n_i'],
              desc: `Per-token indexer head weights: X @ W_w → [S_q, ${n_i}]. Weight w[t,h] scales head h's contribution to query t's index scores.` },
            { id: 'k_I_r',    shape: [S_q, d_i],         label: "k_I'",    stage: 3, row: 7, color: '#2ecc71', dimNames: [lq,'d_i'], bytesPerEl: 1, badge: 'FP8',
              desc: `Indexer keys after partial RoPE (first ${dr} dims rotated) and FP8 quantization. FP8 is what makes the indexer "lightning" — half the bytes of BF16.` },
            { id: 'W_QI',     shape: [d_q, n_i * d_i],     label: 'W_qI',    stage: 4, row: 6, color: '#7b68ee', dimNames: ['d_q','n_i·d_i'], type: 'weight',
              desc: `Indexer query projection: d_q=${d_q} → n_i·d_i=${n_i * d_i}. Projects the compressed query latent c_q into ${n_i} small indexer heads.` },
            { id: 'k_I',      shape: [S, d_i],           label: 'k_I',     stage: 4, row: 7, color: '#2ecc71', dimNames: [ls,'d_i'], cache: true, bytesPerEl: 1, badge: 'FP8',
              desc: `Indexer key cache (FP8): new indexer keys appended → S total. Adds only d_i=${d_i} bytes per token on top of the MLA cache.` },
            { id: 'q_I_pre',  shape: [n_i, S_q, d_i],    label: 'q_I',     stage: 6, row: 6, color: '#e74c3c', dimNames: ['n_i',lq,'d_i'],
              desc: `Indexer queries: c_q @ W_qI → [S_q, ${n_i * d_i}], viewed as [n_i, S_q, d_i]. ${n_i} small heads of ${d_i} dims each.` },
            { id: 'q_I',      shape: [n_i, S_q, d_i],    label: "q_I'",    stage: 7, row: 6, color: '#e74c3c', dimNames: ['n_i',lq,'d_i'], bytesPerEl: 1, badge: 'FP8',
              desc: `Indexer queries after partial RoPE (first ${dr} dims) and FP8 quantization.` },
            { id: 'idx_scores', shape: [n_i, S_q, S],    label: 'q_I·k_Iᵀ', stage: 9, row: 6, color: '#9b59b6', dimNames: ['n_i',lq,ls],
              desc: `Per-head index scores: q_I' @ k_I^T → [n_i, S_q, S]. k_I is broadcast across all ${n_i} indexer heads (MQA-style). FP8 inputs make this GEMM cheap despite being dense over all S positions.` },
            { id: 'logits',   shape: [S_q, S],           label: 'I',       stage: 10, row: 6, color: '#9b59b6', dimNames: [lq,ls],
              desc: `Index scores: I[t,s] = Σ_h w[t,h] · ReLU(q_I'[t,h]·k_I[s]). One relevance score per (query, cached token) pair — the indexer's judgment of which context matters for each query.` },
            { id: 'mask',     shape: [S_q, S],               label: 'Mask',    stage: 10, row: 7, color: '#1abc9c', dimNames: [lq,ls], type: 'mask',
              desc: 'Causal mask, applied to the index scores: a query may only select positions ≤ its own. Causality is enforced HERE, at selection — core attention needs no mask.' },
            { id: 'topk_idx', shape: [S_q, k],           label: 'top-k idx', stage: 11, row: 6, color: '#f1c40f', dimNames: [lq,'k'], bytesPerEl: 4, badge: 'INT32',
              desc: `Selected token indices: for each query, the k=${k} highest-scoring causal positions${kNote}. int32 indices, not values — this is all that flows to core attention.${denseNote}` },
            { id: 'c_kv_sel', shape: [S_q, k, d_c],      label: 'c_kv_sel', stage: 12, row: 4, color: '#e67e22', dimNames: [lq,'k','d_c'], sramOnly: true, badge: 'GATHERED',
              desc: `Per-query gathered KV latents: each query's k=${k} selected rows of c_kv. Never materialized in HBM — the sparse kernel gathers tiles directly into SRAM (gather op accounts the HBM reads).` },
            { id: 'k_r_sel',  shape: [S_q, k, dr],       label: 'k_r_sel', stage: 12, row: 5, color: '#ff7043', dimNames: [lq,'k','d_r'], sramOnly: true, badge: 'GATHERED',
              desc: `Per-query gathered RoPE keys: each query's k=${k} selected rows of k_r'. Gathered into SRAM alongside c_kv_sel.` },
            // Sparse core attention — over k selected tokens, not S
            { id: 's_content', shape: [n_h, S_q, k],     label: "q'c_kvᵀ", stage: 14, row: 1, color: '#9b59b6', dimNames: ['n_h',lq,'k'],
              desc: `Content attention scores over the k=${k} selected tokens only: q' @ c_kv_sel^T. Width k instead of S — this is where DSA's savings land.` },
            { id: 's_rope',   shape: [n_h, S_q, k],      label: "q_r'k_r'ᵀ", stage: 14, row: 2, color: '#9b59b6', dimNames: ['n_h',lq,'k'],
              desc: `RoPE attention scores over the selected tokens: q_r' @ k_r_sel^T.` },
            { id: 'scores',   shape: [n_h, S_q, k],      label: 'Scores',  stage: 15, row: 1, color: '#9b59b6', dimNames: ['n_h',lq,'k'],
              desc: `Combined scores = content + positional, over k=${k} selected tokens. Equivalent to MQA with QK headdim = d_c + d_r = ${d_c + dr}, but only ${k} columns instead of ${S}.` },
            { id: 'attn',     shape: [n_h, S_q, k],      label: 'Attn',    stage: 16, row: 1, color: '#9b59b6', dimNames: ['n_h',lq,'k'],
              desc: 'Attention weights after softmax. No causal mask needed — the top-k selection already excluded future positions.' },
            { id: 'c_ctx',    shape: [n_h, S_q, d_c],    label: 'c_ctx',   stage: 18, row: 1, color: '#e67e22', dimNames: ['n_h',lq,'d_c'], badge: 'LATENT',
              desc: `Latent context: Attn @ c_kv_sel. Attention-weighted sum of the selected compressed values — still in d_c=${d_c} latent space.` },
            { id: 'W_UV',     shape: [d_c, d_h],            label: 'W↑v',     stage: 18, row: 2, color: '#7b68ee', dimNames: ['d_c','d_h'], type: 'weight',
              desc: `Value up-projection (per head): d_c=${d_c} → d_h=${d_h}. Applied once per query position — not per cached token.` },
            { id: 'ctx_head', shape: [n_h, S_q, d_h],    label: 'AV_head', stage: 19, row: 1, color: '#e67e22', dimNames: ['n_h',lq,'d_h'],
              desc: `Per-head decompressed context: c_ctx @ W↑v → [n_h, S_q, d_h].` },
            { id: 'ctx',      shape: [S_q, D],            label: 'Ctx',     stage: 20, row: 1, color: '#e67e22', dimNames: [lq,'D'],
              desc: 'Context vectors merged across heads via view.' },
            { id: 'W_O',      shape: [D, D],                label: 'Wo',      stage: 20, row: 2, color: '#7b68ee', dimNames: ['D','D'], type: 'weight',
              desc: 'Output projection.' },
            { id: 'out',      shape: [S_q, D],            label: 'Out',     stage: 21, row: 1, color: '#3498db', dimNames: [lq,'D'],
              desc: 'Attention output.' },
        ],
        ops: [
            { id: 'compress_q',    type: 'compress',   inputs: ['X','W_DQ'],          output: 'c_Q',      label: 'Down-proj',
              desc: `Compress queries: X @ W↓q → c_q [S_q, ${d_q}].` },
            { id: 'compress_kv',   type: 'compress',   inputs: ['X','W_DKV'],         output: 'c_KV_new', label: 'Down-proj',
              desc: `Compress KV for S_q new tokens: X @ W↓kv → c_kv_new [S_q, ${d_c}].` },
            { id: 'rope_k_proj',   type: 'matmul',     inputs: ['X','W_KR'],          output: 'k_rp_new', label: 'Linear',
              desc: `Project S_q new tokens to decoupled RoPE key space: X @ W_kr → k_r_new [S_q, ${dr}].` },
            { id: 'rope_k',        type: 'rope',       inputs: ['k_rp_new'],          output: 'k_r_new',  label: 'RoPE',
              desc: `Apply RoPE to new decoupled keys before caching.` },
            { id: 'cache_kv',      type: 'cache',      inputs: ['c_KV_new'],          output: 'c_KV',     label: 'KV Cache', alignX: 'kv_cache',
              desc: `Append new compressed KV latents to the cache → [S, ${d_c}].` },
            { id: 'cache_kr',      type: 'cache',      inputs: ['k_r_new'],           output: 'k_r',      label: 'KV Cache', alignX: 'kv_cache',
              desc: `Append new RoPE'd keys to the cache → [S, ${dr}].` },
            { id: 'absorbed_proj', type: 'matmul',     inputs: ['c_Q','W_QK'],        output: 'q_lat',    label: 'Absorbed proj',
              desc: `Content query in latent space: c_q @ W_QK per head → q' [n_h, S_q, d_c=${d_c}].` },
            { id: 'rope_q_proj',   type: 'matmul',     inputs: ['c_Q','W_QR'],        output: 'q_rp',     label: 'Linear',
              desc: `Project c_q to RoPE query space per head: c_q @ W_qr → q_r [n_h, S_q, ${dr}].` },
            { id: 'rope_q',        type: 'rope',       inputs: ['q_rp'],              output: 'q_r',      label: 'RoPE',
              desc: `Apply RoPE to query. Together with k_r', enables position-dependent attention scores.` },
            // Lightning indexer
            { id: 'idx_k_proj',    type: 'matmul',     inputs: ['X','W_KI'],          output: 'k_I_new',  label: 'Linear',
              desc: `Project new tokens to indexer key space: X @ W_kI → [S_q, ${d_i}]. Single shared key per token (no head dim). In vLLM this GEMM is fused with the head-weight projection; a LayerNorm follows.` },
            { id: 'idx_w_proj',    type: 'matmul',     inputs: ['X','W_W'],           output: 'w_I',      label: 'Linear',
              desc: `Project new tokens to indexer head weights: X @ W_w → [S_q, ${n_i}].` },
            { id: 'rope_ki',       type: 'rope',       inputs: ['k_I_new'],           output: 'k_I_r',    label: 'RoPE+FP8',
              desc: `Partial RoPE (first ${dr} of ${d_i} dims) then FP8 quantization of indexer keys before caching.` },
            { id: 'cache_ki',      type: 'cache',      inputs: ['k_I_r'],             output: 'k_I',      label: 'Idx Cache', alignX: 'kv_cache',
              desc: `Append new FP8 indexer keys to the indexer cache → [S, ${d_i}]. Costs 1 byte/element — d_i=${d_i} bytes per token.` },
            { id: 'idx_q_proj',    type: 'matmul',     inputs: ['c_Q','W_QI'],        output: 'q_I_pre',  label: 'Indexer Q',
              desc: `Indexer queries from the query latent: c_q @ W_qI → [S_q, ${n_i * d_i}], viewed as [n_i, S_q, d_i].` },
            { id: 'rope_qi',       type: 'rope',       inputs: ['q_I_pre'],           output: 'q_I',      label: 'RoPE+FP8',
              desc: `Partial RoPE (first ${dr} dims) then FP8 quantization of indexer queries.` },
            { id: 'idx_qk',        type: 'matmul',     inputs: ['q_I','k_I'],         output: 'idx_scores', label: "q_I' @ k_Iᵀ", stage: 8,
              desc: `Index score GEMM: q_I' @ k_I^T → [n_i, S_q, S]. Dense over all S cached tokens, but cheap per position: n_i·d_i = ${n_i * d_i} MACs vs n_h·(d_c+d_r) = ${n_h * (d_c + dr)} for dense absorbed attention — ~${Math.round(n_h * (d_c + dr) / (n_i * d_i))}× fewer, and in FP8.` },
            { id: 'relu_wsum',     type: 'relu_wsum',  inputs: ['idx_scores','w_I'],  output: 'logits',   label: 'ReLU·w Σ',
              desc: `Combine indexer heads: I[t,s] = Σ_h w[t,h] · ReLU(score[h,t,s]). ReLU gates each head's opinion; w weights how much it counts. Output: one scalar relevance per (query, token) pair.` },
            { id: 'topk',          type: 'topk',       inputs: ['logits','mask'],     output: 'topk_idx', label: `Top-k`,
              desc: `Select each query's k=${k} highest-scoring causal positions (k = min(topk=${topk}, S)). Output is int32 indices only.${denseNote}` },
            { id: 'gather_kv',     type: 'gather',     inputs: ['topk_idx','c_KV'],   output: 'c_kv_sel', label: 'Gather',
              desc: `Gather each query's k=${k} selected c_kv rows from the cache. Reads S_q·k·d_c elements — the no-reuse upper bound; real kernels amortize reads across query tiles. Gathered tiles stay in SRAM.` },
            { id: 'gather_kr',     type: 'gather',     inputs: ['topk_idx','k_r'],    output: 'k_r_sel',  label: 'Gather',
              desc: `Gather each query's k=${k} selected k_r' rows from the cache into SRAM.` },
            // Sparse core attention
            { id: 'content_qk',   type: 'matmul',     inputs: ['q_lat','c_kv_sel'],  output: 's_content', label: "q' @ c_kvᵀ", stage: 13,
              desc: `Content attention scores over selected tokens: q' @ c_kv_sel^T → [n_h, S_q, k]. O(k) per query instead of O(S).` },
            { id: 'rope_qk',      type: 'matmul',     inputs: ['q_r','k_r_sel'],     output: 's_rope',   label: "q_r' @ k_r'ᵀ", stage: 13,
              desc: `Positional attention scores over selected tokens: q_r' @ k_r_sel^T → [n_h, S_q, k].` },
            { id: 'add_scores',   type: 'add',        inputs: ['s_content','s_rope'], output: 'scores',   label: '+',
              desc: `Sum content and positional scores.` },
            { id: 'softmax_op',   type: 'softmax',    inputs: ['scores'],            output: 'attn',     label: 'Softmax',
              desc: `Row-wise softmax over the k=${k} selected positions. No causal mask — top-k selection already excluded future tokens.` },
            { id: 'latent_attn_v',type: 'matmul',     inputs: ['attn','c_kv_sel'],   output: 'c_ctx',    label: 'Attn @ c_kv', stage: 17,
              desc: `Latent attention: Attn @ c_kv_sel → c_ctx [n_h, S_q, d_c]. Weighted sum over only the k=${k} selected tokens, still in latent space.` },
            { id: 'decomp_ctx',   type: 'decompress', inputs: ['c_ctx','W_UV'],      output: 'ctx_head', label: 'Up-proj V',
              desc: `Decompress context: c_ctx @ W↑v → ctx_head [n_h, S_q, d_h]. Decompression only at the end.` },
            { id: 'view_out',     type: 'reshape',    inputs: ['ctx_head'],          output: 'ctx',      label: 'View',
              desc: `View heads as flat: [n_h, S_q, d_h] → [S_q, D]. Zero-cost metadata operation.` },
            { id: 'out_proj',     type: 'matmul',     inputs: ['ctx','W_O'],         output: 'out',      label: 'Linear',
              desc: `Output projection: [S_q, D] @ [D, D] → [S_q, D] via Wo.` },
        ],
        groups: [
            { label: 'KV PROJECTION & CACHE', color: '#16a085',
              tensors: ['W_DKV','W_KR','c_KV_new','k_rp_new','k_r_new','c_KV','k_r'],
              ops: ['compress_kv','rope_k_proj','rope_k','cache_kv','cache_kr'],
              desc: `Identical to absorbed MLA: X @ W↓kv → c_kv (d_c=${d_c}), plus decoupled RoPE keys k_r (d_r=${dr}). DSA adds an FP8 indexer key cache on top — total cache per token = ${d_c}+${dr} BF16 elements + ${d_i} FP8 bytes.` },
            { label: 'LIGHTNING INDEXER', color: '#f1c40f', padTop: 40,
              tensors: ['W_QI','W_KI','W_W','k_I_new','w_I','k_I_r','k_I','q_I_pre','q_I','idx_scores','logits','mask','topk_idx'],
              ops: ['idx_k_proj','idx_w_proj','rope_ki','cache_ki','idx_q_proj','rope_qi','idx_qk','relu_wsum','topk'],
              desc: `The indexer decides WHICH tokens deserve attention. ${n_i} small FP8 heads (d_i=${d_i}) score every cached token against each query — I[t,s] = Σ_h w[t,h]·ReLU(q_I·k_I) — then each query keeps its top k=${topk}. The scoring is dense over S but cheap: FP8, one shared key (MQA-style), and ~${n_i * d_i} dims total vs ${n_h}·${d_c + dr} for full attention. Replicated across TP ranks (not sharded).` },
            { label: 'SPARSE ATTENTION', color: '#9b59b6', padTop: 40,
              tensors: ['c_kv_sel','k_r_sel','s_content','s_rope','scores','attn'],
              ops: ['gather_kv','gather_kr','content_qk','rope_qk','add_scores','softmax_op','latent_attn_v'],
              desc: `Absorbed-MLA attention over only the k=${k} selected tokens — MQA-style with QK headdim d_c + d_r = ${d_c + dr}, V headdim d_c = ${d_c}, but ${dense ? 'k = S here (dense fallback)' : `k=${k} columns instead of S=${S}`}. Runs this way for BOTH prefill and decode (vLLM's FlashMLA-sparse kernel). No causal mask — selection was already causal.` },
            { label: 'OUTPUT PROJECTION', color: '#e67e22', padTop: 40,
              tensors: ['c_ctx','W_UV','ctx_head','ctx','W_O','out'],
              ops: ['decomp_ctx','view_out','out_proj'],
              desc: `Latent context c_ctx is decompressed via W↑v (d_c=${d_c} → d_h=${d_h}) once per query position, then merged across heads and projected through Wo back to model dimension D.` },
        ]
    };
}
