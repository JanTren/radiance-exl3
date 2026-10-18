/* kolibri1_exl3 -- Aleph Alpha's Kolibri 1 with its ROUTED EXPERTS IN EXL3 and every other weight
 * plain bf16: the ("kolibri1", "exl3") architecture plugin.
 *
 * ============================== WHAT IT SERVES ==============================
 *
 * The container Code0000/Kolibri-1-EXL3-K6 publishes (Kolibri-1-EXL3-K6.rad, converted with radiance
 * 1.7.0, `quant` "exl3"): the 57,600 routed expert projections as EXL3 K6 / mul1 trellis codes and
 * the 753 other tensors as the bf16 release (Aleph-Alpha/Kolibri-1-BF16) holds them. The model is
 * the in-tree-fork kolibri1 plugin's (radiance-kolibri1, arch/kolibri1/kolibri1.cpp), whose header
 * says what Kolibri 1 is; this file is that one with three changes:
 *
 *   - the routed block is rad_block_moe_exl3.h's MoeEXL3 (sigmoid router with a selection bias,
 *     `route_sigmoid`, and the ungated shared expert, `sgate_one`) instead of MoeFP8;
 *   - THE TRUNK IS bf16 (Geom::w_bf16): the linears are `gemm_nt` over the bf16 activation, so no
 *     norm quantises -- the pre-norms are `add` + `rmsnorm` (libr4d has no plain `rmsnorm_add`
 *     row), the shared expert's gate is `silu_mul` into its bf16 plane;
 *   - Q, K AND V ARE THREE WEIGHTS in this container (`blk.<l>.attn_q/k/v.weight`), not one fused
 *     `attn_qkv`, and `gemm_nt` writes only contiguous rows. So the three GEMMs write their own
 *     buffers, q and k are copied into the fused row `qk_norm_rope` reads (`cast` writes a column
 *     slice), and kv_store reads k from that row and v from its own buffer.
 *
 * THE DECLARED NAMES ARE THE CONTAINER'S, which are not the fp8 plugin's (`input_layernorm` for
 * `attn_norm`, `self_attn.q_norm` for `attn_q_norm`, `ffn_expert_bias.weight` for
 * `ffn_gate_inp.bias`, ...): a container is read by declared name. The checkpoint names mapped to
 * them are the HF release's, so the same file converts the safetensors export too.
 *
 * ============================== THE LAYER ==============================
 *
 *   x  += pn_prev;  h = rmsnorm(x, input_layernorm)          (layer 0: h = rmsnorm(x))
 *   q, k, v = h @ q^T, h @ k^T, h @ v^T                       three bf16 GEMMs
 *   q, k = qk_norm_rope(q, k)      sliding: rotary_dim 128   full: rotary_dim 0 (norm alone)
 *   o   = attn_paged(q, kv)        sliding: window 513       full: window 0
 *   a   = o @ o_proj^T  [+ all_reduce]
 *   pn  = rmsnorm(a, post_attn_norm)
 *   x  += pn;  h = rmsnorm(x, post_attention_layernorm)
 *   d   = moe(h)        [+ all_reduce inside the block]      MoeEXL3, `route_sigmoid`, ungated shared
 *   pn  = rmsnorm(d, post_ffn_norm)                          added at the next layer's input
 *
 * The KV cache is the fp8 plugin's: the ten global layers in a full paged group, the forty sliding
 * ones in a window group (RADIANCE_KOLIBRI_KV_ONE_GROUP=1: one full group for all fifty).
 *
 * ============================== WHAT IT NEEDS ==============================
 *
 * An engine that has the ops Kolibri adds -- `router_sigmoid_topk`, `qk_norm_rope` at rotary_dim 0,
 * the windowed attention at gqa 12 -- and the Kolibri reply format (rad_kolibri1_chat.h): the
 * radiance-kolibri1 fork of radiance 1.7.0 (branch kolibri1-1.7). Stock radiance 1.7.0 refuses the
 * router op by name at declare.
 */
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <arch/rad_arch.h>
#include <arch/rad_fp8.h>
#include <arch/rad_kolibri1_chat.h>

#include "../qwen4exp_exl3/rad_block_moe_exl3.h"
#include "../qwen4exp_exl3/exl3_stage.h"

namespace kolibri1_exl3 {

using namespace rad::arch;

struct Layer {
    bool full = false;
    rad_kvgroup kv = 0;   /* the layer's paged group: the global one or the sliding one */

    /* attention */
    rad_weight w_in = 0, w_qn = 0, w_kn = 0, w_apost = 0;
    LinearFP8  q{}, k{}, v{}, o{};
    rad_op     op_add_in = 0, op_norm_in = 0, op_qcp = 0, op_kcp = 0, op_qkn = 0, op_qlift = 0,
               op_attn = 0, op_kv_store = 0, op_ar = 0, op_apost = 0;

    /* the routed feed-forward */
    rad_weight w_mlpn = 0, w_fpost = 0;
    rad_op     op_add_mlp = 0, op_norm_mlp = 0, op_fpost = 0;
    MoeEXL3    moe{};
};

struct Model {
    Names   nm{""};
    Geom    g{};
    int64_t window = 0;

    rad_kvgroup kv_g = 0, kv_s = 0;   /* global (full) and sliding (window); one when equal */
    bool        kv_one = false;

    rad_buf b_x = 0, b_pn = 0, b_hout = 0, b_logits = 0;
    ActFP8  a_h{};                    /* the bf16 plane alone (w_bf16) */
    rad_buf b_q = 0, b_k = 0, b_v = 0, b_qkv = 0;
    ActFP8  a_attn{};

    MoeEXL3::Config mc{};
    rad_buf b_rlogits = 0, b_eids = 0, b_ew = 0, b_sorted = 0, b_eoff = 0, b_ecnt = 0;
    rad_buf b_eff = 0, b_edn = 0, b_emoe = 0, b_sgu = 0, b_sout = 0, b_sgate = 0;
    ActFP8  a_sff{};

    rad_weight w_tok = 0, w_out_norm = 0, w_lm_head = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_last_add = 0, op_final_norm = 0, op_gather = 0,
           op_logits = 0;

    std::vector<Layer> layers;
};

static Model g_model[MAX_RANKS];

static int check_format(const RadModelMeta* meta) {
    const char* q = meta->quant ? meta->quant : "";
    if (std::strcmp(q, "exl3") != 0) {
        fprintf(stderr, "radiance: kolibri1_exl3 serves Kolibri 1 with EXL3 routed experts and this "
                        "container declares '%s'; the in-tree kolibri1 plugin serves the block-fp8 "
                        "release\n", *q ? q : "(unquantised)");
        return RAD_E_UNSUPPORTED;
    }
    if (meta->n_expert <= 0) {
        fprintf(stderr, "radiance: kolibri1_exl3 routes every layer and this container declares no "
                        "experts\n");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* "sliding_attention full_attention ..." -> which layers are global. rad-convert flattens a
 * config.json array into a space-separated string. Anything but the two names is refused. */
static int parse_layer_types(const char* s, int64_t n_layer, std::vector<Layer>* layers) {
    if (!s || !*s) {
        fprintf(stderr, "radiance: kolibri1_exl3: the container carries no layer_types; which layers "
                        "slide is not a default\n");
        return RAD_E_FORMAT;
    }
    int64_t l = 0;
    for (const char* p = s; *p;) {
        while (*p == ' ' || *p == ',') ++p;
        if (!*p) break;
        const char* e = p;
        while (*e && *e != ' ' && *e != ',') ++e;
        const std::string t(p, (size_t)(e - p));
        p = e;
        if (l >= n_layer) { ++l; continue; }
        if (t == "full_attention")         (*layers)[(size_t)l].full = true;
        else if (t == "sliding_attention") (*layers)[(size_t)l].full = false;
        else {
            fprintf(stderr, "radiance: kolibri1_exl3: layer %lld is '%s'; this plugin runs "
                            "sliding_attention and full_attention\n", (long long)l, t.c_str());
            return RAD_E_UNSUPPORTED;
        }
        ++l;
    }
    if (l != n_layer) {
        fprintf(stderr, "radiance: kolibri1_exl3: layer_types names %lld layers and the model has "
                        "%lld\n", (long long)l, (long long)n_layer);
        return RAD_E_FORMAT;
    }
    return RAD_OK;
}

/* An RMS norm of one buffer into another, against a gain. */
static rad_op decl_rmsnorm(RadBuilder* b, const Geom& g, rad_weight w, rad_buf x, rad_buf y) {
    return rw(b, RAD_OP(b, "rmsnorm",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                             RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                             RAD_STR("dtype", g.dtype)),
                  RAD_WEIGHTS(w)),
              {x}, {y});
}

/* x += d, in place: the residual add in front of a pre-norm. */
static rad_op decl_add(RadBuilder* b, const Geom& g, rad_buf x, rad_buf d) {
    return rw(b, RAD_OP(b, "add",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                             RAD_STR("dtype", g.dtype)),
                  RAD_NOWEIGHTS),
              {x, d}, {x});
}

/* A strided row copy of `n` columns (libr4d's cast_bf16_bf16 reads both pitches off the operands). */
static rad_op decl_copy(RadBuilder* b, const Geom& g, int64_t n, rad_buf x, rad_buf y) {
    return rw(b, RAD_OP(b, "cast",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", n),
                             RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                  RAD_NOWEIGHTS),
              {x}, {y});
}

static rad_op decl_ar(RadBuilder* b, const Geom& g, rad_buf x) {
    if (g.world <= 1) return 0;
    return rw(b, RAD_OP(b, "all_reduce",
                  RAD_PARAMS(RAD_INT("world_size", g.world),
                             RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                             RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                             RAD_INT("min_bytes", g.wire_min_bytes)),
                  RAD_NOWEIGHTS),
              {x}, {x});
}

/* A per-layer gain vector, declared under the container's name. */
static rad_weight decl_gain(RadBuilder* b, Names& nm, const Geom& g, int l, const char* name,
                            const char* ckpt, uint32_t dtype, int64_t n) {
    if (map_copy(b, name, ckpt) != RAD_OK) return 0;
    (void)nm;
    return decl_w(b, name, dtype, {n}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
}

static int declare_attention(RadBuilder* b, Model& m, Layer& y, int l) {
    Geom&  g  = m.g;
    Names& nm = m.nm;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();
    const char* pre = "model.layers";

    /* THE INPUT NORM: the previous layer's post-normed feed-forward delta is added into the
     * stream first; layer 0 norms the embedding as it is. */
    y.w_in = decl_gain(b, nm, g, l, nm.f("blk.%d.input_layernorm.weight", l),
                       nm.ckpt("%s.%d.input_layernorm.weight", pre, l), RAD_F32, g.n_embd);
    if (!y.w_in) return RAD_E_INVAL;
    if (l > 0) y.op_add_in = decl_add(b, g, m.b_x, m.b_pn);
    y.op_norm_in = decl_rmsnorm(b, g, y.w_in, m.b_x, m.a_h.x);

    /* q, k, v: three weights, each rank its rows (heads) of each. */
    RAD_ARCH_TRY(y.q.declare(b, nm, g, nm.f("blk.%d.attn_q", l), q_dim, g.n_embd, RAD_SHARD_ROW,
                             grp_layer(l), { nm.ckpt("%s.%d.self_attn.q_proj.weight", pre, l) },
                             m.a_h, m.b_q));
    RAD_ARCH_TRY(y.k.declare(b, nm, g, nm.f("blk.%d.attn_k", l), kv_dim, g.n_embd, RAD_SHARD_ROW,
                             grp_layer(l), { nm.ckpt("%s.%d.self_attn.k_proj.weight", pre, l) },
                             m.a_h, m.b_k));
    RAD_ARCH_TRY(y.v.declare(b, nm, g, nm.f("blk.%d.attn_v", l), kv_dim, g.n_embd, RAD_SHARD_ROW,
                             grp_layer(l), { nm.ckpt("%s.%d.self_attn.v_proj.weight", pre, l) },
                             m.a_h, m.b_v));
    if (!y.q.bf16 || !y.k.bf16 || !y.v.bf16) {
        fprintf(stderr, "radiance: kolibri1_exl3: layer %d's q/k/v are not plain bf16; this plugin "
                        "serves the bf16 trunk of the EXL3 export\n", l);
        return RAD_E_DTYPE;
    }
    /* q and k into the fused row qk_norm_rope reads ([q | k | v], v unused there). */
    y.op_qcp = decl_copy(b, g, q_dim, m.b_q, m.b_qkv);
    y.op_kcp = decl_copy(b, g, kv_dim, m.b_k, m.b_qkv);

    /* THE PER-HEAD NORMS, AND THE ROTARY ONLY WHERE THE LAYER SLIDES. A global layer is
     * `rotary_dim` 0 -- the norm and nothing turned (no positional encoding). */
    y.w_qn = decl_gain(b, nm, g, l, nm.f("blk.%d.self_attn.q_norm.weight", l),
                       nm.ckpt("%s.%d.self_attn.q_norm.weight", pre, l), RAD_BF16, g.head_dim);
    y.w_kn = decl_gain(b, nm, g, l, nm.f("blk.%d.self_attn.k_norm.weight", l),
                       nm.ckpt("%s.%d.self_attn.k_norm.weight", pre, l), RAD_BF16, g.head_dim);
    if (!y.w_qn || !y.w_kn) return RAD_E_INVAL;
    y.op_qkn = rw(b, RAD_OP(b, "qk_norm_rope",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                 RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", g.n_head_kv),
                                 RAD_F64("theta", g.theta), RAD_F64("eps", g.eps),
                                 RAD_STR("mode", "neox"),
                                 RAD_INT("rotary_dim", y.full ? 0 : g.head_dim),
                                 RAD_F64("wadd", 0.0)),
                      RAD_WEIGHTS(y.w_qn, y.w_kn)),
                  {m.b_qkv}, {m.b_qkv});

    /* q is lifted back out of the fused row: paged attention reads it as one dense block. */
    y.op_qlift = decl_copy(b, g, q_dim, m.b_qkv, m.b_q);

    /* THE WINDOW IS THE QUERY AND THE 512 KEYS BEFORE IT (`sliding_window` 513). Declared before
     * kv_store, which reads the resolved kernel's block size back. */
    y.op_attn = rw(b, RAD_OP(b, "attn_paged",
                       RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                  RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                  RAD_INT("window", y.full ? 0 : m.window),
                                  RAD_INT("n_head", g.n_head),
                                  RAD_INT("block_size", RAD_KV_BLOCK),
                                  RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                                  RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                       RAD_NOWEIGHTS),
                   {m.b_q}, {m.a_attn.x});
    y.op_kv_store = rw(b, RAD_OP(b, "kv_store",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                      RAD_INT("n_head_kv", g.n_head_kv),
                                      RAD_INT("block_size", rad_kv_block_size(b, y.kv)),
                                      RAD_STR("kv_dtype", g.kv_dtype)),
                           RAD_NOWEIGHTS),
                       {m.b_qkv, m.b_v}, {});

    RAD_ARCH_TRY(y.o.declare(b, nm, g, nm.f("blk.%d.attn_output", l), g.n_embd, q_dim,
                             RAD_SHARD_COL, grp_layer(l),
                             { nm.ckpt("%s.%d.self_attn.o_proj.weight", pre, l) },
                             m.a_attn, m.a_h.x));
    y.op_ar = decl_ar(b, g, m.a_h.x);

    y.w_apost = decl_gain(b, nm, g, l, nm.f("blk.%d.post_attn_norm.weight", l),
                          nm.ckpt("%s.%d.post_attn_norm.weight", pre, l), RAD_F32, g.n_embd);
    if (!y.w_apost) return RAD_E_INVAL;
    y.op_apost = decl_rmsnorm(b, g, y.w_apost, m.a_h.x, m.b_pn);
    return RAD_OK;
}

static int declare_ffn(RadBuilder* b, Model& m, Layer& y, int l) {
    Geom&  g  = m.g;
    Names& nm = m.nm;
    const char* pre = "model.layers";

    y.w_mlpn = decl_gain(b, nm, g, l, nm.f("blk.%d.post_attention_layernorm.weight", l),
                         nm.ckpt("%s.%d.post_attention_layernorm.weight", pre, l), RAD_F32,
                         g.n_embd);
    if (!y.w_mlpn) return RAD_E_INVAL;
    y.op_add_mlp  = decl_add(b, g, m.b_x, m.b_pn);
    y.op_norm_mlp = decl_rmsnorm(b, g, y.w_mlpn, m.b_x, m.a_h.x);

    MoeEXL3::Wire mw{};
    mw.x = m.b_x; mw.h = m.a_h;
    mw.logits = m.b_rlogits; mw.ids = m.b_eids; mw.ew = m.b_ew;
    mw.sorted = m.b_sorted;  mw.eoff = m.b_eoff; mw.ecnt = m.b_ecnt;
    mw.eff = m.b_eff; mw.edn = m.b_edn; mw.moe = m.b_emoe;
    mw.sgu = m.b_sgu; mw.sff = m.a_sff; mw.sout = m.b_sout; mw.sgate = m.b_sgate;

    /* THE INPUT IS PREPARED (`norm` null): the norm above wrote it. The block's delta lands in
     * `a_h.x` and is reduced there. */
    MoeEXL3::Src ms{};
    ms.cfg              = m.mc;
    ms.norm             = nullptr;
    ms.router           = nm.ckpt("%s.%d.mlp.gate.weight", pre, l);
    ms.router_bias      = nm.ckpt("%s.%d.moe.router.expert_bias", pre, l);
    ms.router_bias_decl = nm.f("blk.%d.ffn_expert_bias.weight", l);
    ms.experts          = nm.ckpt("%s.%d.mlp.experts", pre, l);
    ms.shared           = nm.ckpt("%s.%d.mlp.shared_experts", pre, l);
    ms.shared_gate      = nullptr;
    RAD_ARCH_TRY(y.moe.declare(b, nm, g, l, mw, ms, /*fold=*/false, /*add_out=*/false,
                               kArNone, /*ar_out=*/false));

    y.w_fpost = decl_gain(b, nm, g, l, nm.f("blk.%d.post_ffn_norm.weight", l),
                          nm.ckpt("%s.%d.post_ffn_norm.weight", pre, l), RAD_F32, g.n_embd);
    if (!y.w_fpost) return RAD_E_INVAL;
    y.op_fpost = decl_rmsnorm(b, g, y.w_fpost, m.a_h.x, m.b_pn);
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;

    Model& m = g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(check_format(meta));
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));
    Geom&  g  = m.g;
    Names& nm = m.nm;
    /* THE TRUNK IS bf16 (the header): every LinearFP8 is gemm_nt over the bf16 plane and checks
     * its own weight against this, and no producer quantises. */
    g.w_bf16 = true;

    if (g.world > g.n_head_kv_all) {
        fprintf(stderr, "radiance: kolibri1_exl3 splits its %lld KV heads between the ranks and "
                        "cannot serve world_size %d\n", (long long)g.n_head_kv_all, g.world);
        return RAD_E_UNSUPPORTED;
    }
    if (g.head_dim != 128) {
        fprintf(stderr, "radiance: kolibri1_exl3: head_dim %lld; the release is 128\n",
                (long long)g.head_dim);
        return RAD_E_UNSUPPORTED;
    }
    if (rad_meta_geti(meta, "tie_word_embeddings", 0)) {
        fprintf(stderr, "radiance: kolibri1_exl3: a tied lm_head is not this release\n");
        return RAD_E_UNSUPPORTED;
    }
    if (calib_dir()) {
        fprintf(stderr, "radiance: kolibri1_exl3 has no calibration run; record Hessians with the "
                        "fp8 container and the in-tree kolibri1 plugin\n");
        return RAD_E_UNSUPPORTED;
    }
    m.window = rad_meta_geti(meta, "sliding_window", 0);
    if (m.window <= 0) {
        fprintf(stderr, "radiance: kolibri1_exl3: the sliding layers need a positive "
                        "sliding_window\n");
        return RAD_E_FORMAT;
    }
    m.layers.resize((size_t)g.n_layer);
    RAD_ARCH_TRY(parse_layer_types(rad_meta_gets(meta, "layer_types", nullptr), g.n_layer,
                                   &m.layers));

    /* ---- THE ROUTING. `n_ff` is already the per-rank expert width: 512 at two ranks is two
     * whole 128-column groups each (the Hadamard width), the even split, so no class slices. */
    {
        MoeEXL3::Config& mc = m.mc;
        mc.n_expert = meta->n_expert;
        mc.top_k    = meta->n_expert_used > 0 ? meta->n_expert_used
                                              : rad_meta_geti(meta, "num_experts_per_tok", 0);
        mc.n_ff_exp = g.n_ff;
        if (g.world > 1 && g.n_ff * g.world != g.n_ff_all) {
            fprintf(stderr, "radiance: kolibri1_exl3: an expert of %lld columns does not split "
                            "evenly over %d ranks\n", (long long)g.n_ff_all, g.world);
            return RAD_E_UNSUPPORTED;
        }
        /* norm_topk_prob is FALSE in this release: the six sigmoid weights are used as they are. */
        mc.norm_topk     = (int)rad_meta_geti(meta, "norm_topk_prob", 0);
        mc.route_sigmoid = 1;
        const int64_t sh_all = rad_meta_geti(meta, "shared_expert_intermediate_size", 0);
        if (sh_all <= 0) {
            fprintf(stderr, "radiance: kolibri1_exl3: every layer has a shared expert and the "
                            "container gives no shared_expert_intermediate_size\n");
            return RAD_E_FORMAT;
        }
        RAD_ARCH_TRY(divide_or_fail(sh_all, g.world, "shared_expert_intermediate_size",
                                    &mc.n_ff_shared));
        const int64_t moe_rows = 2048;
        mc.rows = g.max_tok < moe_rows ? g.max_tok : moe_rows;
        if (mc.top_k <= 0 || mc.top_k > mc.n_expert) {
            fprintf(stderr, "radiance: kolibri1_exl3: top_k %lld is not in [1, %lld]\n",
                    (long long)mc.top_k, (long long)mc.n_expert);
            return RAD_E_INVAL;
        }
        /* THE EXPERTS ARE EXL3, read off layer 0's first expert before anything is sized by it. */
        mc.probed_expert = 0;
        MoeEXL3::Src ps{};
        ps.experts = nm.ckpt("model.layers.0.mlp.experts");
        RAD_ARCH_TRY(MoeEXL3::probe(b, nm, 0, ps, &mc));
    }

    /* ---- the KV cache: the global layers' full group and the sliding layers' window group, or
     * one full group for all of them. */
    {
        const char* one = std::getenv("RADIANCE_KOLIBRI_KV_ONE_GROUP");
        m.kv_one = one && *one && std::strcmp(one, "0") != 0;
    }
    RadKVGroupDecl fd{};
    fd.kind      = RAD_KV_FULL;
    fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    fd.n_head_kv = g.n_head_kv;
    fd.head_dim  = g.head_dim;
    m.kv_g = rad_decl_kv_group(b, nm.f(m.kv_one ? "kv_attn" : "kv_global"), &fd);
    if (m.kv_one) {
        m.kv_s = m.kv_g;
    } else {
        RadKVGroupDecl sd = fd;
        sd.kind   = RAD_KV_WINDOW;
        sd.window = m.window;
        m.kv_s = rad_decl_kv_group(b, nm.f("kv_sliding"), &sd);
    }
    if (!m.kv_g || !m.kv_s) {
        fprintf(stderr, "radiance: kolibri1_exl3: a KV group failed to declare\n");
        return RAD_E_INVAL;
    }
    g.kv_dtype = rad_dtype_name(fd.dtype);
    for (Layer& y : m.layers) y.kv = y.full ? m.kv_g : m.kv_s;

    /* ---- activations. */
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();
    m.b_x      = decl_b(b, nm.f("x"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.b_pn     = decl_b(b, nm.f("post"),     g.act_dtype, {g.max_tok, g.n_embd});
    m.a_h.x    = decl_b(b, nm.f("h"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.b_q      = decl_b(b, nm.f("attn_q"),   g.act_dtype, {g.max_tok, q_dim});
    m.b_k      = decl_b(b, nm.f("attn_k"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_v      = decl_b(b, nm.f("attn_v"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_qkv    = decl_b(b, nm.f("attn_qkv"), g.act_dtype, {g.max_tok, q_dim + 2 * kv_dim});
    m.a_attn.x = decl_b(b, nm.f("attn_out"), g.act_dtype, {g.max_tok, q_dim});
    {
        /* THE STEP'S HEIGHT FOR WHAT IS ONE ROW A TOKEN, A PASS'S FOR WHAT IS ONE ROW A
         * (token, slot) -- rad_block_moe_exl3.h states the rule at Wire. The router's logits are
         * F32 (`route_sigmoid`); the experts' intermediate is F16, as the EXL3 gate_up writes it. */
        const MoeEXL3::Config& mc = m.mc;
        const int64_t R = g.max_tok, ER = mc.rows * mc.top_k;
        m.b_rlogits = decl_b(b, nm.f("moe_logits"), RAD_F32,     {R, mc.n_expert});
        m.b_eids    = decl_b(b, nm.f("moe_ids"),    RAD_I32,     {R, mc.top_k});
        m.b_ew      = decl_b(b, nm.f("moe_w"),      g.act_dtype, {R, mc.top_k});
        m.b_sorted  = decl_b(b, nm.f("moe_sorted"), RAD_I32,     {ER});
        m.b_eoff    = decl_b(b, nm.f("moe_off"),    RAD_I32,     {mc.n_expert + 1});
        /* PERSIST: the histogram's only reader is rad_route_report, not a declared op. */
        m.b_ecnt    = decl_b(b, nm.f("moe_cnt"),    RAD_I32,     {mc.n_expert}, RAD_BUF_PERSIST);
        m.b_eff     = decl_b(b, nm.f("moe_eff"),    RAD_F16,     {ER, mc.n_ff_exp});
        m.b_edn     = decl_b(b, nm.f("moe_down"),   g.act_dtype, {ER, g.n_embd});
        m.b_emoe    = decl_b(b, nm.f("moe_out"),    g.act_dtype, {R, g.n_embd});
        m.b_sgu     = decl_b(b, nm.f("shexp_gate_up"), g.act_dtype, {R, 2 * mc.n_ff_shared});
        m.a_sff.x   = decl_b(b, nm.f("shexp_ffn"),     g.act_dtype, {R, mc.n_ff_shared});
        m.b_sout    = decl_b(b, nm.f("shexp_out"),     g.act_dtype, {R, g.n_embd});
        m.b_sgate   = decl_b(b, nm.f("shexp_gate"),    g.act_dtype, {R, 1});
    }
    m.b_hout   = decl_b(b, nm.f("h_out"),    g.act_dtype, {g.max_logit_rows, g.n_embd});
    m.b_logits = decl_b(b, nm.f("logits"),   RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- the vocabulary edges, both row-sharded with the vocabulary. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("model.norm.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output.weight"), nm.ckpt("lm_head.weight")));
    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.w_out_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.w_lm_head = decl_w(b, nm.f("output.weight"),
                         weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                         {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                   RAD_INT("vocab_offset", g.vocab_off)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    /* EXACT: each token's row comes from one rank and every other adds a zero. */
    if (g.world > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                               RAD_PARAMS(RAD_INT("world_size", g.world),
                                          RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                          RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                               RAD_NOWEIGHTS),
                           {m.b_x}, {m.b_x});

    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& y = m.layers[(size_t)l];
        RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, y.kv));
        RAD_ARCH_TRY(declare_attention(b, m, y, (int)l));
        RAD_ARCH_TRY(declare_ffn(b, m, y, (int)l));
    }
    /* Each routed layer predicts the next one's experts at its router input (MoeEXL3::xp_next). */
    for (int64_t l = 0; l + 1 < g.n_layer; ++l)
        m.layers[(size_t)l].moe.xp_next = &m.layers[(size_t)l + 1].moe;

    /* THE LAST LAYER'S POST-NORMED DELTA has no next input norm to take it, so it is added here,
     * and the final norm is a plain one. */
    m.op_last_add = decl_add(b, g, m.b_x, m.b_pn);
    m.op_final_norm = decl_rmsnorm(b, g, m.w_out_norm, m.b_x, m.a_h.x);
    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {m.a_h.x}, {m.b_hout});
    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n_vocab", g.n_vocab),
                                    RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(m.w_lm_head)),
                     {m.b_hout}, {m.b_logits});

    /* PREFILL STAGING (exl3_stage.h): radiance's own default, on, unless EXL3_PREFILL_STAGE says
     * otherwise; off is layer 0's never-issued expert reader, appended after every op. */
    if (g.n_layer > 1 && !exl3::prefill_stage(exl3::kPrefillStageForced, ctx->world_size,
                                              (long long)ctx->max_tok,
                                              (long long)m.layers[0].moe.c.n_expert_all,
                                              (long long)m.layers[0].moe.c.top_k))
        RAD_ARCH_TRY(m.layers[0].moe.declare_stage_off(b));

    int n_full = 0;
    for (const Layer& y : m.layers) n_full += y.full ? 1 : 0;
    rad_note(b, "rank %d/%d: %lld layers, %d global (no positional encoding) and %lld sliding "
                "(window %lld, rope theta %g); %lld heads / %lld kv x %lld per rank (gqa %lld); "
                "vocab [%lld,%lld); bf16 trunk, q/k/v as three GEMMs",
             g.rank, g.world, (long long)g.n_layer, n_full, (long long)g.n_layer - n_full,
             (long long)m.window, (double)g.theta, (long long)g.n_head, (long long)g.n_head_kv,
             (long long)g.head_dim, (long long)g.gqa(), (long long)g.vocab_off,
             (long long)(g.vocab_off + g.n_vocab));
    rad_note(b, "%s", m.kv_one
        ? "KV: one full paged group for all layers (RADIANCE_KOLIBRI_KV_ONE_GROUP) -- every block "
          "prefix-cacheable, a token costs every layer"
        : "KV: the global layers in a full group, the sliding layers in a window group -- a token "
          "costs the global layers alone; a prefix hit resumes at a checkpoint's window snapshot");
    const MoeEXL3& m0 = m.layers[0].moe;
    rad_note(b, "every layer routed: %lld EXL3 experts (%s, %s) of %lld (this rank's width %lld) at "
                "top %lld, chosen on f32 logit + expert_bias and weighted by sigmoid(logit)%s; one "
                "shared expert of %lld per rank, bf16, added ungated",
             (long long)m.mc.n_expert, m0.eenc[0].dtype ? m0.eenc[0].dtype : "?",
             m0.eenc[0].cb ? m0.eenc[0].cb : "?", (long long)g.n_ff_all,
             (long long)m.mc.n_ff_exp, (long long)m.mc.top_k,
             m.mc.norm_topk ? ", renormalised" : ", not renormalised",
             (long long)m.mc.n_ff_shared);
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_rank(c)];
    const Geom& g = m.g;
    if (batch->draft_pass != 0) {
        rad_step_fail(c, "kolibri1_exl3: the batch asks for a drafter pass and this plugin "
                         "declares no drafter");
        return;
    }
    const int64_t T = batch->n_tok;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();

    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.b_x, T));
    if (m.op_embed_ar)
        RAD_ISSUE_N(c, m.op_embed_ar, T * g.n_embd, brows(m.b_x, T), RAD_NONE);

    for (size_t li = 0; li < m.layers.size(); ++li) {
        const Layer& y = m.layers[li];
        const int layer = (int)li;
        const RadKVGroupBatch* kvb = kv_batch(batch, y.kv);
        const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
        const int32_t* table = kvb ? kvb->block_table  : nullptr;
        const int32_t* used  = kvb ? kvb->seqused      : nullptr;

        /* ---- attention */
        if (y.op_add_in)
            RAD_ISSUE(c, y.op_add_in, brows(m.b_x, T), brows(m.b_pn, T), brows(m.b_x, T));
        RAD_ISSUE(c, y.op_norm_in, brows(m.b_x, T), RAD_W(y.w_in), brows(m.a_h.x, T));
        y.q.step(c, m.a_h, m.b_q, T);
        y.k.step(c, m.a_h, m.b_k, T);
        y.v.step(c, m.a_h, m.b_v, T);
        RAD_ISSUE(c, y.op_qcp, brows(m.b_q, T), bcol(m.b_qkv, 0, q_dim, T));
        RAD_ISSUE(c, y.op_kcp, brows(m.b_k, T), bcol(m.b_qkv, q_dim, kv_dim, T));
        RAD_ISSUE(c, y.op_qkn, brows(m.b_qkv, T), praw(batch->positions, RAD_I32, T),
                  RAD_W(y.w_qn), RAD_W(y.w_kn));
        RAD_ISSUE(c, y.op_kv_store, bcol(m.b_qkv, q_dim, kv_dim, T), brows(m.b_v, T),
                  praw(slot, RAD_I32, T), kv_cache(y.kv, layer));
        RAD_ISSUE(c, y.op_qlift, bcol(m.b_qkv, 0, q_dim, T), brows(m.b_q, T));
        RAD_ISSUE_N(c, y.op_attn, batch->max_q_len,
                    brows(m.b_q, T), kv_cache(y.kv, layer),
                    praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                    praw(used, RAD_I32, batch->n_seq),
                    RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                    brows(m.a_attn.x, T));
        y.o.step(c, m.a_attn, m.a_h.x, T);
        if (y.op_ar) RAD_ISSUE_N(c, y.op_ar, T * g.n_embd, brows(m.a_h.x, T), RAD_NONE);
        RAD_ISSUE(c, y.op_apost, brows(m.a_h.x, T), RAD_W(y.w_apost), brows(m.b_pn, T));

        /* ---- the routed feed-forward */
        RAD_ISSUE(c, y.op_add_mlp, brows(m.b_x, T), brows(m.b_pn, T), brows(m.b_x, T));
        RAD_ISSUE(c, y.op_norm_mlp, brows(m.b_x, T), RAD_W(y.w_mlpn), brows(m.a_h.x, T));
        y.moe.step(c, batch);
        RAD_ISSUE(c, y.op_fpost, brows(m.a_h.x, T), RAD_W(y.w_fpost), brows(m.b_pn, T));
    }

    RAD_ISSUE(c, m.op_last_add, brows(m.b_x, T), brows(m.b_pn, T), brows(m.b_x, T));
    RAD_ISSUE(c, m.op_final_norm, brows(m.b_x, T), RAD_W(m.w_out_norm), brows(m.a_h.x, T));
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.a_h.x, T),
                    praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                    brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                    brows(m.b_logits, batch->n_out));
    }
}

/* THE REPLY FORMAT, stated because the template's analysis cannot read it: rad_kolibri1_chat.h. */
static const RadChatFormat* chat_format(const RadModelMeta*) { return kolibri1_chat_format(); }

}  /* namespace kolibri1_exl3 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_CHAT_FORMAT(kolibri1_exl3)
RAD_ARCH_PLUGIN(kolibri1_exl3, "kolibri1", "exl3", "0.1.0",
                "Aleph Alpha Kolibri 1 with EXL3 routed experts (moe_exl3_gate_up / moe_exl3_down; "
                "384 of 512 at top 6, sigmoid routing with a selection bias, one ungated shared "
                "expert) and a bf16 trunk: GQA 12 at head_dim 128 with q/k norms, four 513-token "
                "sliding-window layers to one global layer without positional encoding, sandwich "
                "norms")
#endif
