#include "models.h"

#include <algorithm>
#include <cmath>

void llama_model_hnet::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_HNET_D_ENC,        hparams.hnet_d_enc);
    ml.get_key(LLM_KV_HNET_D_MAIN,       hparams.hnet_d_main);
    ml.get_key(LLM_KV_HNET_D_DEC,        hparams.hnet_d_dec);
    ml.get_key(LLM_KV_HNET_N_ENC,        hparams.hnet_n_enc);
    ml.get_key(LLM_KV_HNET_N_MAIN,       hparams.hnet_n_main);
    ml.get_key(LLM_KV_HNET_N_DEC,        hparams.hnet_n_dec);
    ml.get_key(LLM_KV_HNET_N_HEAD_MAIN,  hparams.hnet_n_head_main);
    ml.get_key(LLM_KV_HNET_N_HEAD_ENC,   hparams.hnet_n_head_enc);
    ml.get_key(LLM_KV_HNET_HEAD_DIM_ENC, hparams.hnet_head_dim_enc);
    ml.get_key(LLM_KV_HNET_D_FF_MAIN,    hparams.hnet_d_ff_main);
    ml.get_key(LLM_KV_HNET_TARGET_RATIO, hparams.hnet_target_ratio, false);
    std::string dechunk_mode = "ema";
    ml.get_key(LLM_KV_HNET_DECHUNK_MODE, dechunk_mode, false);
    hparams.hnet_dechunk_mode = dechunk_mode == "ema" ? 0 : 1;

    hparams.f_norm_rms_eps = 1e-5f;
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps, false);

    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_HNET_ROPE_BASE, hparams.rope_freq_base_train, false);

    hparams.n_embd          = hparams.hnet_d_main;
    hparams.n_embd_inp_impl = hparams.hnet_d_enc;
    hparams.n_embd_out_impl = hparams.hnet_d_dec;
    hparams.n_layer_all     = hparams.hnet_n_enc + hparams.hnet_n_main + hparams.hnet_n_dec;

    const uint32_t n_embd_head_main = hparams.hnet_n_head_main > 0 ? hparams.hnet_d_main / hparams.hnet_n_head_main : 0;

    hparams.n_embd_head_k_full = n_embd_head_main;
    hparams.n_embd_head_v_full = n_embd_head_main;
    hparams.n_rot_full         = n_embd_head_main;

    ml.get_key(LLM_KV_HNET_SSM_D_STATE, hparams.ssm_d_state, false);
    ml.get_key(LLM_KV_HNET_SSM_D_CONV,  hparams.ssm_d_conv,  false);
    ml.get_key(LLM_KV_HNET_SSM_N_GROUP, hparams.ssm_n_group, false);

    uint32_t ssm_expand = 2;
    ml.get_key(LLM_KV_HNET_SSM_EXPAND, ssm_expand, false);
    hparams.ssm_d_inner = ssm_expand * hparams.hnet_d_enc;

    uint32_t rotary_emb_dim = n_embd_head_main;
    ml.get_key(LLM_KV_HNET_ROTARY_EMB_DIM, rotary_emb_dim, false);
    hparams.n_rot_full = std::min<uint32_t>(hparams.n_rot_full, rotary_emb_dim);

    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        const bool is_main = il >= hparams.hnet_n_enc && il < hparams.hnet_n_enc + hparams.hnet_n_main;

        hparams.n_head_arr[il]    = is_main ? hparams.hnet_n_head_main : 0;
        hparams.n_head_kv_arr[il] = is_main ? hparams.hnet_n_head_main : 0;
        hparams.n_ff_arr[il]      = is_main ? hparams.hnet_d_ff_main    : 0;
        hparams.is_recr_impl[il]  = is_main ? 0 : 1;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_hnet::load_arch_tensors(llama_model_loader & ml) {
    auto create = [&](const LLM_TN_IMPL & t, int flags) -> ggml_tensor * {
        const ggml_tensor * meta = ml.get_tensor_meta(t.str().c_str());
        if (meta == nullptr) {
            throw std::runtime_error(format("missing tensor '%s'", t.str().c_str()));
        }

        return create_tensor(t, {meta->ne[0], meta->ne[1], meta->ne[2], meta->ne[3]}, flags);
    };

    tok_embd = create(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), 0);
    output   = create(tn(LLM_TENSOR_OUTPUT,    "weight"), 0);

    enc_layers.resize(hparams.hnet_n_enc);

    for (int i = 0; i < (int) hparams.hnet_n_enc; ++i) {
        auto & layer = enc_layers[i];

        layer.in_proj  = create(tn(LLM_TENSOR_HNET_ENC_IN,       "weight", i), 0);
        layer.conv1d   = create(tn(LLM_TENSOR_HNET_ENC_CONV1D,   "weight", i), 0);
        layer.conv1d_b = create(tn(LLM_TENSOR_HNET_ENC_CONV1D,   "bias",   i), 0);
        layer.dt_b     = create(tn(LLM_TENSOR_HNET_ENC_DT,                 i), 0);
        layer.a        = create(tn(LLM_TENSOR_HNET_ENC_A,                  i), 0);
        layer.d        = create(tn(LLM_TENSOR_HNET_ENC_D,                  i), 0);
        layer.norm     = create(tn(LLM_TENSOR_HNET_ENC_SSM_NORM, "weight", i), 0);
        layer.out_proj = create(tn(LLM_TENSOR_HNET_ENC_OUT,      "weight", i), 0);
        layer.norm1    = create(tn(LLM_TENSOR_HNET_ENC_NORM,     "weight", i), 0);
    }

    enc_rmsnorm = create(tn(LLM_TENSOR_HNET_ENC_RMSNORM, "weight"), 0);

    router_q = create(tn(LLM_TENSOR_HNET_ROUTER_Q, "weight"), 0);
    router_k = create(tn(LLM_TENSOR_HNET_ROUTER_K, "weight"), 0);

    main_layers.resize(hparams.hnet_n_main);

    for (int i = 0; i < (int) hparams.hnet_n_main; ++i) {
        auto & layer = main_layers[i];

        layer.wqkv     = create(tn(LLM_TENSOR_HNET_BLK_QKV,   "weight", i), 0);
        layer.out_proj = create(tn(LLM_TENSOR_HNET_BLK_OUT,   "weight", i), 0);
        layer.fc1      = create(tn(LLM_TENSOR_HNET_BLK_FC1,   "weight", i), 0);
        layer.fc2      = create(tn(LLM_TENSOR_HNET_BLK_FC2,   "weight", i), 0);
        layer.norm1    = create(tn(LLM_TENSOR_HNET_BLK_NORM1, "weight", i), 0);
        layer.norm2    = create(tn(LLM_TENSOR_HNET_BLK_NORM2, "weight", i), 0);
    }

    blk_rmsnorm = create(tn(LLM_TENSOR_HNET_BLK_RMSNORM, "weight"), 0);

    dec_layers.resize(hparams.hnet_n_dec);

    for (int i = 0; i < (int) hparams.hnet_n_dec; ++i) {
        auto & layer = dec_layers[i];

        layer.in_proj  = create(tn(LLM_TENSOR_HNET_DEC_IN,       "weight", i), 0);
        layer.conv1d   = create(tn(LLM_TENSOR_HNET_DEC_CONV1D,   "weight", i), 0);
        layer.conv1d_b = create(tn(LLM_TENSOR_HNET_DEC_CONV1D,   "bias",   i), 0);
        layer.dt_b     = create(tn(LLM_TENSOR_HNET_DEC_DT,                 i), 0);
        layer.a        = create(tn(LLM_TENSOR_HNET_DEC_A,                  i), 0);
        layer.d        = create(tn(LLM_TENSOR_HNET_DEC_D,                  i), 0);
        layer.norm     = create(tn(LLM_TENSOR_HNET_DEC_SSM_NORM, "weight", i), 0);
        layer.out_proj = create(tn(LLM_TENSOR_HNET_DEC_OUT,      "weight", i), 0);
        layer.norm1    = create(tn(LLM_TENSOR_HNET_DEC_NORM,     "weight", i), 0);
    }

    dec_rmsnorm = create(tn(LLM_TENSOR_HNET_DEC_RMSNORM, "weight"), 0);

    residual_proj   = create(tn(LLM_TENSOR_HNET_RESIDUAL_PROJ, "weight"), 0);
    residual_proj_b = create(tn(LLM_TENSOR_HNET_RESIDUAL_PROJ, "bias"),   0);
}

namespace {

struct hnet_mamba2_dims {
    int64_t d_conv   = 0;
    int64_t d_inner  = 0;
    int64_t d_state  = 0;
    int64_t n_group  = 0;
    int64_t n_head   = 0;
    int64_t head_dim = 0;
    int64_t conv_dim = 0;
};

hnet_mamba2_dims hnet_mamba2_get_dims(const llama_layer_hnet_mamba2 & layer) {
    hnet_mamba2_dims d;

    d.d_conv   = layer.conv1d->ne[0];
    d.d_inner  = layer.out_proj->ne[0];
    d.n_group  = layer.norm->ne[1];
    d.conv_dim = layer.conv1d->ne[1];
    d.d_state  = (d.conv_dim - d.d_inner) / (2 * d.n_group);
    d.n_head   = layer.dt_b->ne[0];
    d.head_dim = d.d_inner / d.n_head;

    GGML_ASSERT(d.conv_dim == d.d_inner + 2 * d.n_group * d.d_state);
    GGML_ASSERT(d.d_inner % d.n_head  == 0);
    GGML_ASSERT(d.d_inner % d.n_group == 0);

    return d;
}

ggml_tensor * hnet_mamba2_layer(
        const llm_graph_context & ctx,
                  ggml_tensor * cur,
   const llama_layer_hnet_mamba2 & layer,
                  ggml_tensor * ids,
                          int   il) {
    ggml_context * ctx0 = ctx.ctx0;

    const hnet_mamba2_dims d = hnet_mamba2_get_dims(layer);

    const int64_t n_tokens = cur->ne[1];

    ggml_tensor * zxBCdt = ctx.build_lora_mm(layer.in_proj, ggml_reshape_3d(ctx0, cur, cur->ne[0], n_tokens, 1));
    zxBCdt = ggml_reshape_3d(ctx0, zxBCdt, zxBCdt->ne[0], n_tokens, 1);

    ggml_tensor * z   = ggml_view_4d(ctx0, zxBCdt, d.head_dim, d.n_head, n_tokens, 1, d.head_dim * zxBCdt->nb[0], zxBCdt->nb[1], zxBCdt->nb[2], 0);
    ggml_tensor * xBC = ggml_view_3d(ctx0, zxBCdt, d.conv_dim, n_tokens, 1, zxBCdt->nb[1], zxBCdt->nb[2], d.d_inner * zxBCdt->nb[0]);
    ggml_tensor * dt  = ggml_view_3d(ctx0, zxBCdt, d.n_head, n_tokens, 1, zxBCdt->nb[1], zxBCdt->nb[2], (2 * d.d_inner + 2 * d.n_group * d.d_state) * zxBCdt->nb[0]);

    {
        ggml_tensor * conv = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d.d_conv - 1, d.conv_dim, 1), 0.0f);
        ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, xBC), 0);

        xBC = ggml_ssm_conv(ctx0, conv_x, layer.conv1d);
        xBC = ggml_add(ctx0, xBC, layer.conv1d_b);
        xBC = ggml_silu(ctx0, xBC);
    }

    {
        ggml_tensor * x = ggml_view_4d(ctx0, xBC, d.head_dim, d.n_head, n_tokens, 1, d.head_dim * xBC->nb[0], xBC->nb[1], xBC->nb[2], 0);
        ggml_tensor * B = ggml_view_4d(ctx0, xBC, d.d_state, d.n_group, n_tokens, 1, d.d_state * xBC->nb[0], xBC->nb[1], xBC->nb[2], d.d_inner * ggml_element_size(xBC));
        ggml_tensor * C = ggml_view_4d(ctx0, xBC, d.d_state, d.n_group, n_tokens, 1, d.d_state * xBC->nb[0], xBC->nb[1], xBC->nb[2], (d.d_inner + d.n_group * d.d_state) * ggml_element_size(xBC));

        dt = ggml_add(ctx0, ggml_cont(ctx0, dt), layer.dt_b);

        ggml_tensor * A = ggml_reshape_2d(ctx0, ggml_neg(ctx0, ggml_exp(ctx0, ggml_cast(ctx0, layer.a, GGML_TYPE_F32))), 1, d.n_head);
        ggml_tensor * s = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, d.d_state, d.head_dim, d.n_head, 1), 0.0f);

        ggml_tensor * y_ssm = ggml_ssm_scan(ctx0, s, x, dt, A, B, C, ids, 1);

        ggml_tensor * y = ggml_view_4d(ctx0, y_ssm, d.head_dim, d.n_head, n_tokens, 1, x->nb[1], d.n_head * x->nb[1], n_tokens * d.n_head * x->nb[1], 0);

        y = ggml_add(ctx0, y, ggml_mul(ctx0, x, ggml_reshape_2d(ctx0, layer.d, 1, d.n_head)));
        y = ggml_swiglu_split(ctx0, ggml_cont(ctx0, z), y);

        y = ggml_reshape_4d(ctx0, y, d.d_inner / d.n_group, d.n_group, n_tokens, 1);
        y = ctx.build_norm(y, layer.norm, nullptr, LLM_NORM_RMS, il);

        y = ggml_reshape_2d(ctx0, y, d.d_inner, n_tokens);

        cur = ctx.build_lora_mm(layer.out_proj, y);
    }

    return cur;
}

ggml_tensor * hnet_attn_layer(
        const llm_graph_context & ctx,
                  ggml_tensor * cur,
   const llama_layer_hnet_main    & layer,
       llm_graph_input_attn_no_cache * inp_attn,
                  ggml_tensor * pos,
                          int   il) {
    ggml_context * ctx0 = ctx.ctx0;

    const int64_t n_head   = ctx.hparams.hnet_n_head_main;
    const int64_t d_main   = layer.wqkv->ne[0];
    const int64_t head_dim = d_main / n_head;
    const int64_t n_tokens = cur->ne[1];

    GGML_ASSERT(d_main % n_head == 0);
    GGML_ASSERT(layer.wqkv->ne[1] == 3 * d_main);

    ggml_tensor * qkv  = ctx.build_lora_mm(layer.wqkv, cur);
    ggml_tensor * qkv3 = ggml_reshape_3d(ctx0, qkv, head_dim, 3 * n_head, n_tokens);

    ggml_tensor * q = ggml_view_3d(ctx0, qkv3, head_dim, n_head, n_tokens, qkv3->nb[1], qkv3->nb[2], 0);
    ggml_tensor * k = ggml_view_3d(ctx0, qkv3, head_dim, n_head, n_tokens, qkv3->nb[1], qkv3->nb[2], ggml_row_size(qkv->type, d_main));
    ggml_tensor * v = ggml_cont(ctx0, ggml_view_3d(ctx0, qkv3, head_dim, n_head, n_tokens, qkv3->nb[1], qkv3->nb[2], 2 * ggml_row_size(qkv->type, d_main)));

    const int64_t n_rot = std::min<int64_t>(head_dim, ctx.hparams.n_rot_full);
    if (n_rot > 0) {
        const int n_ctx_orig = (int) ctx.hparams.n_ctx_train;

        q = ggml_rope_ext(ctx0, q, pos, nullptr, (int) n_rot, GGML_ROPE_TYPE_NEOX, n_ctx_orig, ctx.hparams.rope_freq_base_train, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(ctx0, k, pos, nullptr, (int) n_rot, GGML_ROPE_TYPE_NEOX, n_ctx_orig, ctx.hparams.rope_freq_base_train, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    }

    return ctx.build_attn(inp_attn, layer.out_proj, nullptr, nullptr,
            q, k, v, nullptr, nullptr, nullptr, 1.0f / std::sqrt((float) head_dim), il);
}

ggml_tensor * hnet_ffn_layer(
        const llm_graph_context & ctx,
                  ggml_tensor * cur,
   const llama_layer_hnet_main    & layer) {
    ggml_context * ctx0 = ctx.ctx0;

    const int64_t d_ff     = layer.fc2->ne[0];
    const int64_t n_tokens = cur->ne[1];

    GGML_ASSERT(layer.fc1->ne[1] == 2 * d_ff);

    ggml_tensor * fc = ctx.build_lora_mm(layer.fc1, cur);

    ggml_tensor * up   = ggml_view_2d(ctx0, fc, d_ff, n_tokens, fc->nb[1], 0);
    ggml_tensor * gate = ggml_view_2d(ctx0, fc, d_ff, n_tokens, fc->nb[1], d_ff * ggml_element_size(fc));

    cur = ggml_swiglu_split(ctx0, ggml_cont(ctx0, gate), ggml_cont(ctx0, up));

    return ctx.build_lora_mm(layer.fc2, cur);
}

ggml_tensor * hnet_router(const llm_graph_context & ctx, ggml_tensor * cur) {
    ggml_context * ctx0 = ctx.ctx0;

    const auto & m = static_cast<const llama_model_hnet &>(ctx.model);

    const int64_t d        = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];

    ggml_tensor * ones = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 1), 1.0f);

    if (n_tokens <= 1 || m.router_q == nullptr || m.router_k == nullptr) {
        return ones;
    }

    ggml_tensor * q_in = ggml_view_2d(ctx0, cur, d, n_tokens - 1, cur->nb[1], 0);
    ggml_tensor * k_in = ggml_view_2d(ctx0, cur, d, n_tokens - 1, cur->nb[1], cur->nb[1]);

    ggml_tensor * q = ggml_l2_norm(ctx0, ctx.build_lora_mm(m.router_q, q_in), 1e-12f);
    ggml_tensor * k = ggml_l2_norm(ctx0, ctx.build_lora_mm(m.router_k, k_in), 1e-12f);

    ggml_tensor * cos = ggml_sum_rows(ctx0, ggml_mul(ctx0, q, k));

    ggml_tensor * ones_row = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens - 1), 1.0f);
    ggml_tensor * prob     = ggml_clamp(ctx0, ggml_scale(ctx0, ggml_sub(ctx0, ones_row, cos), 0.5f), 0.0f, 1.0f);

    return ggml_concat(ctx0, ones, prob, 1);
}

ggml_tensor * hnet_dechunk(
        const llm_graph_context & ctx,
                  ggml_tensor * h,
                  ggml_tensor * cp,
                  ggml_tensor * ids) {
    ggml_context * ctx0 = ctx.ctx0;

    const int64_t d        = h->ne[0];
    const int64_t n_tokens = h->ne[1];

    ggml_tensor * ones  = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens), 1.0f);
    ggml_tensor * om    = ggml_sub(ctx0, ones, cp);
    ggml_tensor * logom = ggml_log(ctx0, om);
    ggml_tensor * dt    = ggml_neg(ctx0, logom);

    ggml_tensor * x = ggml_div(ctx0, h, dt);

    ggml_tensor * dt_fed = ggml_add(ctx0, ggml_log(ctx0, cp), dt);

    ggml_tensor * x4  = ggml_reshape_4d(ctx0, x,  d, 1, n_tokens, 1);
    ggml_tensor * dt3 = ggml_reshape_3d(ctx0, dt_fed, 1, n_tokens, 1);
    ggml_tensor * B   = ggml_reshape_4d(ctx0, cp, 1, 1, n_tokens, 1);
    ggml_tensor * C   = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens, 1), 1.0f);
    ggml_tensor * A   = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 1), -1.0f);
    ggml_tensor * s   = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, d, 1, 1), 0.0f);

    ggml_tensor * y = ggml_ssm_scan(ctx0, s, x4, dt3, A, B, C, ids, 1);

    return ggml_view_2d(ctx0, y, d, n_tokens, d * ggml_element_size(y), 0);
}

}

llama_model_hnet::graph_enc::graph_enc(const llm_graph_params & params) : llm_graph_context(params) {
    const auto & m = static_cast<const llama_model_hnet &>(model);

    GGML_ASSERT(!m.enc_layers.empty());
    GGML_ASSERT(m.enc_rmsnorm != nullptr);
    GGML_ASSERT(m.blk_rmsnorm != nullptr);
    GGML_ASSERT(m.dec_rmsnorm != nullptr);

    const int64_t d_enc = m.tok_embd->ne[0];

    auto inp = std::make_unique<llm_graph_input_hnet>(cross);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    inp->ids    = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(inp->tokens);
    ggml_set_input(inp->ids);

    ggml_tensor * tokens = inp->tokens;
    ggml_tensor * ids    = inp->ids;

    res->add_input(std::move(inp));

    ggml_tensor * cur = ggml_get_rows(ctx0, m.tok_embd, tokens);
    cb(cur, "tok_embd", -1);

    GGML_ASSERT(cur->ne[0] == d_enc);

    for (int i = 0; i < (int) m.enc_layers.size(); ++i) {
        const auto & layer = m.enc_layers[i];

        ggml_tensor * inpL = cur;

        cur = build_norm(cur, layer.norm1, nullptr, LLM_NORM_RMS, i);
        cur = hnet_mamba2_layer(*this, cur, layer, ids, i);
        cur = ggml_add(ctx0, cur, inpL);
        cb(cur, "enc_out", i);
    }

    ggml_tensor * enc = build_norm(cur, m.enc_rmsnorm, nullptr, LLM_NORM_RMS, -1);
    cb(enc, "enc_norm", -1);

    ggml_tensor * residual = build_lora_mm(m.residual_proj, enc);
    if (m.residual_proj_b) {
        residual = ggml_add(ctx0, residual, m.residual_proj_b);
    }

    ggml_tensor * p = hnet_router(*this, enc);

    ggml_tensor * packed = ggml_concat(ctx0, ggml_concat(ctx0, enc, residual, 0), p, 0);
    cb(packed, "result_enc", -1);

    res->t_embd = packed;
}

llama_model_hnet::graph_dec::graph_dec(const llm_graph_params & params) : llm_graph_context(params) {
    const auto & m = static_cast<const llama_model_hnet &>(model);

    GGML_ASSERT(!m.main_layers.empty());

    const int64_t d_enc  = m.tok_embd->ne[0];
    const int64_t d_main = m.main_layers[0].wqkv->ne[0];

    GGML_ASSERT(d_main >= d_enc);
    GGML_ASSERT(m.dec_layers.empty() || m.dec_layers[0].in_proj->ne[0] == d_enc);

    auto inp = std::make_unique<llm_graph_input_hnet>(cross);

    inp->enc       = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_enc, n_tokens);
    inp->res       = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_enc, n_tokens);
    inp->prob      = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    inp->chunk_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    inp->byte_map  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    inp->pos       = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    inp->ids       = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);

    for (ggml_tensor * t : { inp->enc, inp->res, inp->prob, inp->chunk_idx, inp->byte_map, inp->pos, inp->ids }) {
        ggml_set_input(t);
    }

    ggml_tensor * enc       = inp->enc;
    ggml_tensor * residual  = inp->res;
    ggml_tensor * prob      = inp->prob;
    ggml_tensor * chunk_idx = inp->chunk_idx;
    ggml_tensor * byte_map  = inp->byte_map;
    ggml_tensor * pos       = inp->pos;
    ggml_tensor * ids       = inp->ids;

    res->add_input(std::move(inp));

    ggml_tensor * cur = ggml_get_rows(ctx0, enc, chunk_idx);

    if (d_main > d_enc) {
        ggml_tensor * pad = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_main - d_enc, n_tokens), 0.0f);
        cur = ggml_concat(ctx0, cur, pad, 0);
    }

    auto * inp_attn = build_attn_inp_no_cache();

    for (int i = 0; i < (int) m.main_layers.size(); ++i) {
        const auto & layer = m.main_layers[i];

        ggml_tensor * inpL = cur;

        cur = build_norm(cur, layer.norm1, nullptr, LLM_NORM_RMS, i);
        cur = hnet_attn_layer(*this, cur, layer, inp_attn, pos, i);
        cur = ggml_add(ctx0, cur, inpL);
        cb(cur, "main_attn_out", i);

        inpL = cur;

        cur = build_norm(cur, layer.norm2, nullptr, LLM_NORM_RMS, i);
        cur = hnet_ffn_layer(*this, cur, layer);
        cur = ggml_add(ctx0, cur, inpL);
        cb(cur, "main_out", i);
    }

    cur = build_norm(cur, m.blk_rmsnorm, nullptr, LLM_NORM_RMS, -1);

    ggml_tensor * h = ggml_view_2d(ctx0, cur, d_enc, n_tokens, cur->nb[1], 0);

    ggml_tensor * cp = ggml_clamp(ctx0, ggml_get_rows(ctx0, prob, chunk_idx), 1e-4f, 1.0f - 1e-4f);
    ggml_tensor * y  = hnet_dechunk(*this, h, cp, ids);

    cur = ggml_get_rows(ctx0, y, byte_map);
    cur = ggml_add(ctx0, cur, residual);

    for (int i = 0; i < (int) m.dec_layers.size(); ++i) {
        const auto & layer = m.dec_layers[i];

        ggml_tensor * inpL = cur;

        cur = build_norm(cur, layer.norm1, nullptr, LLM_NORM_RMS, i);
        cur = hnet_mamba2_layer(*this, cur, layer, ids, i);
        cur = ggml_add(ctx0, cur, inpL);
        cb(cur, "dec_out", i);
    }

    cur = build_norm(cur, m.dec_rmsnorm, nullptr, LLM_NORM_RMS, -1);

    ggml_tensor * inp_out_ids = build_inp_out_ids();
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_lora_mm(m.output, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
}

std::unique_ptr<llm_graph_context> llama_model_hnet::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph_enc>(params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph_dec>(params);
        default:
            GGML_ABORT("invalid graph type");
    }
}
