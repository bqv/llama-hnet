#include "models.h"

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

[[noreturn]]
std::unique_ptr<llm_graph_context> llama_model_hnet::build_arch_graph(const llm_graph_params &) const {
    GGML_ABORT("hnet: graph is not implemented");
}
