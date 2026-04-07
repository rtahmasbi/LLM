
# q_proj, k_proj, v_proj, up_proj, down_proj
https://medium.com/@saratbhargava/mastering-llama-math-part-1-a-step-by-step-guide-to-counting-parameters-in-llama-2-b3d73bc3ae31

(pdf)

- Embedding Block
- Attention block (q_proj, k_proj, v_proj, o_proj)
- MLP Block (gate_proj, up_proj, down_proj)
- RMS Norm layers


```py

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 5120)
    (layers): ModuleList(
      (0-39): 40 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (k_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (v_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=5120, out_features=13824, bias=False)
          (up_proj): Linear(in_features=5120, out_features=13824, bias=False)
          (down_proj): Linear(in_features=13824, out_features=5120, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=5120, out_features=32000, bias=False)
)

```

https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_llama/tutorial_accelerate_hf_llama_with_te.html
