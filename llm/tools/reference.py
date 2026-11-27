# The reference llm/model's architectures are held to: small models of random weights of each architecture and option
# (Llama 3's and YaRN's rope scaling, Granite's multipliers, SmolLM3's layers without rotation, Mistral's sliding
# window, the mixtures of experts of Qwen3-MoE (a dense layer among them), Granite-MoE and OLMoE), built and run by Hugging Face transformers in F32 on the CPU. Each is saved as a checkpoint under
# $OANN_MODELS/oann-tiny-NAME with ref.json: the token ids read and the logits of every position.
#     python3 llm/tools/reference.py        (pip install torch transformers safetensors)
import json, os, sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM

root = os.environ.get("OANN_MODELS", os.path.expanduser("~/.cache/oann/models"))
base = dict(vocab_size=96, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, intermediate_size=96, rms_norm_eps=1e-6, max_position_embeddings=4096,
            tie_word_embeddings=False, rope_theta=10000.0)
cases = {
    "llama3": ("llama", dict(rope_scaling=dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0,
                                               high_freq_factor=4.0, original_max_position_embeddings=64))),
    "yarn": ("qwen3", dict(rope_scaling=dict(rope_type="yarn", factor=4.0, original_max_position_embeddings=64))),
    "granite": ("granite", dict(embedding_multiplier=12.0, residual_multiplier=0.22, attention_multiplier=0.05,
                                logits_scaling=4.0)),
    "smollm3": ("smollm3", dict(no_rope_layers=[1, 1, 1, 0], use_sliding_window=False, pad_token_id=0)),
    "mistral": ("mistral", dict(sliding_window=50)),
    "qwen3moe": ("qwen3_moe", dict(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64, norm_topk_prob=True,
                                   mlp_only_layers=[1])),
    "granitemoe": ("granitemoe", dict(num_local_experts=8, num_experts_per_tok=3, embedding_multiplier=6.0,
                                      residual_multiplier=0.5, attention_multiplier=0.1, logits_scaling=2.0)),
    "olmoe": ("olmoe", dict(hidden_size=128, num_experts=8, num_experts_per_tok=2, norm_topk_prob=False, pad_token_id=0, eos_token_id=1,
                            bos_token_id=None)),
}
n = 150
for name, (arch, extra) in ([] if "--real" in sys.argv else cases.items()):
    if len(sys.argv) > 1 and name not in sys.argv[1:]: continue
    torch.manual_seed(1)
    cfg = AutoConfig.for_model(arch, **{**base, **extra})
    cfg._attn_implementation = "eager"
    m = AutoModelForCausalLM.from_config(cfg).float().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 1: p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else: p.normal_(0.0, 0.15)
    dir = os.path.join(root, "oann-tiny-" + name)
    m.save_pretrained(dir)
    # BF16 as the checkpoints are published: the reference runs the rounded weights
    m2 = AutoModelForCausalLM.from_pretrained(dir, torch_dtype=torch.float32, attn_implementation="eager").eval()
    with torch.no_grad():
        for p in m2.parameters(): p.copy_(p.bfloat16().float())
    m2.save_pretrained(dir, safe_serialization=True)
    ids = torch.Generator().manual_seed(7)
    toks = torch.randint(0, 96, (1, n), generator=ids)
    with torch.no_grad():
        logits = m2(toks).logits[0]
    json.dump({"tokens": toks[0].tolist(), "logits": [round(x, 6) for x in logits.flatten().tolist()]},
              open(os.path.join(dir, "ref.json"), "w"))
    print(name, dir, float(logits.abs().max()))

# real checkpoints named on the command line after --real: "The capital of France is" read, the last position's
# logits over the whole vocabulary and ten greedy tokens after it, F32, into DIR/ref.json
if "--real" in sys.argv:
    from transformers import AutoTokenizer
    for d in sys.argv[sys.argv.index("--real") + 1:]:
        tok = AutoTokenizer.from_pretrained(d)
        ids = tok("The capital of France is", add_special_tokens=False, return_tensors="pt").input_ids
        m = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32).eval()
        with torch.no_grad():
            last = m(ids).logits[0, -1]
            out = m.generate(ids, max_new_tokens=10, do_sample=False)[0, ids.shape[1]:]
        json.dump({"tokens": ids[0].tolist(), "last": [round(x, 5) for x in last.tolist()], "greedy": out.tolist()},
                  open(os.path.join(d, "ref.json"), "w"))
        print(d, ids[0].tolist(), out.tolist(), tok.decode(out))
        del m
