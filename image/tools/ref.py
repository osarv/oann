# image/tools/ref - the reference image/tools/check holds oann's FLUX.2 klein to: diffusers and transformers on the CPU,
# F32 arithmetic on the weights as stored (BF16; each block is widened to F32 only while it runs, so 15 GB of RAM is
# enough). Three stages, each a process of its own, each writing OUT.<stage>.safetensors:
#   ref.py MODEL OUT text|flow|vae [PROMPT] [SIZE] [SEED] [STEPS]
#   text: the prompt's tokens and embeddings (and OUT.prompt.raw.txt, the prompt)
#   flow: the noise, the noise levels, each step's velocity and latents
#   vae:  the decoder's latents and image (and OUT.png)
# pip install torch diffusers (0.37 or later) transformers safetensors pillow
import sys, torch, numpy as np
from safetensors.torch import save_file, load_file
M, OUT, STAGE = sys.argv[1], sys.argv[2], sys.argv[3]
PROMPT = sys.argv[4] if len(sys.argv) > 4 else "A cat holding a sign that says hello world"
SIZE = int(sys.argv[5]) if len(sys.argv) > 5 else 256
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 0
STEPS = int(sys.argv[7]) if len(sys.argv) > 7 else 4
torch.set_grad_enabled(False)
bf = torch.bfloat16

def widen_while_running(blocks):
    for b in blocks:
        b.register_forward_pre_hook(lambda m, a: (m.float(), None)[1])
        b.register_forward_hook(lambda m, a, o: (m.to(bf), None)[1])

if STAGE == "text":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(M + "/tokenizer")
    te = AutoModelForCausalLM.from_pretrained(M + "/text_encoder", torch_dtype=bf)
    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    te.model.embed_tokens.register_forward_hook(lambda m, a, o: o.float())
    widen_while_running(te.model.layers)
    te.model.norm.float(); te.lm_head = torch.nn.Identity()
    out = te(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], output_hidden_states=True, use_cache=False)
    emb = torch.stack([out.hidden_states[k] for k in (9, 18, 27)], dim=1).float()
    emb = emb.permute(0, 2, 1, 3).reshape(1, 512, -1)
    open(OUT + ".prompt.raw.txt", "w").write(PROMPT)
    save_file({"ids": inp["input_ids"][0].to(torch.int32), "mask": inp["attention_mask"][0].to(torch.int32), "embeds": emb[0].contiguous()}, OUT + ".text.safetensors")
    print("text", text.__repr__(), "real tokens", int(inp["attention_mask"].sum()), emb.shape, float(emb.abs().mean()))

if STAGE == "flow":
    from diffusers import Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline, compute_empirical_mu, retrieve_timesteps
    tr = Flux2Transformer2DModel.from_pretrained(M + "/transformer", torch_dtype=bf)
    widen_while_running(list(tr.transformer_blocks) + list(tr.single_transformer_blocks))
    for m in (tr.time_guidance_embed, tr.double_stream_modulation_img, tr.double_stream_modulation_txt, tr.single_stream_modulation, tr.x_embedder, tr.context_embedder, tr.norm_out, tr.proj_out): m.float()
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(M + "/scheduler")
    emb = load_file(OUT + ".text.safetensors")["embeds"][None]
    h = w = 2 * (SIZE // 16)
    g = torch.Generator("cpu").manual_seed(SEED)
    lat = torch.randn((1, 128, h // 2, w // 2), generator=g, dtype=torch.float32)
    lat_ids = Flux2KleinPipeline._prepare_latent_ids(lat)
    txt_ids = Flux2KleinPipeline._prepare_text_ids(emb)
    lat = Flux2KleinPipeline._pack_latents(lat)
    dump = {"noise": lat[0].contiguous()}
    sig = np.linspace(1.0, 1 / STEPS, STEPS)
    mu = compute_empirical_mu(lat.shape[1], STEPS)
    ts, _ = retrieve_timesteps(sch, STEPS, "cpu", sigmas=sig, mu=mu)
    sch.set_begin_index(0)
    dump["sigmas"] = sch.sigmas.float().clone()
    print("mu", mu, "timesteps", ts, "sigmas", sch.sigmas)
    for i, t in enumerate(ts):
        pred = tr(hidden_states=lat, timestep=t.expand(1) / 1000, guidance=None, encoder_hidden_states=emb, txt_ids=txt_ids, img_ids=lat_ids, return_dict=False)[0]
        dump["pred%d" % i] = pred[0].contiguous()
        lat = sch.step(pred, t, lat, return_dict=False)[0]
        dump["lat%d" % i] = lat[0].contiguous()
        print("step", i, float(t), float(pred.abs().mean()), flush=True)
    save_file(dump, OUT + ".flow.safetensors")

if STAGE == "vae":
    from diffusers import AutoencoderKLFlux2
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
    from PIL import Image
    vae = AutoencoderKLFlux2.from_pretrained(M + "/vae", torch_dtype=torch.float32)
    lat = load_file(OUT + ".flow.safetensors")["lat%d" % (STEPS - 1)][None]
    hw = SIZE // 16
    ids = Flux2KleinPipeline._prepare_latent_ids(torch.zeros(1, 128, hw, hw))
    lat = Flux2KleinPipeline._unpack_latents_with_ids(lat, ids, hw, hw)
    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    z = Flux2KleinPipeline._unpatchify_latents(lat * std + mean)
    img = vae.decode(z, return_dict=False)[0]
    save_file({"z": z[0].contiguous(), "image": img[0].contiguous()}, OUT + ".vae.safetensors")
    px = ((img[0].permute(1, 2, 0) / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    Image.fromarray(px).save(OUT + ".png")
    print("image", img.shape, float(img.min()), float(img.max()))
