# References for llm/vision: a Gemma 3 vision-language checkpoint DIR reading a fixed synthetic picture (shapes on a
# gradient, made here). Writes into DIR:
#   vision_test.png/.rgb the picture, and its raw 640 x 480 RGB bytes
#   vision_pixels.f32    the processor's pixel values, [3, 896, 896] F32 little-endian
#   vision_embeds.f32    the projector's output for them, [256, text width] F32 (the tower and projector in F32)
#   vision_ref.json      the prompt's ids (chat template, the image then "Describe this image in one sentence."), the
#                        last position's logits and twelve greedy tokens (the language model in BF16: 4B in F32 does not
#                        fit 15 GB)
#     python3 llm/tools/visionref.py DIR
import json, os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

d = sys.argv[1]
img = Image.new("RGB", (640, 480))
px = img.load()
for y in range(480):
    for x in range(640): px[x, y] = (x * 255 // 639, y * 255 // 479, 128)
dr = ImageDraw.Draw(img)
dr.ellipse((60, 60, 260, 260), fill=(220, 30, 30))
dr.rectangle((360, 200, 580, 420), fill=(20, 20, 200))
img.save(os.path.join(d, "vision_test.png"))
np.asarray(img, dtype=np.uint8).tofile(os.path.join(d, "vision_test.rgb"))   # 640 x 480 x 3, for the resize's check

proc = AutoProcessor.from_pretrained(d)
m = Gemma3ForConditionalGeneration.from_pretrained(d, dtype=torch.bfloat16).eval()
m.model.vision_tower.float()
m.model.multi_modal_projector.float()
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image in one sentence."}]}]
text = proc.apply_chat_template(msgs, add_generation_prompt=True)
inp = proc(text=text, images=[img], return_tensors="pt")
pix = inp["pixel_values"].float()
pix[0].numpy().astype("<f4").tofile(os.path.join(d, "vision_pixels.f32"))
with torch.no_grad():
    feats = m.model.get_image_features(pixel_values=pix)
    feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats
    feats = feats[0] if isinstance(feats, (list, tuple)) else feats
    feats.reshape(-1, feats.shape[-1]).float().numpy().astype("<f4").tofile(os.path.join(d, "vision_embeds.f32"))
    out = m(**{k: (v.to(torch.bfloat16) if k == "pixel_values" else v) for k, v in inp.items()})
    last = out.logits[0, -1].float()
    m.model.vision_tower.to(torch.bfloat16)
    m.model.multi_modal_projector.to(torch.bfloat16)
    gen = m.generate(**{k: (v.to(torch.bfloat16) if k == "pixel_values" else v) for k, v in inp.items()}, max_new_tokens=12, do_sample=False)[0, inp["input_ids"].shape[1]:]
ids = inp["input_ids"][0].tolist()
json.dump({"tokens": ids, "last": [round(x, 4) for x in last.tolist()], "greedy": gen.tolist()}, open(os.path.join(d, "vision_ref.json"), "w"))
print(len(ids), feats.shape, gen.tolist(), proc.decode(gen))
