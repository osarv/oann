# speech/tools/reference.py - the reference a speech test holds oann's Whisper to: Hugging Face transformers (F32, CPU)
# on a clip. Writes, for MODEL_DIR and CLIP.wav, into OUT/: CLIP.mel.f32 (the log-mel, 80 x 3000), CLIP.logits.f32
# (the decoder's logits at every position of the greedy sequence) and CLIP.json (the greedy, timestamped and beam
# sequences and their text, and the segments with their times); a clip longer than 30 s gets only its segments. Needs torch, transformers, numpy; run with python3 -I.
# MODEL_DIR needs preprocessor_config.json too (openai/whisper-*'s, not in a bare weights directory).
#
# With a prompt - reference.py MODEL_DIR CLIP.wav OUT --prompt LANG TEXT [TEXT ...] - writes OUT/CLIP.prompt.json instead:
# the greedy tokens generate(..., language=LANG, prompt_ids=processor.get_prompt_ids(TEXT)) writes (the decoder's prompt
# not in them) for no prompt and for each TEXT, with the prompt ids (<|startofprev|> and the tokens of " " + TEXT).
import json, sys, wave
import numpy as np, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_dir, clip, out = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = sys.argv[6:] if len(sys.argv) > 5 and sys.argv[4] == "--prompt" else None
torch.set_num_threads(4)
with wave.open(clip) as w:
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
proc = WhisperProcessor.from_pretrained(model_dir)
model = WhisperForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
name = clip.rsplit("/", 1)[-1].rsplit(".", 1)[0]

def segments(feats, **kw):
    with torch.no_grad():
        out = model.generate(feats, return_timestamps=True, return_segments=True, **kw)
    return [{"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2), "text": proc.tokenizer.decode(s["tokens"], skip_special_tokens=True)}
            for s in out["segments"][0]]

if prompts is not None:
    lang = sys.argv[5]
    feats = proc.feature_extractor(audio[:480000], sampling_rate=16000, return_tensors="pt").input_features
    cases = []
    for text in [""] + prompts:
        kw = dict(prompt_ids=torch.tensor(proc.get_prompt_ids(text))) if text else {}
        with torch.no_grad():
            seq = model.generate(feats, language=lang, task="transcribe", max_new_tokens=200, **kw)[0].tolist()
        cases.append({"prompt": text, "prompt_ids": [int(i) for i in proc.get_prompt_ids(text)] if text else [], "tokens": seq,
                      "text": proc.tokenizer.decode(seq, skip_special_tokens=True)})
        print(name, lang, repr(text), "->", cases[-1]["text"])
    json.dump({"language": lang, "cases": cases}, open(f"{out}/{name}.prompt.json", "w"), indent=1, ensure_ascii=False)
    sys.exit(0)
if len(audio) > 480000:
    # a long clip: only its segments, read window after window
    f = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True)
    res = {"segments": segments(f.input_features, attention_mask=f.attention_mask)}
    json.dump(res, open(f"{out}/{name}.json", "w"), indent=1)
    print(name, " | ".join(s["text"] for s in res["segments"]))
    sys.exit(0)
feats = proc.feature_extractor(audio[:480000], sampling_rate=16000, return_tensors="pt").input_features
feats[0].numpy().astype(np.float32).tofile(f"{out}/{name}.mel.f32")
res = {}
with torch.no_grad():
    for key, kw in [("greedy", dict()), ("timestamps", dict(return_timestamps=True)), ("beam", dict(num_beams=5))]:
        ids = model.generate(feats, max_new_tokens=200, **kw)
        if not torch.is_tensor(ids): ids = ids["sequences"]
        seq = ids[0].tolist()
        res[key] = {"tokens": seq, "text": proc.tokenizer.decode(seq, skip_special_tokens=True)}
    enc0 = model.model.encoder(feats).last_hidden_state
    lang = model(encoder_outputs=(enc0,), decoder_input_ids=torch.tensor([[50258]])).logits[0, -1]
    ids = list(model.generation_config.lang_to_id.values())
    langid = max(ids, key=lambda i: lang[i].item())
    res["language"] = langid
    prefix = [50258, langid, 50359, 50363]
    res["greedy"]["tokens"] = prefix + res["greedy"]["tokens"]
    res["beam"]["tokens"] = prefix + res["beam"]["tokens"]
    res["timestamps"]["tokens"] = prefix[:3] + res["timestamps"]["tokens"]
    seq = torch.tensor([res["greedy"]["tokens"]])
    enc = model.model.encoder(feats).last_hidden_state
    logits = model(encoder_outputs=(enc,), decoder_input_ids=seq).logits[0]
    logits.numpy().astype(np.float32).tofile(f"{out}/{name}.logits.f32")
    res["logits_shape"] = list(logits.shape)
res["segments"] = segments(feats)
json.dump(res, open(f"{out}/{name}.json", "w"), indent=1)
print(name, res["greedy"]["text"])
