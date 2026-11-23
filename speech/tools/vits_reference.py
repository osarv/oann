# speech/tools/vits_reference.py - the references speech/vits is held to, with no noise (noise_scale and the duration
# predictor's 0) so they are deterministic: Hugging Face transformers' VitsModel (F32, CPU) on kakao-enterprise/vits-ljs,
# and onnxruntime on Piper's voices (rhasspy/piper-voices en_US-ljspeech-medium, sv_SE-nst-medium). Each takes IPA
# strings, mapped to ids by hand as the tokenizers do (no phonemizer, no espeak): vits-ljs's vocab.json with a blank (0)
# around every character, Piper's phoneme_id_map with "^", "_" after every phoneme, "$". Writes into OUT/<model>/:
# cases.json (each case's IPA and ids), <case>.dur.f32 (every id's frames, the ceiled durations) and <case>.wav.f32
# (the waveform); for vits-ljs also <case>.wav64.f32, the waveform of the model run in F64 (rounded to F32), since
# transformers' own F32 waveform is up to 2.2e-4 from it. Needs torch, transformers, onnx, onnxruntime, numpy; run with python3 -I:
#
#     python3 -I speech/tools/vits_reference.py MODELS_DIR OUT
#
# MODELS_DIR holds kakao-enterprise--vits-ljs/ (config.json, vocab.json, model.safetensors) and piper/ (<voice>.onnx,
# <voice>.onnx.json); a model whose files are absent is skipped.
import json, os, sys
import numpy as np

models, out = sys.argv[1], sys.argv[2]

CASES = {
    "vits-ljs": {
        "hello": "həlˈoʊ wˈɜːld.",
        "fox": "ðə kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ.",
        "genesis": "ɪn ðə bɪɡˈɪnɪŋ, ɡˈɑːd kɹiːˈeɪɾᵻd ðə hˈɛvən ænd ðɪ ˈɜːθ.",
    },
    "en_US-ljspeech-medium": {
        "hello": "həlˈoʊ wˈɜːld.",
        "fox": "ðə kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ.",
    },
    "sv_SE-nst-medium": {
        "hej": "hɛj vˈɛːɖɛn.",
        "hund": "dɛt ˈɛːr ɛn vˈakɛr hˈɵnd.",
    },
}


def write(name, cases):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    json.dump(cases, open(os.path.join(d, "cases.json"), "w"), ensure_ascii=False, indent=1)
    return d


def vits_ljs():
    import torch
    from transformers import VitsModel
    src = os.path.join(models, "kakao-enterprise--vits-ljs")
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        return print("vits-ljs: no model, skipped")
    vocab = json.load(open(os.path.join(src, "vocab.json")))
    model = VitsModel.from_pretrained(src, torch_dtype=torch.float32).eval()
    model.noise_scale = 0.0
    model.noise_scale_duration = 0.0
    exact = VitsModel.from_pretrained(src, torch_dtype=torch.float32).double().eval()
    exact.noise_scale = 0.0
    exact.noise_scale_duration = 0.0
    seen = {}
    model.duration_predictor.register_forward_hook(lambda m, i, o: seen.__setitem__("logw", o))
    cases = {}
    for key, ipa in CASES["vits-ljs"].items():
        ids = [0]
        for ch in ipa:
            ids += [vocab[ch], 0]
        cases[key] = {"ipa": ipa, "ids": ids}
    d = write("vits-ljs", cases)
    torch.manual_seed(0)
    for key, c in cases.items():
        with torch.no_grad():
            wav = model(torch.tensor([c["ids"]])).waveform[0]
            wav64 = exact(torch.tensor([c["ids"]])).waveform[0]
        # the forward's own durations: ceil(exp(log_duration) * mask * length_scale), the mask and the scale 1
        dur = torch.ceil(torch.exp(seen["logw"]) * 1.0 * 1.0)[0, 0]
        dur.numpy().astype(np.float32).tofile(os.path.join(d, key + ".dur.f32"))
        wav.numpy().astype(np.float32).tofile(os.path.join(d, key + ".wav.f32"))
        wav64.numpy().astype(np.float32).tofile(os.path.join(d, key + ".wav64.f32"))
        print("vits-ljs", key, len(c["ids"]), "ids,", int(dur.sum()), "frames,", wav.shape[0], "samples")


def piper(voice):
    import onnx, onnxruntime
    src = os.path.join(models, "piper", voice + ".onnx")
    if not os.path.exists(src):
        return print(voice + ": no model, skipped")
    conf = json.load(open(src + ".json"))
    idmap = conf["phoneme_id_map"]
    g = onnx.load(src)
    # the durations are the Ceil's output: made an output of the graph too
    ceil = [n for n in g.graph.node if n.op_type == "Ceil"][0].output[0]
    g.graph.output.append(onnx.helper.make_tensor_value_info(ceil, onnx.TensorProto.FLOAT, None))
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = onnxruntime.InferenceSession(g.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    cases = {}
    for key, ipa in CASES[voice].items():
        ids = idmap["^"] + idmap["_"]
        for ch in ipa:
            ids += idmap[ch] + idmap["_"]
        ids += idmap["$"]
        cases[key] = {"ipa": ipa, "ids": ids}
    d = write(voice, cases)
    for key, c in cases.items():
        x = np.array([c["ids"]], dtype=np.int64)
        wav, dur = sess.run(None, {"input": x, "input_lengths": np.array([x.shape[1]], dtype=np.int64),
                                   "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32)})
        dur = dur.reshape(-1).astype(np.float32)
        wav = wav.reshape(-1).astype(np.float32)
        dur.tofile(os.path.join(d, key + ".dur.f32"))
        wav.tofile(os.path.join(d, key + ".wav.f32"))
        print(voice, key, len(c["ids"]), "ids,", int(dur.sum()), "frames,", wav.shape[0], "samples")


vits_ljs()
piper("en_US-ljspeech-medium")
piper("sv_SE-nst-medium")
