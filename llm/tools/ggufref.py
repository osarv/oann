# References for llm/gguf: for each GGUF file named, transformers' own reading of it (dequantized to F32): the last
# position's logits for "The capital of France is" and ten greedy tokens, into FILE.ref.json; and the first two rows of
# one tensor of each quantization type the file holds, as gguf-py dequantizes them, into FILE.deq.json.
#     python3 llm/tools/ggufref.py $OANN_MODELS/gguf/*.gguf
import json, os, sys
import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from transformers import AutoModelForCausalLM, AutoTokenizer

for path in sys.argv[1:]:
    d, name = os.path.dirname(path), os.path.basename(path)
    r = GGUFReader(path)
    seen, deq = set(), {}
    for t in r.tensors:
        if t.tensor_type.name in seen or len(t.shape) < 2: continue
        seen.add(t.tensor_type.name)
        rows = dequantize(t.data, t.tensor_type).reshape(-1, int(t.shape[0]))[:2]
        deq[t.name] = {"type": t.tensor_type.name, "rows": [[round(float(x), 6) for x in row] for row in rows]}
    json.dump(deq, open(path + ".deq.json", "w"))
    tok = AutoTokenizer.from_pretrained(d, gguf_file=name)
    ids = tok("The capital of France is", add_special_tokens=False, return_tensors="pt").input_ids
    m = AutoModelForCausalLM.from_pretrained(d, gguf_file=name, dtype=torch.float32).eval()
    with torch.no_grad():
        last = m(ids).logits[0, -1]
        out = m.generate(ids, max_new_tokens=10, do_sample=False)[0, ids.shape[1]:]
    json.dump({"tokens": ids[0].tolist(), "last": [round(x, 5) for x in last.tolist()], "greedy": out.tolist()},
              open(path + ".ref.json", "w"))
    print(name, sorted(seen), ids[0].tolist(), out.tolist(), tok.decode(out))
    del m
