# surgery - changing a trained model instead of retraining it

Train a model now, and later make it deeper, wider, longer-sighted, smaller, or a different kind, or switch its
optimizer, without relearning what it knows. A model is a `model.State`, which holds:

- its architecture (`Spec`, block by block);
- its parameters and optimizer moments by name;
- the optimizer's steps, the training step, a warmup restart and the data position;
- a lineage: the parent checkpoint and every operation applied since.

`State.Save` writes all of it into one safetensors file, and `State.Open` reads it back, recording the file as the
parent. Operations in `ops` turn one State into another. `model.Net.Build` puts a State on a graph, and
`CaptureAdamW`/`CaptureMuon` read the trained parameters and moments back.

```olang
st := model.State.Fresh(model.Spec.Uniform(vocab, 64, 64, 2, 2), seed)
net := model.Net.Build(st, batch)
o := net.AdamW(st)
...train...
st = net.CaptureAdamW(st, o)
try st.Save("run/small.safetensors")

st = try model.State.Open("run/small.safetensors")
st = ops.Deepen(st, 1, block, seed)           # each keeps the function
st = ops.WidenFfn(st, 0, 512, seed)
st = ops.AddHeads(st, 0, 2, seed)
st = ops.WidenDim(st, 128, seed)
st.Warm = st.Step                             # rate := model.Rewarm(schedule, st.Warm, 50)
net = model.Net.Build(st, batch)
o = net.AdamW(st)                             # moments carried; new parts start at zero
```

## The model

The model has a token embedding (plus learned positions when Rope is off), then pre-norm blocks, a final norm, and
logits computed against the embedding (tied) or a head of its own. Each block is described separately:

- `Heads` × `HeadDim` (the attention width need not equal the model width);
- `Hidden` feed-forward units, GELU or gated (SwiGLU);
- an attention `Window` (0 means the whole context).

Linear layers have no biases. Norms are RMSNorm, or LayerNorm. Parameters have names such as `embed`, `b2.q`,
`b2.down`, `final.g`, and moments use the same names with an `m.`, `v.` or `buf.` prefix - and `outer.` for DiLoCo's
outer momentum (`nn/dist`), whose states are States: `Net.Layout()` and `State.DistMeta()` make a DiLoCo write them,
the operations move the outer momentum as they move a first moment, and `Net.Outer(st)` gives it back for
`DiLoCo.Publish` to seed the grown model's run (README "Distributed training").

## Operations

| operation | keeps the function? | what it does |
|---|---|---|
| `Deepen(st, at, block, seed)` | exactly | inserts a block whose `o` and `down` are zero |
| `WidenFfn(st, i, hidden, seed, noise)` | exactly when noise is 0 | widens the feed-forward the Net2Net way: units are copied and their outgoing weights divided; noise (default 0.01) breaks the symmetry |
| `AddHeads(st, i, n, seed)` | exactly | adds heads with drawn q, k, v and zero o columns |
| `WidenDim(st, dim, seed)` | exactly with RMSNorm | widens the model; new stream dimensions start at zero, gains are scaled by sqrt(old/new) and eps by old/new. With LayerNorm it is close but not exact |
| `GrowVocab(st, vocab, seed)` | old tokens' logits exactly | grows the vocabulary; new rows start at the mean row |
| `LengthenContext(st, ctx, ntk)` | old positions exactly (without ntk) | lengthens the context; RoPE keeps its base or scales it NTK-aware, learned positions are extended |
| `DropBlock`, `DropLeastBlock` | no | removes a block; `DropLeastBlock` removes the least important by `BlockImportance` (the loss added by removing it) |
| `DropHeads`, `PruneHeads` | no | removes heads; `PruneHeads` keeps the most important by `HeadImportance` |
| `PruneFfn(st, i, hidden)` | no | keeps the units with the largest \|up\|·\|down\| |
| `Replace(st, i, block, ids, steps)` | approximately, after distilling | switches a block to another kind (window, feed-forward kind, heads, width), copying parameters where shapes match and distilling the block onto the old one's outputs |
| `ToMuon`, `ToAdamW`, `ResetOptimizer`, `model.Rewarm` | n/a | change the optimizer; Muon's momentum comes from AdamW's first moment and back, a reset restarts the warmup, and the schedule's step carries on |

Every growth operation is tested to leave the logits unchanged, to within 2e-4. A distilled replacement is tested to
end closer to the old model's loss than an undistilled one.

## Measured

`bench/grow` (`make grow`) runs on tiny Shakespeare at equal compute. It trains a 2-block, width-64 model for a share
of the budget, grows it to a 4-block, width-128 model, and trains on. This is compared with training the large model
from scratch for the whole budget. The numbers are in `PROGRESS.md`.
