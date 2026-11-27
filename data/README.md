# data

What a training loop reads from.

- `idx.olang` - the IDX file format MNIST is distributed in.
- `loader.olang` - `Dataset`, `Labeled` and `Loader<D>`: seeded random batches, filling a batch allocates nothing.
- `mnist.olang` - MNIST (and its family), fetched once with curl into `data/mnist` (git ignores it).
- `text.olang` - a text as characters: vocabulary, tokens, `Windows` over them; tiny Shakespeare is fetched into
  `data/shakespeare`.
- `tokens.olang` - a corpus of token ids in one binary file (little-endian `U16` or `U32`, nanoGPT's `.bin`), streamed:
  `Tokens.Open(path, ctx)`, then `Random` (random windows of `ctx + 1`, seeded) or `At` (an ordered pass) into the
  graph's token and target rows. The file stays on disk and a fill allocates nothing (the test measures resident memory
  over 20000 batches); a fill may run on a task while the previous batch trains.
- `chat.olang` - chat conversations for fine-tuning: a JSON-lines file (the OpenAI `messages` shape or ShareGPT's
  `conversations`) rendered through Qwen3's chat template, tokenized with llm/bpe, laid out in rows of a context
  (packed, or one conversation a row padded with `<|endoftext|>`), each token's class the next token on the assistant
  turns (their text and `<|im_end|>`) and -1 elsewhere. `ChatData.Load(path, tokenizer, ctx, valid, seed)` carves a
  validation share of whole conversations out of the file by the seed; `ChatRows.Random` walks the training rows in a
  new seeded order every epoch, `At` in the file's order, and `Save`/`Load` put where the reading stands into a
  checkpoint. `testdata/chat.jsonl` holds the tests' conversations.

Test with `olang test data/FILE.olang`; `make test` runs them all.
