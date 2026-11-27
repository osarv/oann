# data

What a training loop reads from.

- `idx.olang` - the IDX file format MNIST is distributed in.
- `loader.olang` - `Dataset`, `Labeled` and `Loader<D>`: seeded random batches, filling a batch allocates nothing.
- `mnist.olang` - MNIST (and its family), fetched once with curl into `data/mnist` (git ignores it).
- `text.olang` - a text as characters: vocabulary, tokens, `Windows` over them; tiny Shakespeare is fetched into
  `data/shakespeare`.
- `parquet.olang` - Apache Parquet files read a column at a time (`Parquet.Open`, `Texts(group, column)`,
  `Numbers(group, column)`): the footer's Thrift metadata, PLAIN and dictionary pages, version 1 and 2, uncompressed or
  SNAPPY (`snappy.olang`), flat schemas - how Hugging Face publishes its datasets. A row group at a time, read with pread.
- `snappy.olang` - Snappy's raw block format decompressed (`Unsnappy`).
- `corpus.olang` - a pre-training corpus from Hugging Face datasets: the sources and their licences (`Sources`: FineWeb-Edu,
  FineMath, the Common Pile's openly licensed Stack-Edu), each source's files fetched one at a time, filtered (score at
  least 3, language), exact duplicates dropped, split into train, validation and test by the text's hash, tokenized and
  written as token files (below), with a resumable state per source. `examples/corpus.olang` is its command line.
- `tokens.olang` - a corpus of token ids in one binary file (little-endian `U16` or `U32`, nanoGPT's `.bin`), streamed:
  `Tokens.Open(path, ctx)`, then `Random` (random windows of `ctx + 1`, seeded) or `At` (an ordered pass) into the
  graph's token and target rows. The file stays on disk and a fill allocates nothing (the test measures resident memory
  over 20000 batches); a fill may run on a task while the previous batch trains.

Test with `olang test data/FILE.olang`; `make test` runs them all.
