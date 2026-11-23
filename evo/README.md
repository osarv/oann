# evo - evolutionary architecture search

Evolutionary algorithms for finding neural architectures (and their hyperparameters): a search space, a population,
selection, and an evaluation that trains each candidate. The package knows no model. A genome decodes to a network
through the program's own code, and the program's fitness trains and scores that network.

| module | what it holds |
|---|---|
| `schema.olang` | `Schema`: a search space of named genes. A gene is a choice, an integer or real range (log scale optional), or a block repeated a variable number of times. A gene can be active only when another gene holds a given value. `Genome` is one point of the space. |
| `neat.olang` | `Space` and `Net`: graph genomes with historical markings (NEAT). Nodes carry an operation and a width, and edges can be switched off. Includes a topological order and a registry of markings. |
| `search.olang` | `Evolution<G>`: the search. Regularized (aging) evolution, tournament, (mu + lambda) and NSGA-II, with successive halving, weight-inheritance paths, a result cache, checkpoints and worker processes. Also the fitness contract (`Trial`, `Result`, `Split`). |
| `pareto.olang` | non-dominated sorting and crowding distance (every objective minimized) |
| `inherit.olang` | copying weights between networks: `Overlap` (where shapes match), `Widen` and `Identity` (function-preserving widen and deepen) |

## A search space

```olang
s := schema.Schema()
layer := s.Block("layer")
s.Choice(layer, "op", Array<String>["mlp", "attention", "conv"])
s.Int(layer, "width", 16, 512, true)          # true: a size, which width scaling doubles or halves
s.Int(layer, "heads", 1, 8)
try s.When(layer, "heads", "op", "attention")  # heads counts only where op is attention
s.Float(schema.Root, "lr", 1e-4, 1e-2, true)   # true: log scale
s.Repeat(schema.Root, "layers", layer, 1, 8)   # 1 to 8 layers
```

Decoding reads genes by name:

- `s.Children(g, i, "layers")` gives the instances a repeat holds.
- `s.Num(g, i, "width")` gives a number.
- `s.Pick(g, i, "op")` gives a choice's option.

The root instance is `schema.Root`.

The mutations (`Schema.Rates` weights them) are:

- **point:** a gene drawn afresh or perturbed;
- **swap:** a choice set to another option;
- **width:** a size doubled or halved;
- **insert:** a block added, copied from a sibling or drawn fresh;
- **delete:** a block removed.

`Cross` is uniform over aligned genes: each gene comes from either parent, and the children of a repeat are aligned by position. `Key` is the canonical text of the active genes. Two genomes that decode alike share a key, which is what the result cache uses.

## Graph genomes

`neat.Space(inputs, outputs, ops, minWidth, maxWidth)` starts each network with every input joined to every output.

The mutations are:

- add a node, splitting an edge;
- add an edge, keeping the graph acyclic;
- delete a hidden node;
- swap an operation;
- scale a width;
- a point mutation: toggle an edge or redraw a width.

Every structural change is named by the space's registry. The same edge gets the same marking, and the same split gets the same node id, for the whole search, so crossover aligns genes by marking. A child is a subgraph of its fitter parent and stays acyclic. Decode a net in `n.Order()`, reading each node's inputs with `n.Into(i)`.

## The fitness contract

A fitness is any type with:

```olang
fn (f mut Fit&) Evaluate(g G&, t search.Trial&, data search.Split<D>&) search.Result ? E
```

The call does four things:

- builds the network that `g` decodes to;
- trains it on `data.Train` for `t.Budget`, seeded by `t.Seed`;
- returns the loss on `data.Val` in `Result.Loss`;
- returns any costs in `Result.Costs`, for example MACs, parameters or latency.

Any error marks the genome failed, and the search drops it. The budget's unit is the program's own: seconds, steps or epochs. For "train N seconds on one thread, return the validation loss and the TMACs", set `Rungs` to seconds and push the TMACs into `Costs`.

`Split` holds training and validation data and nothing else. The search never receives test data, so no selection, halving or choice of the best can read it. Evaluate the final genome on the test set yourself, once. `Split<Array<T>>.Carve(items, share, seed)` takes a validation share out of training data.

## Running a search

```olang
c := search.Config(search.Algorithm.Regularized(64, 16), seed)  # population 64, sample 16
c.Rungs = Vec<F64>[100.0, 300.0, 900.0]   # successive halving over budgets; the last is the full one
c.Batch = 27                              # children per round, halved by Eta (3) at each rung
c.Crossover = 0.2
c.Workers = 4                             # one process per trial, up to four at once
c.Dir = "runs/arch"                       # weights and the checkpoint
e := search.Evolution<schema.Genome>(c)
if os.Exists(c.Dir + "/state.evo"): try e.Resume(space, try os.ReadFile(c.Dir + "/state.evo"))
e.Run(space, fit, split, rounds)
best := e.Best()      # lowest validation loss at the full budget
front := e.Front()    # loss against the first Objectives - 1 costs, non-dominated
```

### Seeding

Call these before the first round:

- `e.Seed(g)` puts genome `g` in the first population. It is evaluated at the full budget with the rest of that population, and random genomes fill the remaining places.
- `e.Add(space, g, Result(loss))` adds a genome already evaluated elsewhere. It gets the next id, joins the population and the history, and its result is cached, so it is never trained again.

### Algorithms

| algorithm | how one round works |
|---|---|
| `Regularized(population, sample)` | The default. Takes the best of a random sample, mutates it, and the oldest member leaves. |
| `Tournament(population, size, elite)` | Generational: parents are chosen by tournaments, and the elite survive. |
| `MuPlusLambda(mu, lambda)` | Keeps the best `mu` of parents and children together. |
| `Nsga2(population)` | Selects by front, then by crowding distance, over `Config.Objectives` objectives: the loss, then the costs. |

Crossover takes the fitter parent first and is followed by a mutation.

### Multi-fidelity

Each round's children go through the rungs. The best `1/Eta` of each rung move on, and only the children that reach the last rung join the population and the history. The first population is evaluated at the full budget.

### Weight inheritance

With `Dir` set, a trial is told where things are:

- `Save`: where to write its weights;
- `Inherit`: its parent's weights;
- `Resume` and `Spent`: past the first rung, its own weights from the rung before and the budget they were trained for.

Copy them with `inherit.Overlap`, tensor by tensor where shapes match. Where the child differs from its parent by one layer's width, or by one added layer, use `inherit.Widen` or `inherit.Identity` so it starts with exactly its parent's function. Only the population's weights and the best member's are kept.

### Cache

Results are cached by `Key` and rung, so a genome that appears again is not trained again.

### Checkpoints

With `Dir` set, the whole state is written to `Dir/state.evo` after every round: the generator, the population, the history, the cache, and the space's registry. `Resume` reads it back, and the run then continues exactly as if it had never stopped.

### Workers

With `Workers > 1`, the program runs each trial in a new process of itself, started as `PROGRAM --evo-worker JOB ARGS...`. Up to `Workers` processes run at once. The program must do three things, in this order:

1. read its arguments with `search.Args()`, which gives the same arguments in either kind of process;
2. load its data;
3. call `search.Worker(space, fit, split)` before searching. In a worker process this call runs the job and ends the process.

A trial's seed depends only on the run's seed, the trial id and the rung. Results are taken in id order, so a run gives the same result for any number of workers.

## Advice

`RunWith(space, fit, split, adv, rounds)` runs the same search with an advisor: archnet's `Advisor`, or any type with
`Ready`, `Observe(g, rung, curve, loss)`, `Acquire(g)`, `Forecast(g, curve)` and `Propose(r)`. Three settings turn
its parts on:

- `Config.Screen`: each child is the most promising of this many candidates, outside the `Explore` share (default 32).
- `Config.Proposals`: this share of children comes from the advisor's generator, once the advisor is `Ready` (default 0.3).

The defaults are archnet's best on its toys: final gap to the optimum 0.17 against 0.20 for plain evolution, and on
chain landscapes 0.06 against 0.17 (16 genes) and 0.41 against 0.88 (32 genes); see archnet/README.md.
- `Config.Forecast`: halving ranks by forecast final loss.

Every result at every rung is passed to the advisor, and each member records how it was made in `From` (drawn,
mutated, crossed or proposed). evo has no other dependency on archnet: `Run` passes a no-op advisor. After `Resume`,
`Replay(adv)` rebuilds an advisor from the history.

## Examples

`make evolm ARGS="rounds workers seed dir"` runs `examples/evo_lm.olang`. It searches tiny character language models (`models/gpt`: depth, width, heads, window, feed-forward and norm kinds, learning rate) on tiny Shakespeare. Rungs are 60, 180 and 540 steps, and trials resume across rungs. It reports the validation loss, then the training TMACs and parameter count as costs.


`make evomlp ARGS="rounds workers seed dir"` runs `examples/evo_mlp.olang`. It searches multilayer perceptrons for a synthetic regression and uses every part of the package. It prints the best genome, its test loss (the only time the test set is read), the loss-against-MACs front, and a digest of the history that is equal for any number of workers.
