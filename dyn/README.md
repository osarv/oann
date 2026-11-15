# dyn

The dynamical-system core of the redesign (docs/redesign.md 1.4; the model is docs/settling.md): a `System<T>` is a
tree of modules holding populations of neurons with named state, connections between them as objects, signals and
rules on clocks. It is recorded through handles, laid out once by `Plan` and then settled with nothing allocated.

| file | what |
|---|---|
| `system.olang` | `System<T>` (T the state's and the weights' element: F32 to run, F64 for checks), the handles `Pop`, `Conn`, `Mod`, `Sig`, `Clock`, `Rule`; recording (`Module`, `Population`, `Connect`, `Signal`, `Attach`), `Plan`, data in (`Clamp`, `Target`, `TargetValues`, `Set`), `Settle`, `Nudged`, `Reset`, `Warm`, state and parameters out (`State<C>`, `Activity`, `Predict`, `Loss`, `Weights`, `Bias`, `Params`, `ParamGrads`, `Read`), structure while running (`Prune`, `Retire`, `Compact`) and stability (`Gain`, `Certify`, `Restrain`, `Rate`, `LessonTolerance`, `BudgetFor`) |
| `neuron.olang` | the cells (`RateCell`, `LifCell`, `@Keep` for state kept across settles), `Rho`, `Neuron` (rate, spiking, `Custom` through the `Dynamics` trait), `Role`, `Cost`, and the per-element work of a sweep: the residual and step, a spiking time step and window, a readout's force and cost |
| `conn.olang` | `Wiring` (Dense, Symmetric, Csr, Tiles, Events, LowRank) and `Link<T>`: its weights as a view of the parameter region, carrying activities forward and back, its share of the units' incoming mass, pruning (`Prune`, `MaskGrads`) and compaction in place |
| `settle.olang` | `Phase`, `Budget`, `Settlement` and the `Engine`: population state double-buffered as `Columns` of cells, every phase side by side on one axis, Jacobi sweeps cut into units (a population's rows of a phase, 128 at most) run on the system's tasks - the same bit for bit for any thread count |

Programs import the four modules (`import "dyn/system"` and so on) and write `system.System<F32>(batch, threads)`,
`neuron.Neuron.Rate(neuron.Rho.Tanh)`, `conn.Wiring.Symmetric`, `settle.Budget()`. Learning rules, energies and
implicit gradients are `dyn/learn.olang` and `dyn/energy.olang` (B2b).

Tests: `olang test dyn/FILE.olang` (`make test` runs them). Speed against `circuit.olang`, the same network and
weights side by side: `make settlebench ARGS="ROUNDS THREADS"` (bench/settle.olang).
