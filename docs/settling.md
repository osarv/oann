# Settling networks in oann

A settling network is a network of reciprocally connected regions whose neuron potentials iterate to an equilibrium.
The answer to an input is the fixed point the potentials settle to, or a refusal when they do not settle. It learns
from local rules that contrast two settlings instead of differentiating through the iterations; reward arrives as one
broadcast scalar; a fast memory and a persistent memory sit beside the weights; an arousal signal decides when
learning and memory writes happen at all; and an offline phase consolidates what the fast memory holds into slow
weights. This document (1) states the model - equations, convergence conditions and costs, (2) designs how oann
supports it beside its backprop graph (DESIGN.md), (3) lists what the matrix library and olang must provide, (4) maps
it onto the target board (PYNQ-Z2), (5) says how it is validated, and (6) records the decisions taken and the questions
still open.

**Provenance.** The model is assembled from published work, cited where each mechanism is introduced and listed under
References: Hopfield-style graded-response relaxation, equilibrium propagation and its centered variant,
eligibility traces with temporal-difference learning, reward-modulated three-factor rules, delta-rule associative
memory, complementary learning systems, and neuromodulator-gated plasticity. Each constant is a value from that
literature, a value derived here from a stated condition, or oann's own choice, and which of the three is said where
it is introduced. All are defaults for the tests of section 5 to tune,
not measured optima. Figures marked *planning* are estimates from the arithmetic shown, to be replaced by
measurement.

---

## 1. The model

### 1.1 Parts and sources

| part | computation | learning | sources |
|---|---|---|---|
| **Settling circuit** | rate neurons relax to a fixed point over a directed graph of regions | contrasts of free and nudged phases (equilibrium propagation) | Hopfield 1984; Scellier & Bengio 2017; Laborieux et al. 2021 |
| **Agent** (a life) | one circuit plus a context trace, a fast/persistent associative memory and an arousal law, acting continuously | the above with reward: eligibility traces, a critic, a broadcast TD error; memory writes; all gated by arousal | Sutton & Barto 2018; Barto, Sutton & Anderson 1983; Izhikevich 2007; Frémaux & Gerstner 2016; McClelland et al. 1995; Pearce & Hall 1980 |
| **Consolidator** (sleep) | a gated linear recurrent cell with a sparse-code store | slow weights by backpropagation through time; store by delta rule; offline replay | Marr 1969; Albus 1971; McClelland et al. 1995; Robins 1995 |

Sections 1.2-1.8 and 1.10 describe the first two rows, 1.9 the third.

### 1.2 The neuron and the settle (Hopfield 1984; Scellier & Bengio 2017)

Each neuron `i` has a potential `v[i]` and an activation `s[i] = rho(v[i])`. The continuous dynamics are a leaky
integrator, `dv/dt = -v + total`, stepped by forward Euler with `dt` in `(0, 1]`; for every neuron at once:

```text
total[i] = sum_{e: post[e]=i} W[e] s[pre[e]]  +  d[i]  +  b[i]   (+ nudge[i] in a nudged phase)
v[i]    <- v[i] + dt (total[i] - v[i])
s[i]     = rho(v[i])
```

`W[e]` is the weight of synapse `e`, `b` a bias, and `d` the drive: the observation on an input neuron, a memory read on
a readout neuron. A drive is added to the total, it does not clamp the potential. `dt = 1` is plain fixed-point
iteration `v <- total` (a synchronous Hopfield update) and has the smallest contraction rate (below); it is oann's
default, with `dt` halved when damping is needed.

**Activation.** oann's default is `rho = tanh` (Hopfield's graded response; rest at 0, slope at most 1, smooth, so the
learning theorems below apply without a kink). The neuron model is a closed enum: the hard sigmoid `clip(v, 0, 1)` of
Scellier & Bengio is the second member, and a linear `rho(v) = v` exists for tests, where the equilibrium has a closed
form.

**What "settled" means.** The residual is `r = total - v`, evaluated at the state and including the nudge of a nudged
phase. A state **qualifies** when `max_i |r[i]| <= tau` in every batch row. A small movement of the activation is not a
certificate - saturation can stall movement while the residual is large. A settle advances all rows together and has a
sweep budget. An answer that does not qualify within the budget is a **refusal**: nothing is committed and no action is
issued. Damping splits the budget into attempts with `dt` halved each time, always testing the undamped residual; it
changes the path, not the fixed points, and it removes oscillation (a period-2 cycle of the undamped iteration). It
cannot create an equilibrium that does not exist, and an equilibrium that is unstable for the continuous dynamics never
attracts however small `dt` is.

**When it converges** (the contraction argument). Let `L = max rho'` (1 for `tanh` and the hard sigmoid), `rho_W` the
largest absolute incoming weight sum of any neuron (its **incoming mass**, counting every recurrent block that feeds
it - a block whose source is an input region is folded into the constant `c`, and a constant does not enter the
map's Lipschitz constant; *decided at implementation*, section 6), and `kappa = L rho_W` the feedback gain. If
`kappa < 1` the map `T(v) = c + W rho(v)` is a contraction in the sup norm, so there is one equilibrium `v*`, and:

```text
step rate             q = 1 - dt (1 - kappa)                 (q = kappa at dt = 1)
residual              ||r_k|| <= q^k ||r_0||
distance to v*        ||v_k - v*|| <= q^k / (1 - q) ||v_1 - v_0||
at the stopping rule  ||v - v*|| <= tau / (1 - kappa)         (from ||v - v*|| <= ||r|| + kappa ||v - v*||)
sweeps to tolerance   k >= ln(tau / ||r_0||) / ln q
```

For a cold start (`||r_0|| ~ 1`) and `tau = 1e-3`, the bound gives:

| `q` | 0.5 | 0.7 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|
| sweeps | 10 | 20 | 31 | 66 | 135 |

A warm start (the previous equilibrium under a changed drive) begins with a smaller `r_0` and saves
`ln(1/||r_0||) / ln(1/q)` sweeps. The condition is sufficient, not necessary: a trained circuit may leave the certified
region (`kappa >= 1`), and then only the residual test and the refusal remain. Since `rho_W < 1` bounds every single
recurrent weight, a certified circuit has `|W[e]| < 1` on every recurrent synapse - a fact the hardware mapping uses
(section 4); the folded input blocks are not bounded by the certificate, so the board's range argument (4.2) needs
its own bound on them, the incoming mass with the inputs counted (phase 3).

**A nudged phase** adds the force's own feedback. The force `beta f(s)` on a readout neuron falls as its activity
rises - by `p (1 - p) / T <= 1 / (4 T)` per unit of `s` for the cross-entropy force, by 1 for the squared error - so a
phase nudged with `beta > 0` has a negative self-feedback of up to `beta lambda` (`lambda = 1 / (4 T)` or 1) beside
the circuit's gain. At a full step it overshoots: the sweep's Jacobian gets diagonal entries down to `-beta lambda`,
and where `beta lambda` nears 1 the iteration swings in a period-2 cycle (measured: at `beta = 0.5`, `T = 0.25` a
plus phase took 627 sweeps at `dt = 1` and 27 at `dt = 0.75`). *Decided at implementation*: a phase nudged with
`beta > 0` steps at `dt_beta = dt / (1 + beta lambda)`, which keeps every diagonal entry of the sweep's Jacobian
nonnegative; the nudge then only damps, and the sweep contracts at `q_beta = 1 - dt_beta (1 - kappa)` - exactly for
the squared error (checked), and for the cross-entropy up to the force's off-diagonal terms, which cancel its diagonal
ones while the readout's slopes `rho'` are equal and add at most `beta (1 - rho') / (4 T)` otherwise. A phase nudged
with `beta < 0` pushes away from the target, a positive self-feedback, and keeps `dt`.

**Energy** (Hopfield 1984). With symmetric effective weights, no adaptation and a monotone `rho`, the continuous
dynamics descend

```text
E(v) = sum_i int_0^{v_i} u rho'(u) du  -  1/2 sum_{i,j} W_ij s_i s_j  -  sum_i h_i s_i          h = d + b
dE/dv_i = -rho'(v_i) r_i             so       dE/dt = -sum_i rho'(v_i) r_i^2  <=  0
```

(inputs from one-way projections belong to `h`). Equilibria with `rho' > 0` are exactly the stationary points of `E`.
This is the continuous Hopfield energy that equilibrium propagation builds on; symmetry is what makes the learning
rule of 1.4 a gradient.

**Wiring as blocks.** Neurons are numbered contiguously by region; every ordered pair of regions with synapses is one
dense block `[post, pre]`, and transport is the sum of block products. A reciprocal projection stores **one** weight
per pair and reads it in both directions (`s_pre W^T` into post, `s_post W` into pre), which is what keeps the weights
symmetric by construction. A block whose source region receives nothing (an input region) is evaluated once per settle
and folded into a constant. Large sparse circuits use CSR or a segmented sum instead of dense blocks.

### 1.3 Circuits: regions and projections

A circuit is a set of **regions** with roles - **input** (no inbound synapses; driven by data, a context trace or a
memory), **hidden**, and **readout** (nudged in learning, read as an action or a class; it may be split into groups,
each with its own softmax) - and **projections** between them: one-way or reciprocal, dense or masked by a
connectivity density, each with an optional frozen flag. A reciprocal projection from a region to itself is a symmetric
lateral block.

**Initialization.** Weights are drawn Glorot-uniform (Glorot & Bengio 2010) from a seeded generator and then
**certified**: all recurrent weights are multiplied by one factor so that the largest incoming mass equals the target
`kappa_0`. oann's default is `kappa_0 = 0.9`, which by the table above bounds a cold settle at about 70 sweeps (at the
default tolerances of 1.10). A Glorot draw is not certified by itself - its incoming mass grows like the square root of
the fan-in - so the factor shrinks wide layers. Training may optionally project back into the certified set after each
step (scaling a block by the factor its worst neuron needs, which keeps a reciprocal pair symmetric); it is off by
default for supervised training and on whenever a circuit is deployed on the board.

**Sizes** are not fixed by the model; 1.10 lists the classes oann plans for.

### 1.4 Supervised learning: free and nudged phases (Scellier & Bengio 2017; Laborieux et al. 2021)

Equilibrium propagation trains the symmetric energy model of 1.2 with two settlings of the same circuit. Add a cost `C`
on the readout to the energy, `E_beta = E + beta C`, so that a nudged phase has the extra force `-beta dC/ds` on the
readout:

1. **Free phase**: settle under the stimulus; the state is `s0`.
2. **Nudged phases**, both from `s0`: settle with `nudge = +beta f` to `s+`, and with `nudge = -beta f` to `s-` (the
   **centered** estimate), where `f = -dC/ds` evaluated at the current state.
3. **Update**, every synapse from its own two neurons and every neuron from itself, averaged over the batch rows:

```text
contrast[e] = mean_b ( s+[pre] s+[post] - s-[pre] s-[post] ) / (2 beta)
            = mean_b ( s+[pre] (s+[post] - s-[post]) + (s+[pre] - s-[pre]) s-[post] ) / (2 beta)    (computed form)
bias[i]     = mean_b ( s+[i] - s-[i] ) / (2 beta)
G           = -contrast                                       the gradient estimate handed to the optimizer
```

The computed form is a rank-2 update that never subtracts two large products, which matters in low precision. The
uncentered rule of Scellier & Bengio (2017) uses `s0` in place of `s-` and `beta` in place of `2 beta`. A reciprocal pair
stored once receives one contrast. A frozen synapse is skipped.

**The cost and its force.** For classification oann uses the softmax cross-entropy of the scaled readout,
`C = T CE(softmax(s_out / T), y)`, per group; its force is

```text
f = y - p,        p = softmax(s_out / T)
```

(derivation: `dC/ds_i = p_i - y_i`). `T` is a logit temperature: the readout activation lies in `(-1, 1)`, so a softmax
needs a gain to be able to approach certainty. Derived default: **`T = 0.25`**, for which saturated outputs
(`s = +-1`) give the right class a probability of at least 0.99 among up to about 30 classes
(`1 / (1 + 29 e^{-2/T}) = 0.990`). A squared-error cost `1/2 ||s_out - y||^2` with force `y - s_out` (the cost used by
Scellier & Bengio 2017) is available and needs no softmax.

**Why it is a gradient.** On a smooth stable branch with symmetric weights and converged phases, the contrast tends to
minus the gradient of the cost at the free fixed point as `beta -> 0` (Scellier & Bengio 2017, theorem 1; with
`dE/dW = -s_pre s_post` for a pair stored once and `dE/db = -s`). The centered estimate has bias `O(beta^2)` (Laborieux
et al. 2021); phases solved only to residual `tau` add an error of order `eps / beta`, where `eps <= tau / (1 - kappa)`
is the distance to the equilibrium. Setting the phase-error term at most equal to the bias term, `eps <= beta^3`, gives
the training tolerance rule used here:

```text
tau_lesson = max( beta^3 (1 - kappa) / 10 ,  100 u )          u = unit roundoff of T
```

(the factor 10 is oann's margin, the floor keeps the target above rounding noise).

**Defaults.** `beta = 0.5` for training (the equilibrium propagation papers use nudging strengths between roughly 0.1
and 1; a larger `beta` clears rounding noise and shortens the nudged phases, a smaller one lowers the `beta^2` bias),
and small `beta` (1e-3) only in gradient checks, where `tau` follows the rule above. The step is taken by any oann
optimizer on the gradient region; oann's default is Adam (Kingma & Ba 2015) at `lr = 1e-3`, `(0.9, 0.999)`, from
DESIGN.md section 5. A nudged phase starts from `s0` and its initial residual is at most `beta`, so it needs about
`ln(tau / beta) / ln q_beta` sweeps, at the nudged step of 1.2 (`q_beta = 0.933` at `beta = 0.5`, `T = 0.25`,
`kappa = 0.9`: 86 sweeps at the lesson tolerance).

### 1.5 Reward learning: three factors (Barto, Sutton & Anderson 1983; Williams 1992; Izhikevich 2007; Frémaux & Gerstner 2016)

Learning from an outcome that arrives after the action needs a third factor beside the two neurons' activities: a
broadcast scalar, here the TD error. oann composes the contrast of 1.4 with eligibility traces and a critic.

**Acting.** Settle freely; the policy is `pi = softmax(s_out / T)` per readout group; sample an action `a ~ pi`; then
run the two nudged phases with the target `y = onehot(a)` and `+-beta`. Because the nudge force is the cross-entropy
force, the contrast of the taken action is, by the same theorem,

```text
contrast[e]  ~=  T  d log pi(a) / d W[e]                 (oann's derivation: C = -T log pi(a))
```

the temperature-scaled policy score. Sampling and nudging use the same `T`, which keeps the estimate on-policy.

**Learning**, when the outcome arrives:

```text
e[w]    <- gamma lam e[w] + contrast[w]                  per synapse and bias, per stream (an eligibility trace)
V(x)     = w_c . phi(x) + b_c                            a linear critic on phi = the activations of the hidden regions
delta    = r + gamma V(x') - V(x)                        x' = the next free settle, warm, under the updated drive
delta    = clip(delta, -1, 1)
G        = -delta e                                      handed to the optimizer: the actor step
z       <- gamma lam z + [phi(x), 1]
w_c, b_c += alpha_c delta z / (1 + ||[phi(x), 1]||^2)    TD(lambda) for the critic, normalized (Widrow-Hoff)
```

This is a local actor-critic: the actor's update is the policy-gradient estimator `(r - baseline) grad log pi`
(Williams 1992) with a TD baseline and eligibility traces (Sutton 1988; Sutton & Barto 2018), written as the
three-factor rule "presynaptic x postsynaptic contrast, gated by a broadcast error" (Frémaux & Gerstner 2016). The
trace is a full per-synapse array per stream.

**Defaults.** `gamma = 0.99` (the common deep-RL value, Mnih et al. 2015), `lam = 0.9` (the order used in Sutton and
Barto's examples), rewards scaled to `[-1, 1]` and `delta` clipped there (reward clipping, Mnih et al. 2015),
`alpha_c = 0.1`. The trace decays by `gamma lam = 0.891` per moment, so a moment's contribution falls below 1e-3 after
`ln(1e-3) / ln(0.891) = 60` moments. The critic's step is far larger than the actor's Adam step, matching the
two-timescale recipe in which the critic tracks faster than the actor moves (Konda & Tsitsiklis 2000).

**Cautions** (failure modes to test for): *saturation* - a readout unit on its rail has `rho' ~ 0`, a nudge cannot
move it, the contrast vanishes and learning stalls (certification keeps units off the rails at initialization, and `T`
is chosen so certainty does not require saturation); *temperature against the activation range* (the derivation of `T`
above); *a critic that does not track* - `delta` then carries critic noise into every actor step.

### 1.6 Memories

- **Context trace** (Elman 1990): after each action, `tr <- (1 - a_tr) tr + a_tr s_hidden`, and the next moment's drive
  on the context input region is `kappa_tr tr`. Short-term memory as a stimulus, cleared by a reset. oann's defaults:
  `a_tr = 0.3` (a time constant of about three moments) and `kappa_tr = 1` (the same scale as other drives, so a
  unit-range state gives a unit-range drive). A strong trace can pin a policy; the end-to-end tests of 5.5 include a
  task that detects it.
- **Graph plasticity**: the weights and biases of 1.4 and 1.5.
- **Fast and persistent associative memory** (linear associator and delta rule: Kohonen 1972, Anderson 1972, Widrow &
  Hoff 1960; two stores of different speed: complementary learning systems, McClelland et al. 1995; episodic value
  memory: Blundell et al. 2016). The key is the observation `k` (a row vector, unit-normalized), the value is an
  action-value vector; a persistent matrix `C` is shared and a fast matrix `F` fades, both `keys x actions`:

```text
F    <- phi F                                              fast decay
C    <- C + min(1, c_0 (1 + |delta|)) k^T (v - k C)        persistent: slow delta rule, weighted by surprise
F    <- F + k^T (v - k (C + F))                            fast: one-shot, the read at k becomes v at once
read: drive[readout] += g k (C + F)                        a linear drive into the settle
```

  The value written is the reward at the chosen action's component, the error masked to that component. For a unit key
  and rate 1, the read after a write returns `v` exactly (`k k^T = 1`); orthogonal keys do not interfere and
  overlapping keys interfere by `dot(q, k)`; repeated writes move `C` toward `v` geometrically, as
  `(1 - c')^n`. Surprise-weighted consolidation follows the finding that arousal strengthens the consolidation of what
  caused it (McGaugh 2004). Defaults (oann's): `phi = 0.95` (a half-life of 13.5 moments), `c_0 = 0.02`, `g = 1` -
  *`c_0 = 0.2` as built*, tuned on the tasks of 5.5 (decision 21). This is the online consolidation; sleep (1.9) is the
  slower, offline one. *As built*, the fast decay is kept pending (one number, folded in at the next write), so a moment
  that writes nothing touches neither matrix (5.3.4).
- **Sparse-code store** (Marr 1969; Albus 1971; Kanerva 1988): a cerebellum-style associative memory used by the
  consolidator (1.9). A reading minus its running mean `x~` is projected by a fixed random matrix `R` onto `N` cells,
  the `k` strongest are kept, rectified and unit-normalized into a code; one table `T` per predicted field is read as
  `y = code T` and written by the delta rule at the active rows only:

```text
z     = R x~ ,   R entries N(0, 1/n)                       fixed, regenerable from a seed
code  = unit( relu( top_k(z) ) )
read:   y = code T
write:  T[idx] += rate code[idx]^T (target - y)              idx = the k active cells
```

  With a unit code and `rate = 1` a write stores its target exactly; for noisy targets a smaller rate averages over
  about `1 / rate` visits (the delta rule is stable for `0 < rate < 2`). Defaults (oann's): expansion `N = 20 n` and
  about 5% active cells, following the sparse expansion of the fly olfactory circuit (Dasgupta et al. 2017).

### 1.7 Arousal: the gate on learning and memory writes

Two broadcast scalars play the neuromodulator roles: the TD error `delta` of 1.5 (dopamine; Schultz et al. 1997) and an
**arousal level**, a leaky running measure of unexpectedness and disappointment (associability tracks the size of
recent prediction errors: Pearce & Hall 1980; norepinephrine as a signal of unexpected uncertainty and adaptive gain:
Yu & Dayan 2005, Aston-Jones & Cohen 2005; neuromodulators as controllers of learning parameters: Doya 2002). Computed
per outcome of the preceding action, with `delta` the TD error against the forecast made before acting:

```text
u_bar <- u_bar + alpha_s (|delta| - u_bar)                  running scale of the error (slow)
u      = |delta| / (u_bar + eps_u)                           relative surprise: about 1 on average, much more after a change
r_f   <- r_f + alpha_f (r - r_f) ,  r_s <- r_s + alpha_s (r - r_s)       recent and long-run reward
sig   <- sig + alpha_s (|r - r_s| - sig)                     reward spread
want   = clip( (r_s - r_f) / (sig + eps_u) , 0, 1 )          recent reward below its long-run level: disappointment
level <- level + alpha_f (u + want - level)
aroused = (t < t_0)  or  (level >= theta)
```

Defaults (oann's; none from the cited papers): `alpha_s = 0.01`, `alpha_f = 0.1` (windows of about 100 and 10 moments),
`eps_u = 1e-3` (rewards are scaled to `[-1, 1]`), `t_0 = 1 / alpha_s = 100` moments (a critical period, until `u_bar` is
established), and `theta = 2`: when nothing is unusual `E[u] ~ 1` by construction of `u_bar`, so `level` hovers near 1 and
`theta = 2` asks for errors twice the usual (or a full unit of disappointment). The `want` term is oann's own addition.
*As built, tuned on the tasks of 5.5 (decision 21)*: `alpha_s = 0.003` and `theta = 1.5` (`t_0` stays 100). With the
values above an agent in a noisy bandit turned calm for good at the end of its critical period, while its policy was
still poor, and after a reversal was aroused for about 14 moments before disappointment habituated - too few for its
learners to relearn. A slower long-run level keeps a lasting drop disappointing for longer.

A **calm** moment is one qualified settle and a greedy answer: no nudged phases, no learning, no memory write; the
eligibility trace only decays. An **aroused** moment samples from the policy, runs the nudged phases, keeps its
eligibility, learns from the outcome and writes memory (gating plasticity on neuromodulatory state: Bear & Singer
1986 is the classic case; Frémaux & Gerstner 2016 call it the third factor). Calm is cheap; aroused costs about three
settles and a learning pass (1.10).

### 1.8 One moment of an agent's life

```text
step(obs, reward):                                          one call per moment
  drive <- obs, context trace, memory read                 written in place
  settle free (warm) -> x', V(x')                          qualified at the life tolerance, or refuse
  delta = reward + gamma V(x') - V(x_prev)                 V(x') = 0 after a terminal outcome
  arousal.update(delta, reward)
  if the previous moment was aroused:                      it recorded its contrast
      memory.write(key_prev, action_prev, reward, |delta|)
      e <- gamma lam e + contrast_prev ;  actor step from delta e ;  critic step
      settle again, warm (the weights moved a little)
  else:
      e <- gamma lam e                                     lazily
  if aroused now:  a ~ softmax(s_out / T), nudged phases +beta and -beta from x' -> contrast ; keep s+, s-
  else:            a = argmax s_out
  context <- (1 - a_tr) context + a_tr s_hidden
  return a
```

There is no training/inference switch. `Imagine(obs)` settles a supplied observation with a private copy of the trace
and changes nothing. A refused settle is transactional: no action, no write. A checkpoint carries the parameters,
optimizer moments, eligibility, traces, memories, arousal statistics, generator state and the pending action.

### 1.9 Sleep: offline consolidation (McClelland et al. 1995; Robins 1995; Hinton & Plaut 1987)

Complementary learning systems hold that a fast, sparse, high-interference store acquires experience at once and a slow
distributed learner integrates it by interleaved replay offline. oann's consolidator pairs a slow sequence model with
the sparse-code store of 1.6. The slow part is a **gated linear recurrence** - the gates depend on the input only, so
the state is linear in its predecessor (the family of QRNN-style cells, Bradbury et al. 2017; gating as in Cho et al.
2014):

```text
l = sigmoid(g + G u) ;  z = tanh(B u + b) ;  h = l h_prev + (1 - l) z        slow weights g, G, B, b
m = read( code([alpha u, rho h]) )                                            the store's read; no gradient flows through it
y = C h + c + m                                                               readout; slow weights C, c
```

`alpha` and `rho` scale the input part and the state part of the code to contribute comparably.

- **By day**, each observed moment writes the residual `y_obs - (C h + c)` into the store (one exposure, the delta rule
  of 1.6), while the slow weights learn by backpropagation through time over a window (Werbos 1990), with a
  backtracking line search on the step.
- **By night**, `Sleep(cues)` dreams every cue once (the free path, store included), freezes the dreams, and teaches the
  slow weights the cue-to-dream pairs with store writes off, for a number of passes (replay: Wilson & McNaughton 1994;
  pseudorehearsal: Robins 1995).
- **At dawn** the store is rewritten with the residual between each dream and what the slow weights alone now produce,
  so it holds only what the weights did not absorb (fast weights to deblur old memories: Hinton & Plaut 1987).

Sleep consolidates what the system produces, mistakes included; it does not correct the store's errors. The
asymmetry is deliberate and stated: the day is local (delta rule), the night is backpropagation through time. It reuses
oann's graph machinery (2.7).

### 1.10 Sizes, budgets and cost

**Planning classes** (oann's; the model fixes none):

| class | example | regions | weights |
|---|---|---|---|
| small agent | 16 observations, 4 actions | 16 + 64 input, 64 hidden, 4 readout | about 5 k |
| image-scale supervised | 784-128-10 | 784 input, 128 hidden, 10 readout | about 100 k |
| two hidden regions | 784-512-256-10 | 784 input, 512, 256 hidden, 10 readout | about 535 k |
| large sparse | an imported wiring diagram | CSR blocks | 10^6 and up |

**Derived defaults** (oann's, except where a source is given):

| quantity | value | basis |
|---|---|---|
| activation, `dt` | `tanh`, `dt = 1` | Hopfield 1984; smallest `q` |
| certification target | `kappa_0 = 0.9` | cold settle at most about 70 sweeps |
| lesson tolerance | `max(beta^3 (1 - kappa) / 10, 100 u)`: 1.25e-3 at `beta = 0.5`, `kappa = 0.9` | 1.4 |
| life tolerance | `T (1 - kappa) / 40`: 6.25e-4 at `kappa = 0.9` | 1.5 and below |
| sweep budget | 128 | the cold need at `kappa = 0.9` is 71 sweeps; next power of two |
| imagination tolerance | the life tolerance divided by 100 | oann's choice |

The life tolerance is derived from what the settle is used for: every logit of the policy `softmax(s_out / T)` may be off
by at most `eps`, which changes a log-probability by at most `2 eps / T`; keeping that below 5% gives `eps = T / 40`, and
`tau = eps (1 - kappa)` by the distance bound of 1.2.

**Sweeps per moment** (planning; `q = 0.9`): a cold settle `ln(tau) / ln q` - 64 sweeps at the lesson tolerance, 71 at
the life tolerance. A warm calm moment starting from a residual of 0.3 takes about 59. An aroused moment adds two nudged
phases of `ln(tau / beta) / ln q` (64 at the life tolerance) and a warm re-settle: about three times a calm moment. A
supervised lesson is 64 + 2 x 57 = 178 sweeps at the lesson tolerance (or 121 uncentered).

**Cost against backpropagation.** A sweep costs one product per direction per recurrent block, so a lesson costs
about `2 x 178 = 356` products of each recurrent block, while one-way input blocks (evaluated once per settle) cost
one product and one outer product. Per sample, in multiply-adds, with `P_in` the one-way weights and `P_rec` the
reciprocal ones: settling about `2 P_in + 356 P_rec`; backpropagation `2 P_in + 3 P_rec` (the first layer needs only a
forward and a weight-gradient product, every other layer a forward, an input-gradient and a weight-gradient product).
Examples at `q = 0.9`: 784-256-10, `P_in = 200 704`, `P_rec = 2 560`: 1.31 M against 0.41 M
(3.2x); 784-512-256-10, `P_in = 401 408`, `P_rec = 133 632`: 48 M against 1.2 M (40x). Locality is paid for in
arithmetic when the recurrent part is large, which is why the board target (section 4) matters.

**What dominates**, per moment at batch 1:

1. **The settle**: per sweep, one GEMV per recurrent block in each direction (`W s` and `W^T s` on the same block) and
   an elementwise activation pass. Small, latency-bound matvec on blocks of 64x64 to 256x512 that fit L1/L2, repeated
   some tens of times per moment, so the loop must be compiled and allocation-free.
2. **Source projections** (input to first hidden) once per settle: one GEMV over the largest block.
3. **Aroused learning**: one pass over every plastic weight - eligibility decay plus a rank-2 outer product, the
   error-weighted step, the optimizer's moments. Memory-bound: four arrays of `E` elements (weight, eligibility, two
   Adam moments); at batch `B` the contrast is a TN GEMM of rank `B`.
4. **The sparse-code store** (if used): the `n x N` projection GEMV, a top-k, a `k`-row gather and scatter.

Nothing is sparse in the default circuit except the store's codes and large sparse circuits.

---

## 2. Supporting settling networks in oann

### 2.1 Shared and new

| | oann's backprop graph (DESIGN.md) | settling circuit |
|---|---|---|
| operand | `Matrix<T>`, rows = batch (streams), columns = features (neurons) | the same: a region is a column block of `[rows, n]` |
| weights | `[out, in]`, `Linear` forward `x W^T`, backward `dy W`, `dW = dy^T x` | a reciprocal projection is **one** `[post, pre]` block used as `s_pre W^T` and `s_post W`; its contrast is `dS_post^T S+_pre + S-_post^T dS_pre` - exactly `Linear`'s three GEMM forms |
| structure | recorded DAG, `Plan()`, one arena, zero allocation per step | recorded **cyclic** wiring, `Plan()`, one arena, zero allocation per sweep and per moment |
| execution | forward once, backward once | iterate to a fixed point (data-dependent count), several phases, state persists between calls |
| learning signal | VJPs, reverse order | contrast of phases (local); eligibility x TD error (three-factor) |
| update | optimizer over the flat gradient region | **the same**: the contrast is written as a gradient estimate into the gradient region |
| shared code | kernels, `Matrix`, planner, `Rand`, initializers, optimizers, checkpoints, `ParallelRows`, gradient checks | |
| new | the settle loop and residual, the certificate, nudges, phase buffers, contrast, eligibility, critic, memories, arousal, the moment loop, the store and sleep | |

### 2.2 Decision: a second recorded structure, `Circuit<T>`, beside `Graph<T>`

A settle is not a `Graph` op: the graph's contract is one forward and one reverse pass of VJPs, while a settle iterates
an unknown number of times, keeps its state between calls (warm starts, a life), is cyclic, and learns from a contrast
with no VJP to write (backpropagating through its sweeps would be BPTT, the thing equilibrium propagation avoids).
So `Circuit<T>` is a sibling: **recorded once, planned once, replayed with zero allocation**, on the same `Matrix`,
kernels, arena discipline, initializers and optimizers. Handles are numbers, as `Var` is: `type Region I64`,
`type Projection I64`; the circuit holds a `List` of plain records and one `Array<T>` arena; the neuron model and the
cost kinds are closed enums dispatched by `match`.

The two compose in one direction cleanly: a `Graph` encoder can feed a circuit's drive, and the equilibrium-propagation
estimate supplies the gradient that encoder needs (for a free neuron, `dC/dd = dC/db = -(s+ - s-) / (2 beta)`; through
a source projection, `-(dS_post W) / (2 beta)`), so "settling inside, backprop outside" trains end to end. The
consolidator (1.9) maps onto the existing `Graph` instead: its slow weights are an unrolled recurrent cell trained by
backprop, its store a custom op whose read carries no gradient, and `Sleep` is a training run on dreams (2.7).

### 2.3 The plan and the arena

`Plan()`:

- orders regions so each is a contiguous column range of `[rows, n]`;
- lays out **parameters** (one `[post, pre]` block per projection, biases, the critic) contiguously, with **gradients**
  (the contrast), **optimizer moments** and **eligibility** (per stream row) in the same layout - one flat view each,
  as for the graph;
- lays out **state** per phase - `Free`, `Plus`, `Minus`, `Scratch` (imagination, refused attempts) - as `v` and `s`
  `[rows, n]`, **double-buffered** so a sweep reads one copy and writes the other;
- lays out **workspace**: the drive, the per-settle constant `c`, `total`, `dS = S+ - S-`, nudge targets and softmax
  probabilities, a `U8` connectivity mask for each projection with `density < 1`;
- classifies blocks: a block whose pre region is an `Input` is **folded** into the constant; the rest are recurrent;
- computes the certificate's incoming masses (absolute row and column sums of the blocks, combined per neuron) and
  derives `kappa`, `q`, the tolerance and the budget from them.

Input regions take their fixed point in closed form, `s = rho(d + b)`, once per settle - the same equilibrium as
relaxing them, in fewer sweeps (the trajectory differs, the equilibrium does not).

### 2.4 The settle loop: no allocation, one transport per sweep

```text
once per settle:  c = d + b + sum_{folded blocks} s_in W^T             (GEMV over the source projections)
per sweep k:      total = c + sum_{recurrent blocks} [s_pre W^T into post ; s_post W into pre if reciprocal]
                  total += nudge(s_k) on readout groups
                  r = total - v_k,   res[row] = max |r|                   (the residual of state k, for free)
                  if every res <= tol: state k qualifies; stop
                  v_{k+1} = v_k + dt r,  s_{k+1} = rho(v_{k+1})          (into the other buffer)
```

Because the residual of state `k` is a by-product of computing step `k+1`, every sweep is checked at no extra
transport, and the qualified state is the one returned. Where a reduction per sweep is costly on a host, checking
every `chunk` sweeps (one extra transport) is a parameter. Damping halves `dt` over portions of the same budget.
Finite phases (`Run(phase, sweeps)`) skip the check and report the final residual. All rows advance together;
per-row residuals are reported. A refused settle leaves the committed phases untouched because it ran in the double
buffer. At batch 1, the recurrent GEMVs read contiguous rows of the `[post, pre]` block in both directions (row dots
for `W^T`, row axpys for `W`), so both vectorize.

### 2.5 Learning as planned passes

- **Contrast** (batch `B`): `dS = S+ - S-` once; per plastic block two TN GEMMs into the gradient region,
  `G = -(dS_post^T S+_pre + S-_post^T dS_pre) / (2 beta B)`, masked by connectivity; biases `-ColumnSums(dS) / (2 beta B)`.
  Reciprocal blocks need nothing more. Then any oann optimizer steps the parameter region, followed by the optional
  projection into the certified set (1.3).
- **Eligibility and the TD error** (batch 1): at act time keep `s+` and `s-` (vectors); at learn time, **one fused pass
  per plastic block**: `E = gl E + (d_post s+_pre^T + s-_post d_pre^T) / (2 beta)` then `G = -delta E` - a decayed
  rank-2 update and a scale, reading and writing `E` once. Calm moments only decay `E`; this is kept lazy - a scalar
  multiplier on `E`, folded in at the next aroused moment - so **a calm moment touches no synapse**. A truncated
  trace (the last 60 moments, each a rank-2 term, so rank 120) replaces the `E`-sized array by `120 (m + n)` numbers per
  block; it saves memory only when `120 (m + n) < m n`.
- **Critic**: a dot for `V`, an axpy for its update.
- **Gradient checks** (F64): section 5.2.

### 2.6 The agent: memories, arousal, and the moment loop

`Agent<T>` composes a circuit with a critic, a `Trace`, a `Memory` (the fast and persistent matrices) and an `Arousal`,
each a struct allocated once (the memory's `C` and `F` are small `Matrix` values). The moment loop is 1.8, as
straight-line olang over planned buffers: the drive is written in place (observation, the trace, the memory read
`k (C + F)`), and every step after is a kernel call on views. Nothing allocates per moment; independent lives run as
`spawn`ed tasks, each with its own arena (at these sizes threads inside one settle cost more than they save: a cached
spawn is about 10 us, a 64x64 GEMV about 0.1 us). Refusal is an error (`NotSettled`), and transactional: candidate
states live in `Scratch` or the double buffer, and candidate parameter steps are applied only after every phase
succeeded. Checkpoints write the parameter region, moments, eligibility, traces, memories, arousal statistics,
generator state and the pending action (safetensors plus a metadata block, DESIGN section 10).

### 2.7 The store and sleep (later phases)

The sparse-code store needs a top-k, a gather-weighted read and a `k`-row write (section 3); its projection is stored
on a CPU (on an FPGA it can be regenerated from the seed instead, section 4). The consolidator is a `Graph` model (the
recurrent cell unrolled over a window, BPTT) with the store as a `Custom` op; `Sleep` is: dream the cues (forward with
the store), freeze the dreams, train the slow weights on them with the store excluded, rewrite the store.

### 2.8 API sketch (olang)

An agent living one stream - as built (`agent.olang`, `examples/bandit_settle.olang`):

```olang
import "agent"          # oann's modules, by path relative to the importing file (DESIGN section 12)
import "circuit"
import "std/io"

fn main() ? circuit.NotSettled + io.IoError {
    s := agent.Settings()                          # defaults; set fields to change them
    a := agent.Agent<F32>(16, 4, I64[64], 7, s)    # 16 observation inputs, 4 actions, one hidden region of 64, seed 7
    world := ContextBandit(1)
    obs := Array<F32>(16)
    world.Observe(obs)
    action := try a.Begin(obs)                     # the first moment has no reward to report
    for moment in range 600 {
        reward := world.Act(action)                # execute, measure
        world.Observe(obs)
        action = try a.Step(obs, reward, true)     # learn from it if the last moment was aroused, then choose again
    }
    try io.Print("calm " $(a.Calm) " aroused " $(a.Aroused) " sweeps " $(a.Sweeps) "\n")
}
```

A custom circuit taught by contrast, batch 32 - as built (`circuit.olang`, `examples/mnist_settle.olang`):

```olang
c := circuit.Circuit<F32>(32)                                # rows (the batch)
pixels := c.Input(784)                                       # linear by default: the drive is the activity
hidden := c.Hidden(128)                                      # tanh by default
digits := c.Readout(10)                                      # one softmax group, CrossEntropy(T = 0.25)
c.Project(pixels, hidden, circuit.Wiring.OneWay)
c.Project(hidden, digits)                                    # Reciprocal by default
c.Plan(r)                                                    # lays out the arena, draws Glorot-uniform weights
c.Certify(0.9)                                               # scale the recurrent blocks to a gain of 0.9
opt := circuit.Adam<F32>(c, 1e-3)
lesson := circuit.Lesson(0.5)                                # beta; tolerance and budget come from the certificate
for b in range train.Batches() {
    n := train.Fill(b, c.DriveData(pixels), c.LabelData(digits))
    try c.Teach(n, lesson)                                   # free, +beta, -beta, contrast into the gradient region
    opt.Step(c)
}
```

The core surface, as built:

```olang
type Phase enum { Free  Plus  Minus  Scratch }               # one per line in the source
type Rho enum { Tanh  HardSigmoid  Linear  LifRate(leak F64, threshold F64) }
type Neuron enum { Rate(rho Rho)  Lif(leak F64, threshold F64, synapse F64, window I64) }
type Cost enum { CrossEntropy(temperature F64)  Squared }
type Wiring enum { OneWay  Reciprocal }
error NotSettled { BUDGET  NONFINITE }

fn (c mut Circuit<<T>>&) Input(size I64, model Neuron = Neuron.Rate(Rho.Linear)) Region
fn (c mut Circuit<<T>>&) Hidden(size I64, model Neuron = Neuron.Rate(Rho.Tanh)) Region
fn (c mut Circuit<<T>>&) Readout(size I64, fit Cost = Cost.CrossEntropy(0.25), model Neuron = ...) Region
fn (c mut Circuit<<T>>&) Project(pre Region, post Region, kind Wiring = Reciprocal, density F64 = 1, frozen Bool = false) Projection
fn (c mut Circuit<<T>>&) Plan(r mut rand.Rand&)
fn (c mut Circuit<<T>>&) Settle(p Phase, budget I64, tolerance F64) Settlement ? NotSettled  # qualified, warm
fn (c mut Circuit<<T>>&) Run(p Phase, sweeps I64) Settlement               # finite phase, residual reported
fn (c mut Circuit<<T>>&) Warm(p Phase, from Phase)                         # a nudged phase starts from the free state
fn (c mut Circuit<<T>>&) Nudge(p Phase, beta F64)                          # every readout, by its Cost
fn (c mut Circuit<<T>>&) Residual(p Phase, out mut Array<<T>>&) F64        # per row, no settling
fn (c mut Circuit<<T>>&) Totals(p Phase) linalg.Matrix<<T>>                # every neuron's total input
fn (c mut Circuit<<T>>&) Contrast(beta F64, centered Bool = true)          # Plus/Minus -> gradient region
fn (c mut Circuit<<T>>&) Teach(n I64, l Lesson) I64 ? NotSettled           # free, +beta, -beta, contrast
fn (c mut Circuit<<T>>&) DriveGrad(reg Region, out mut Array<<T>>&, beta F64, centered Bool = true)
fn (c mut Circuit<<T>>&) Gain() F64                                        # kappa
fn (c mut Circuit<<T>>&) Rate(beta F64 = 0) F64                            # q (of a phase nudged with beta), or 1
fn (c mut Circuit<<T>>&) Certify(target F64 = 0.9)                         # one factor on every recurrent block
fn (c mut Circuit<<T>>&) Restrain(target F64 = 0.9)                        # project back into the certified set
fn (c mut Circuit<<T>>&) Energy(p Phase, row I64) F64
fn (c Circuit<<T>>&) Value(p Phase) linalg.Matrix<<T>>                     # views of state, as g.Value(v)
fn (c Circuit<<T>>&) Loss(p Phase) F64
fn (t mut Eligibility<<T>>&) Add(c mut Circuit<<T>>&, decay F64)          # also Decay(decay), Credit(c, delta)
fn (a mut Agent<<T>>&) Step(obs Array<<T>>&, reward <T>) I64 ? NotSettled   # phase 2
fn (a mut Agent<<T>>&) Imagine(obs Array<<T>>&) Settlement ? NotSettled    # phase 2: private; changes nothing
```

### 2.9 Phases of work

Settling networks follow the transformer work (section 6).

1. **Done (2026-10-09, `circuit.olang`):** `Circuit` with `tanh` neurons, `Settle`/`Run`/`Residual`, the certificate,
   supervised `Teach`, and the checks of 5.1-5.2 in F64; the neuron enum with a working `Lif` variant (4.6); and, of
   phase 2, the pieces phase 1's structures made cheap: the eligibility trace (`Eligibility`, lazy and fused) and the
   checks of 5.2.4-5.2.6. Results in 2.10 and 5.7.
2. **Done (2026-10-09, `agent.olang`):** the `Agent` - critic, `Trace`, `Memory`, `Arousal`, the moment loop,
   checkpoints - with the checks of 5.3 (all but the sparse-code store, which belongs to phase 3) and the end-to-end
   tasks of 5.5. Results in 2.11 and 5.8.
3. The fixed-point simulation of the board engine (4.2, 5.6); the sparse-code store, extra readout groups, CSR
   projections for large circuits.
4. The consolidator and `Sleep` on the `Graph`; the PYNQ-Z2 overlay; further neuron models (spiking, section 4.6).

### 2.10 As built: phase 1

One module, `circuit.olang`: the circuit, its neurons, learning, an Adam over its parameters, and its tests (`make
test`); `examples/xor_settle.olang`, `examples/xor_spiking.olang` and `examples/mnist_settle.olang` run it end to end.

- **Records and one arena.** Regions and projections are plain records in `List`s, their handles numbers.
  `Plan(r)` lays out one `Array<T>`: the parameters (each projection's `[post, pre]` block in recording order, then a
  bias per neuron - an input's is unused), their gradients in the same layout, each region's drive (`Rows x size`,
  contiguous, so a loader fills it as it fills a graph's input), the squared-error readouts' targets, four phases of
  two slots of state, and the workspace (the constant, the totals, spike counts, `dS`, the per-row residuals, the
  masses). Classes are an `Array<I32>`, connectivity masks an `Array<U8>`. A settle, a lesson, a contrast and an Adam
  step allocate nothing.
- **The double buffer is a transaction.** A phase has a committed slot and a candidate slot. A settle copies the
  committed state into the candidate, sweeps there in place (the totals are their own workspace, so a sweep needs no
  second copy of the state) and commits by flipping the phase's slot; a refusal leaves the committed state as it was,
  bit for bit (checked).
- **A sweep**: the totals - the constant plus each recurrent block's transport (`linalg.Gemv` per batch row; a
  reciprocal block's two directions read the one `[post, pre]` block, as rows dotted and as rows added) plus the
  nudge; a pass for the residual of every row; on qualifying, the commit; otherwise a pass stepping `v += dt r`,
  `s = rho(v)` (`linalg.FastTanh`). The residual has its own pass so that the committed state is the one whose
  residual was measured.
- **The constant** (drives, biases, folded blocks) is made once per settle, and once per lesson: the nudged phases
  reuse the free phase's. Input regions take `v = d`, `s = rho(d)`; they have no bias, since their state does not
  differ between phases and so no contrast could teach one.
- **Damping** (`Attempts`) splits the budget into parts with `dt` halved in each, continuing from where the last part
  ended. **Nudged phases** step at `dt / (1 + beta lambda)` (1.2).
- **Budgets and tolerances**: `LessonTolerance(beta)` is 1.4's rule with `kappa` held at most 0.9, so that a circuit
  trained out of the certified region still has one; `Budget(tolerance, beta)` is twice the certified cold need
  (`q` held at most 0.95) and at least 128. Tolerances and nudges are `F64`, as std/linalg's `alpha` and `beta` are.
- **The contrast** is the rank-2 form, computed a block row at a time with four batch rows per pass over the row, so
  the row stays in the first-level cache; masked synapses and frozen blocks get 0. Checked against its definition.
- **The optimizer** is `circuit.Adam<T>` over the parameter region, Kingma and Ba's update in one pass. oann's
  `optim` steps an `nn.Graph`; a flat-region entry point there would let every oann optimizer serve both (follow-up).
- **The encoder's gradient** is `DriveGrad(region, out, beta)`, of the mean cost over the rows, so it adds to a graph's
  loss gradient directly.
- **The eligibility trace** (phase 2's first piece): `Eligibility<T>(c)` - `Decay(g)` at a calm moment multiplies one
  pending number, `Add(c, g)` folds it in while adding the contrast `Teach` left (one pass), `Credit(c, delta)` writes
  the actor's step `G = -delta E`. One stream (decision 19).
- **Spiking**: 4.6.
- **Cost, measured** (784-128-10, batch 32, F32, on a shared four-core machine): a lesson about 4.5 ms - the folded
  constant 0.9 ms, about 40 sweeps over the three settles at about 70 us each, the contrast 0.7 ms, Adam 0.1 ms. The
  folded 784-wide block is a matrix-vector product per batch row, read 32 times; std/linalg's blocked `Gemm` would do
  it at several times the speed but allocates its packing panels per call in this compiler, and the runtime's chunk
  pool then leaks them (DESIGN.md section 8) - the workspace `Gemm` now on olang master is the fix to take up.

### 2.11 As built: phase 2 - the agent

`agent.olang`: `Agent<T>` and its parts, each a struct of its own and checked on its own (5.3); the end-to-end tasks of
5.5 are `examples/bandit_settle.olang`.

- **Parts.** `Arousal` (1.7, plain data), `Memory<T>` (1.6: the persistent and fast `keys x actions` matrices, keys the
  unit-normalized observation, the fast decay pending), `Trace<T>` (the context trace), `Critic<T>` (1.5: linear on the
  hidden activities, its weights and trace in F64, the step normalized by `1 + |[phi, 1]|^2`), `circuit.Eligibility<T>`
  (2.5) and `optim.FlatAdamW<T>` for the actor.
- **The circuit.** The observation and the context trace are input regions, one-way into the first hidden region; the
  hidden regions are reciprocal in a chain, the last reciprocal with the readout (one neuron per action, the policy
  `softmax(s / T)`); the memories' read is the readout's drive. Certified at `kappa_0` when made.
- **The moment** (1.8) is `Step(obs, reward, terminal)`. It settles free (warm, at the life tolerance) and measures the
  TD error, clipped (`V(x_prev)` with the critic as it is now). It updates arousal. If the last moment was aroused, it
  learns: the memory write; the contrast of the nudged phases kept from then; the eligibility (`E <- gamma lambda E +
  contrast`); the actor's step on `-delta E`; the critic's step. Then it settles again on the moved weights. Otherwise
  the traces only decay (the eligibility lazily). Finally it acts: sampled, with the nudged phases at `+-beta`, when
  aroused; greedy when calm. `Begin(obs)` is the first moment. `Imagine(obs)` settles in the scratch phase and changes
  nothing.
- **Episodes.** A terminal outcome clears the eligibility and the critic's trace after learning from it. The context
  trace persists (only `Ctx.Clear()` resets it).
- **Refusals.** A refused free settle is a refused moment: no action, nothing written (`NotSettled`). A refused
  nudged phase leaves the moment acted on but with no contrast to learn from. A refused settle after learning keeps the
  state settled before it. All are counted in `Refused`.
- **One stream** (open question 5): an agent lives one stream (`B = 1`); many lives are many agents.
- **Checkpoints.** `Save(path)` / `Restore(path)` in the safetensors layout: an 8-byte header length, a JSON header,
  and every tensor as F64 (so an F32 agent's values round-trip exactly).
  - Saved: the parameters; the free state and the nudged states the next contrast reads; Adam's moments; the
    eligibility; the critic and its trace; the context trace; both memories; the last moment's observation and
    features; and the scalars (arousal's statistics, pending decays, counters, the pending action and the generator).
  - std/rand gives no access to a generator's state, so the agent owns its generator (made from a seed) and the
    checkpoint keeps the seed and the count of numbers drawn. A restore replays them, so it is into a new agent of the
    same shape, seed and settings, and the restored agent then lives exactly the saved one's moments (checked).
- **The circuit is made by a function** (`circuitFor`), not recorded in the agent's constructor. A constructor
  that pushes onto a `List` of one of its own reference fields puts the list's storage in its own scope, which closes
  when it returns: a use-after-free in olang ef939ae (`repro/ctorpush.olang`). Recording the circuit in the constructor
  crashed an agent several moments later, once a loop body allocated text over the freed chunk.
- **Cost, measured.** The small agent of 1.10 (16 + 64 inputs, 64 hidden, 4 actions; 5.6 k parameters) takes about
  15 us a moment and 4-5 sweeps a moment in the bandit (warm starts, drives far from the rails): 160 000 moments in
  2.4 s.

---

## 3. What the matrix library and olang must provide

DESIGN section 13 already has `Gemm` with transposes, `alpha` and `beta`, views, `AddRow`, `ColumnSums`, `AddScaled`, the
`Map` family, row reductions, `Dot`/`Norm`, random fills and `ParallelRows`. Settling circuits add the following.
Items 1, 4 (softmax) and 6 are shared with the transformer work, which comes first: a transformer's single-token step
is a batch-1 GEMV, and its softmax and activations need the same vectorized row reductions and `exp`/`tanh`.

1. **A batch-1 / small-row path inside `Gemm`** (rows <= 4): no packing, no threads, GEMV kernels for `NT` (row dots)
   and `NN` (row axpys) at L1 speed on 64x64 to 256x512 blocks, zero allocation. This is the hot loop: some tens of
   times per moment, a few blocks each. Or explicitly: `matrix.Gemv(y mut Array<<T>>&, a Matrix<<T>>&, trans Bool, x Array<<T>>&, alpha <T>, beta <T>)`.
2. **Rank-1 and decayed rank-2 updates**: `matrix.Ger(a mut Matrix<<T>>&, x Array<<T>>&, y Array<<T>>&, alpha <T>, beta <T>)`
   (`A = beta A + alpha x y^T`) and the fused eligibility form `E = g E + alpha (u1 v1^T + u2 v2^T)`; at batch `B` the
   TN `Gemm` with `beta` covers it. The general statement: `W += eta (a b^T - c d^T)` computed as
   `a (b - d)^T + (a - c) d^T`, never as a difference of two large products (cancellation).
3. **Masked updates**: `w.AddScaledMasked(g, alpha, mask Array<U8>&)` for sparse connectivity and frozen synapses, and
   a scalar rescale of a block for the certified projection.
4. **Row reductions over column blocks**: `RowMaxAbs` (the residual's sup norm), `RowNorms` and `NormalizeRows` (memory
   keys), and a fused `RowSoftmax(dst, src, temperature)` over a `Cols` view (the cross-entropy nudge and the policy, one
   softmax per readout group).
5. **Absolute sums**: `RowAbsSums`, `ColumnAbsSums` (the incoming masses of the certificate).
6. **Vectorizable activations with derivatives**: `tanh`, `sigmoid`, `exp` as polynomial or rational approximations
   written in olang, F32 and F64, with a stated maximum error (a settle needs about 1e-6 against a tolerance of 1e-3).
   Under olang's X8, `std/math`'s `Tanh`/`Exp` are always calls of the C library and never replaced, so a loop calling
   them per neuron per sweep does not vectorize; at n = 64 that costs as much as the GEMV. The same kernels serve
   oann's backprop activations.
7. **Fused relax pass** (oann may write it over row slices): from `total`, `v`, `s`, `dt` and the neuron model, produce
   the per-row residual and the next `v`, `s` into the other buffer, in one pass. The library's part is that row-slice
   loops vectorize and views cost nothing.
8. **Top-k per row** (k-winners for the store): indices and values, linear-time selection, a caller-supplied workspace.
9. **Sparse-code products**: `GatherRowsWeighted(out, table, idx, w)` (`sum_j w_j table[idx_j]`) and
   `ScatterRank1(table, idx, w, err, rate)` (`table[idx_j] += rate w_j err`).
10. **CSR sparse matrices** (phase 3): SpMV/SpMM in both transposes, and a sampled dense-dense product (the contrast at
    existing synapses only) for large sparse circuits.
11. **Bernoulli fills**: `FillMask(r, density)`, seeded from the shared `Rand`.
12. **Precision**: F64 for every kernel (gradient checks, references); F32 for lives and learning. F16/BF16 cannot
    resolve the settle tolerance (their epsilons, 9.8e-4 and 7.8e-3, are at or above it) and contrasts cancel, so only
    stored inference weights may be low precision, read with F32 accumulation.

**From olang**, none of it blocks phase 1:

- **Vector math** (item 6) is library code under the current rules; a vectorized `exp`/`tanh` in std (or a stated
  exception to X8 for a vector library) is an olang direction question, not an oann one.
- **Const generics** (planned): region and block sizes as compile-time constants give fully unrolled small GEMVs, a
  compile-time wiring check, and the fixed shapes an FPGA build needs.
- **Two-operand `At`/`SetAt`** (already noted in DESIGN section 13).
- **Fixed-point types** for the board simulation are expressible today (`type Q1_15 extends I16` with operator
  methods); saturating arithmetic would be library code.
- **Cross-compilation to the board's ARM cores** (olang's `TargetArch` is the host's today) for the processing-system
  side of section 4.
- **Serialization by reflection** (deferred in olang) would shorten checkpoints of a whole life; hand-written writers
  suffice meanwhile.

---

## 4. The FPGA target: PYNQ-Z2

**The board.** A Zynq-7020 system on chip: a dual-core Cortex-A9 *processing system* (PS) with DDR memory, and a
*programmable logic* (PL) fabric of about 53 k LUTs, 220 DSP48E1 slices (a 25x18 signed multiplier with a 48-bit
accumulator) and 140 BRAM36 blocks (36 Kbit each, about 4.9 Mbit or 630 KB in all; each splits into two 18 Kbit halves
and is true dual-port). There is no UltraRAM. Tools are Vivado and Vitis HLS; a PYNQ overlay (a bitstream with its
hardware description and a driver) is loaded and driven from the ARM cores. Data moves between DDR and the PL by AXI
DMA (the AXI high-performance ports are 64 bits wide: 800 MB/s peak at a 100 MHz fabric clock), control through AXI-Lite
registers. Figures about the part follow the Zynq-7000 and 7-series documents listed under References; the rest are
*planning*.

### 4.1 What maps well

- **Local learning, no backward pass.** Every update reads the synapse's own pre and post activity in two phases and
  one broadcast scalar (`delta`); nothing is stored across a backward pass. Weights stay where they are used
  (weight-stationary).
- **The settle is a fixed dataflow.** A sweep is block GEMVs, an activation, a max-reduction; a moment is some tens of
  sweeps with a data-dependent stop.
- **The control is scalar.** Arousal, the critic, budgets, the residual stop, sampling: a small FSM, or the ARM cores.
- **The sparse-code store is hardware-shaped**: the projection is regenerable from a seed by a PRNG (no storage), top-k
  is a streaming selection, a read or write touches `k` rows of thousands - only the tables need external memory.
- **The plan is the hardware description.** A planned `Circuit` (regions, blocks, phases, passes) is data. It can be
  emitted as HLS C++ (as hls4ml does for other models) or, better here, as a table of block descriptors for one fixed
  engine (4.2), with const generics making shapes constants.

### 4.2 The engine and its formats

One engine in the PL runs the sweeps for any circuit within its budget; descriptors (block shape, address, direction,
which neuron model) are loaded at run time, so a new circuit needs no re-synthesis.

**Lanes.** 64 multiply-accumulate lanes, each one DSP48E1 (a 16x16 product, 48-bit accumulate, one per clock, fully
pipelined) with its own weight memory of at least one BRAM18 (1 K x 18 bits). Sixty-four independent memories are the
minimum for 64 weights per clock: 64 BRAM18 = 32 BRAM36, holding up to 65 536 weights; each further layer of 64 BRAM18
adds 65 536. A block's longer dimension is spread over the lanes.

**One stored copy serves both directions.** For a block `[m, n]`, lane `i` holds a row (or column) of the longer
dimension. `W s` accumulates inside each lane; `W^T s` multiplies the same words by the broadcast other operand and
reduces across the lanes through a pipelined adder tree (63 adders, about 2.5 k LUTs). So a reciprocal weight is stored
once and the two products differ only in where accumulation happens.

**Activation and residual.** `tanh` by a 1 024-entry table over `[-8, 8)` with linear interpolation: the interpolation
error is at most `h^2/8 max|tanh''|` with `h = 16/1024` and `max|tanh''| = 0.77`, i.e. 2.4e-5, about one unit in the
last place of the activation format. Eight such units (one DSP and a share of a BRAM18 each) keep the activation pass
from dominating small circuits. A comparator tree gives the row maximum of `|r|` for the stop rule. The nudge needs a
softmax over a readout group every sweep (the force depends on the current state): a small `exp` table and one divider,
within the per-sweep overhead; the squared-error nudge avoids it.

**Formats** (fixed point; derived from the certificate). A certified circuit has incoming mass below 1 and activations
below 1 in magnitude, so:

| quantity | format | range, step | why |
|---|---|---|---|
| weight | Q1.15, 16 bit | [-1, 1), 3.1e-5 | `|W| < 1` for a certified circuit |
| activation `s` | Q1.15, 16 bit | [-1, 1), 3.1e-5 | `|tanh| < 1` |
| potential `v`, total | Q4.14, 18 bit (a BRAM word) | [-8, 8), 6.1e-5 | headroom for biases; `tanh(8) = 1 - 2.3e-7` |
| product | Q2.30, 32 bit, accumulated in 48 | | no overflow for any block of fewer than 2^16 terms |

One rounding per neuron per sweep (to Q4.14) costs at most 3.1e-5. The residual floor, a difference of two quantized
values plus the table error, is about 1e-4, so the board's tolerance is `max(tau, 3e-4)`. The distance to the true
equilibrium is at most the floor over `1 - kappa`, about 1e-3 at `kappa = 0.9`, comfortably below the life tolerance's
`eps = T/40 = 6.25e-3`. The board therefore runs only *certified* circuits (the projection of 1.3 on). The arithmetic is
integer, so it is associative and the simulation of 5.6 can match the engine bit for bit.

**Why not floating point.** The DSP48E1 has no floating-point unit: an F32 multiply-add built from the vendor cores
takes about five DSP slices, which would leave roughly 40 lanes of 220 and double the weight memory. F16/BF16 are not
native either. Sixteen-bit fixed point is both the cheapest and, by the derivation above, sufficient.

**Resource plan** (*planning*; confirm by synthesis):

| resource | available | plan |
|---|---|---|
| DSP48E1 | 220 | 64 lanes + 8 activation units + a few for the divider: about 75 (34%). 128 lanes (58%) is the realistic ceiling |
| BRAM36 | 140 | weights up to 96 (3 layers of 32, 196 608 weights, 393 KB); state, drive, constants, tables and FIFOs about 30; 14 spare |
| LUT | about 53 k | engine and adder tree 10-12 k, activation and residual 2 k, control 2 k, AXI/DMA/interconnect 6-8 k: about 25 k (45%) |
| clock | | 100 MHz planning (150 MHz to try once the adder tree is pipelined) |

**Cycles.** A sweep costs `sum over blocks of 2 x short x ceil(long / 64)` MAC cycles plus about 40 cycles of pipeline
latency (the activation and residual passes overlap the MACs). For a balanced block this is `2P/64 + 40` for `P` weights.

### 4.3 What fits

All at 64 lanes and 100 MHz, one stream, 64 sweeps (a cold settle at the default tolerance). "Weights on chip" counts
the PL-resident weights (the recurrent core, plus the folded input blocks where they fit).

| circuit | folded (one-way) weights | recurrent core | on chip | sweep (cycles) | 64 sweeps |
|---|---|---|---|---|---|
| small agent (16 + 64 input, 64 hidden, 4 readout) | 5 120 | 256 | 5 376 weights, 32 BRAM36 (under-filled) | 49 | 31 us, plus 1 us for the folded blocks |
| 784-128-10 | 100 352 | 1 280 | 101 632 weights, 64 BRAM36 | 80 | 51 us, plus 17 us for the folded block |
| 784-512-256-10 | 401 408 | 133 632 | core only: 96 BRAM36 | 4 216 | 2.7 ms; the folded block (803 KB) streams from DDR |
| the weight-budget limit | | 196 608 | 96 BRAM36 | 6 184 | 4.0 ms |

Reading the table:

- Circuits up to the 784-128-10 class fit entirely, folded input blocks included. Cycles are latency-dominated for the
  small ones (of the agent's 49 cycles per sweep, 40 are fixed pipeline latency) and compute-dominated for the large ones.
- The 784-512-256-10 class fits as a *core*: the recurrent weights (68% of the weight budget) are resident, the 0.8 MB
  folded block is applied once per settle from DDR - 803 KB at 800 MB/s peak is at least 1 ms, comparable to the sweeps -
  or by the ARM cores. Folded blocks are one-way and used once per settle, so streaming them costs time but no
  on-chip memory.
- 128 lanes halve the compute-dominated rows at the price of 128 DSPs and 64 BRAM36 of lane memory.
- Anything beyond about 197 k resident weights is out of scope for this board.

An aroused moment adds about two calm moments' worth of sweeps (1.10); the PL times per moment scale accordingly.

### 4.4 Division of work between PL and PS

- **PL**: settle, nudged phases, residual and stop, refusal flag; resident weights; the memory read as part of the drive.
- **PS (ARM cores)**: the moment loop, arousal, critic, sampling, and **learning**. Master weights, eligibility and
  optimizer moments live in DDR in F32; after an aroused moment the PS applies the update and pushes quantized Q1.15
  copies of the changed blocks by DMA. This also answers the precision problem of 4.5: increments are accumulated in F32,
  the quantized copy only needs to be the best Q1.15 rounding of it. Because learning happens only in aroused moments,
  the PS cost is rare. *Planning*, at about 1 GB/s sustained memory bandwidth on the A9 cores and four F32 arrays
  (weight, eligibility, two moments) read and written: 0.2 ms for the small agent, 3 ms for 784-128-10, 17 ms for
  784-512-256-10; the DMA push of the resident weights is at most 0.7 ms.
- **Stage 2 (later)**: a fused update pass in the PL for small cores. It needs per weight 16 bits (weight) + 32 + 32
  (eligibility, moment) = 80 bits, so about 39 k weights in 96 BRAM36 (3.1 Mbit); 16-bit learning state with stochastic
  rounding needs 48 bits and allows about 65 k.
- **Host work off the board**: sleep (BPTT with a line search) is sequential and float-hungry; it runs on the PS or a host,
  and the board receives its slow weights and store.
- **PS software**: olang cross-compiled to the ARM cores (an olang item, question 7) or, until then, a thin driver that
  moves buffers and the plan's descriptor table, with everything else in the plan export.

### 4.5 What needs care

- **Reciprocal weights** are read as `W` and `W^T`; one stored copy serves both (4.2). Updating them touches each word
  once.
- **Precision.** Activations need the fractional bits derived in 4.2 to settle to the life tolerance. Contrasts are
  differences of phase products separated by `O(beta)`: the PS computes them in the factored form from `dS` directly, in
  F32.
- **Eligibility doubles synapse memory** and adds a full pass per aroused moment; on the PS (DDR) this is free of
  pressure. A truncated rank-120 trace (2.5) pays only for blocks with `120 (m + n) < m n`, e.g. 401 408 weights of the
  784-512 block against 155 k numbers.
- **Variable latency**: the sweep count is data-dependent; a real-time body needs the sweep cap (the budget) and the
  refusal path, reported in a status register.
- **Timing closure** for a 64-input adder tree at 150 MHz needs pipelining; at 100 MHz it is routine.
- **Synthesis time** is minutes to tens of minutes per build, which is why the descriptor-driven engine is preferred to
  a circuit-specific netlist.

### 4.6 Toward spiking

The rate neuron is a leaky integrator, a LIF neuron without spikes, stepped in time - exactly the time-stepped op
DESIGN section 14 question 3 imagines. Replacing `s` by spike trains turns each GEMV into event-driven column
accumulation (additions, no multiplies), and the contrast into correlations of spike counts in the two phases; spiking
variants of equilibrium propagation exist in the literature (O'Connor, Gavves and Welling 2019; EqSpike, Martin et al.
2021 - to be checked before relying on them; neither could be reached from this environment, so what follows is built
from the LIF neuron's textbook dynamics and this document's own phases). On this board an accumulate-only lane costs LUTs, not a DSP, so the 220-DSP
limit stops being the cap on parallelism and BRAM ports become it. The cost is variance: estimating a small contrast
from spike counts needs long integration windows, multiplying the sweep count. The design consequence for oann is small
and worth taking now: the neuron model is a closed enum in the plan (`Neuron.Rate(...)`, later `Neuron.Lif(...)`), so the
settle loop, phases, contrast and moment loop do not change when spikes arrive.

**As built (phase 1, the user's "we want spiking set up").** `Neuron.Lif(leak, threshold, synapse, window)` is a
working variant, not a stub. A spiking circuit's Hidden and Readout neurons are all `Lif`; its inputs stay rate-coded
(`s = rho(d)`, folded into a constant current). One sweep is one time step:

```text
syn    <- syn + synapse (W . spikes_{t-1} - syn)        a synaptic current per neuron, through the same transport
I       = c + syn  (+ beta f(trace) on a nudged readout)
u      <- u + leak (I - u)                              leak = the step over the membrane time constant
spike   = u >= threshold ;  u <- u - threshold if spike  reset by subtraction, at most one spike a step
trace  <- trace + (4 / window) (spike - trace)          a running rate, what a nudge's force reads
```

A settle runs whole **windows**; at a window's end the activity `s` becomes the window's spike count per step, and a
row's residual is the largest change of a rate since the window before. It qualifies when every row's rates have
stopped moving to within the tolerance (default `2 / window`: two spikes); refusal, commit, warm starts, phases and
`Contrast` are the rate circuit's, read on spike counts. Two choices were forced by measurement: the force reads the
**rate trace**, updated every step, because a force recomputed once a window from the window's rates swung from
window to window where the rate curve is steep (residual stuck at 0.023); and reset is **by subtraction**, which
keeps the discrete-time rate within one step's period of the continuous-time formula instead of rounding the period
up to a whole step (a staircase rate curve, whose small nudges change no count at all).

`Rho.LifRate(leak, threshold)` is the rate model of these neurons: `s = 1 / (tau ln(v / (v - threshold)))` above the
threshold, 0 below, at most 1, with `tau = -1 / ln(1 - leak)` - the steady rate under the constant drive `v`. A rate
circuit of `LifRate` neurons is the mean field the spiking one is compared with.

**Limits, stated.** No certificate: the rate curve's slope is unbounded at the threshold, so `Gain` is infinite and a
spiking settle has only its empirical window test. No refractory period, one spike a step at most, deterministic
neurons (no noise), inputs as constant currents rather than spike trains, the transport dense (a product on a 0/1
vector, not the event-driven accumulation of the board). Spike-count contrasts are noisy where the feedback that
carries them is weak - hidden neurons' contrasts need long windows; a neuron near its threshold fires on fluctuations
its rate model ignores; and a lesson costs `window`s of steps, about 1 000 steps at `window = 256` once warm.

---

## 5. Validation

Everything below runs as olang `test` blocks in `F64` (the reason `Circuit` is generic over `T`), on small circuits,
against quantities computed independently of the code under test. Nothing external is run.

### 5.1 Fixed points and the certificate

1. **A single neuron**: `v = w rho(v) + d`, compared with bisection on `g(v) = v - w rho(v) - d`, monotone when
   `|w| L < 1`.
2. **Linear neurons** (the test-only activation `rho(v) = v`): the settle must equal the direct solve of
   `(I - W) v = d + b` by Gaussian elimination, for random `W` with `rho_W < 1`.
3. **The certificate**: for random circuits with `kappa < 1` check (i) the same equilibrium from different starts,
   (ii) `||v_k - v*|| <= q^k / (1 - q) ||v_1 - v_0||` at every `k`, (iii) `||r_k|| <= q^k ||r_0||`, (iv) at the stop,
   `||v - v*|| <= tau / (1 - kappa)`, and (v) the sweep count at most `ceil(ln(tau / ||r_0||) / ln q)`. For circuits with
   strong positive feedback (`kappa >> 1`, bistable) the settle either returns a state whose residual meets `tau` or
   refuses; it never returns a state that does not qualify.
4. **Energy**: for symmetric `W`, central differences of `E` equal `-rho'(v) r`, `E` is non-increasing along small-`dt`
   sweeps, and the settled state is a stationary point of `E`.
5. **Closed-form inputs**: the equilibrium with input regions evaluated in closed form equals the one found by relaxing
   them as ordinary regions.
6. **Damping**: the equilibrium does not depend on `dt`; plain and damped settles agree within tolerance where both
   converge.

### 5.2 Learning rules against finite differences

With `tanh` (smooth everywhere), symmetric weights, `T = 0.25`:

1. **Supervised contrast**: the centered contrast with small `beta` and `tau` from the rule of 1.4, against central
   finite differences of `L(theta) = C(s_out(theta); y)` taken at the free fixed point, for weights, biases and the
   source projection. Sweeping `beta` over 1e-1 ... 1e-3: the error falls as `beta^2` (slope 2 on log-log) until it
   reaches the phase-error floor `eps / beta`; the uncentered estimate falls as `beta` (slope 1).
2. **The encoder gradient**: `-(s+ - s-) / (2 beta)` on a free neuron against the finite difference of `L` with respect to
   its drive, and through a source projection (the "settling inside, backprop outside" claim of 2.2).
3. **A deliberate violation**: with asymmetric directed weights the estimate must *not* match; the test documents that the
   theorem's condition is necessary.
4. **The policy score**: for a one-step menu, `E_a[contrast_a]` with the nudge toward each action `a`, weighted by `pi(a)`,
   must vanish, and the contrast of a single action must equal the finite difference of `T log pi(a)`.
5. **Three-factor expectation**: with the critic frozen at the exact mean reward `V = sum_a pi(a) r_a` (a baseline) and
   `gamma = 0`, the exact expectation of the actor's update over the actions, computed by enumeration,
   `sum_a pi(a) (r_a - V) T grad log pi(a)`, must equal `T grad J` with `J(theta) = sum_a pi(a) r_a`; compare with the
   finite difference of `J`.
6. **Eligibility**: the fused pass equals the explicit sum of past rank-2 terms weighted by `(gamma lam)^age`; lazy
   decay equals eager decay.

### 5.3 Memories and arousal

1. Read-after-write with a unit key and `rate = 1` returns the value exactly; orthogonal keys do not interfere;
   overlapping keys interfere by `dot(q, k)`; `F` decays by `phi` per moment; `C` approaches `v` as `(1 - c')^n`.
2. The sparse-code store: code norm 1 with exactly `k` active cells, a write at `rate = 1` followed by a read at the same
   input returns the target, a different input interferes only through shared active cells.
3. Arousal: the recursions checked against hand-computed sequences; the critical period and threshold transitions; a
   step change in reward raises `level` above `theta` within a few `1 / alpha_f` moments and it falls back.
4. **Calm moments change nothing**: the parameter region, moments and memories are byte-identical before and after a calm
   moment.

### 5.4 Sleep

1. The recurrent cell's gradient through the unrolled window against finite differences (the `Graph` op checks cover it).
2. After enough passes the slow weights alone reproduce the dreamed outputs on the cues within a tolerance; the dawn store
   holds the residual, so `slow + store` output on the cues is unchanged by consolidation (recall is preserved).
3. A small finite-state sequence task: before and after a night, accuracy of the slow weights alone on held-out sequences
   rises over nights; the store alone stays where its exposures put it. The expected outcome is stated before the run.

### 5.5 End to end

A contextual bandit and a reversal task (the best arm swaps). Against the analytic optimum (the best arm's reward rate)
over many seeds with confidence intervals: regret falls; arousal rises after a reversal and falls back; the fraction of
calm moments rises as the policy settles; a trace-pinning case (1.6) is included.

### 5.6 The board

Simulate the engine in olang with `type Q1_15 extends I16` (weights, activations) and an `I64` accumulator, rounding to
Q4.14: the simulated equilibrium differs from the F64 one by less than the bound `floor / (1 - kappa)` of 4.2 on the
fixtures above. On the board, the PL result must equal the simulation exactly (integer arithmetic, associative sums) and
the F64 result within the bound.

---

### 5.7 Results of phase 1 (2026-10-09)

All in F64, as `circuit.olang`'s tests, unless marked; errors are the largest over the elements checked.

- **5.1.1** One neuron, `v = w tanh(v) + d`, against bisection: within 1e-12 for `w` in {0.5, -0.8, 0.9, -0.3} and four
  drives, at `tau = 1e-14`; the gain is `|w|` exactly.
- **5.1.2** Linear neurons (seven relaxing, two lateral blocks, `kappa = 0.8`) against Gaussian elimination of
  `(I - W) v = d + b + W_in d_in`, assembled independently from the blocks: within 1e-12.
- **5.1.3** The certificate at `kappa = 0.8`: three random starts reach one equilibrium (1e-12); bounds (ii) and (iii)
  hold at every one of 79 sweeps; (iv) and (v) at `tau` = 1e-3, 1e-6, 1e-9. A bistable circuit (gain 6), 30 starts
  with budgets 0-29: some qualify, some refuse; every state returned qualifies, and every refusal leaves the committed
  state bit for bit. A **nudged** phase (squared error, `beta = 1`) steps at `dt = 1/2` and its residual shrinks at
  the certified `q_beta = 0.9` at every one of 59 sweeps.
- **5.1.4** Energy: central differences of `E` against `-rho'(v) r` within 1e-8; 200 sweeps at `dt = 0.05` never
  raise it; at the equilibrium its gradient is below 1e-9.
- **5.1.5** Inputs in closed form against relaxing them as ordinary regions: within 1e-12.
- **5.1.6** Damped (`dt = 1/2`, three attempts) and plain settles agree within 2e-12; a linear neuron with
  self-weight -1 swings forever at `dt = 1` (refused, residual 1) and lands on `d / 2` in one damped step.
- **5.2.1** The estimate against central differences of the free equilibrium's cost (three tanh inputs, four hidden
  with a lateral block, three cross-entropy readouts, two rows), relative to the largest element:

  | `beta` | 1e-1 | 1e-2 | 1e-3 |
  |---|---|---|---|
  | centered | 1.6e-3 | 1.6e-5 | 1.6e-7 |
  | uncentered | 6.9e-2 | 6.8e-3 | 6.8e-4 |

  slope 2 and slope 1 exactly; at `beta = 1e-3` with the rule's tolerance (1e-11, not a tight one) 1.6e-7.
- **5.2.2** The drive's gradient, on a hidden region and through the source projection of a tanh input region:
  1.9e-6 of the largest element, both.
- **5.2.3** With the feedback a one-way block of its own (asymmetric): 0.88 - the estimate is no gradient.
- **5.2.4** The policy score: the mean of the actions' contrasts under `pi` is 3.8e-7 of the largest, and an action's
  contrast against central differences of `T log pi(a)` 1.0e-6.
- **5.2.5** Three factors, the exact expected actor update with an exact baseline against `T grad J`: 7.3e-7.
- **5.2.6** The eligibility trace (`Eligibility`, lazy and fused) against the explicit sum of past contrasts weighted
  by `(gamma lambda)^age` over eleven moments: within 1e-14; lazy and eager decay within 1e-15.
- **Spiking** (4.6): one Lif neuron's rate under six constant drives within the derived bound of the analytic rate
  (e.g. 0.0455 against 0.0467 at drive 1.5, 0.274 against 0.281 at 6; none below the threshold). A small spiking
  circuit's free rates are within 0.009 of its `LifRate` rate model's, and its spike-count contrast agrees in sign on
  all 11 of the rate model's larger entries (the largest difference 0.23 of the largest entry).
- **End to end, XOR** (rate, F64, eight hidden): every pattern right and the cost 9.6e-5 after 400 lessons, 9 sweeps a
  lesson. **Spiking XOR** (window 256): every pattern right, cost 2.2e-5 after 200 lessons, the readouts on their
  targets to a spike; about 1 080 steps a lesson.
- **End to end, MNIST** (`examples/mnist_settle.olang`, F32, 784-128-10, batch 32, Adam 1e-3, `beta = 0.5`, test set):
  95.69% after one epoch, 96.73% after three, 97.48% after ten (97.49% at nine); about 40 sweeps a lesson for the three
  settles and 11-20 an answer; no lesson refused, though the gain rises to 15 - far outside the certified region,
  where only the residual test vouches for a settle. 784-256-10: 97.57% after five epochs. Projected back into the
  certified set after every step (`Restrain(0.9)`, what the board needs): 91.09% after three epochs, at 16 sweeps a
  lesson and 4 an answer - the certificate bounds each readout neuron's input to 0.9, which the cross-entropy at
  `T = 0.25` cannot separate ten classes well with; a deployed circuit wants a lower temperature or training aware of
  the bound. An epoch takes 8.7 s on the shared machine (backprop's 784-128-10 epoch: 2.4 s; 2.10 says where it goes).
- **End to end, spiking MNIST** (the same example with `window` 256: Lif hidden and readout neurons, squared error
  toward 0.2 spikes a step for the class, `beta = 1`, Adam 5e-3, 10 000 training samples): 89.73% after one pass,
  90.48% after two; about 1 870 steps a lesson and 780 an answer, 45-66 s a pass. Learning works on spike counts;
  it is slow and noisy beside the rate circuit, as 4.6 expects (long windows, many steps).

### 5.8 Results of phase 2 (2026-10-09)

`agent.olang`'s tests (F64), and `examples/bandit_settle.olang` (F32) for 5.5.

- **5.3.1 Memory.**
  - A read after a write at a unit key returns the value (1e-15).
  - An orthogonal key reads exactly 0.
  - An overlapping key reads `0.7 dot(q, k)` (1e-15).
  - The fast store fades by `phi` per moment.
  - Repeated writes bring the persistent store to the value as `(1 - c')^n` (1e-14).
- **5.3.3 Arousal.**
  - Three outcomes agree with values computed by hand, independently (1e-12).
  - In a step-change scenario the agent is aroused through the critical period, calm after it (level 0.99), aroused
    2 moments after the change, and calm again 142 moments later.
- **5.3.4 Calm moments** leave the parameters, Adam's state, the eligibility, the critic and both memories bit for bit
  as they were. The critic converges to the reward.
- **Checkpoints.** A restored agent lives exactly the saved one's next 100 moments: actions, parameters, eligibility,
  memories, arousal and draws, over a stretch of both calm and aroused moments. A used agent and a file that is no
  checkpoint are refused.
- **5.5, the contextual bandit**:
  - Setup: 4 contexts as overlapping random unit observations of 16; 4 arms; rewards of -1 or +1, the best arm paying
    with probability 0.8 and the others 0.1-0.5; 40 lives of 4 000 moments; mean with 95% interval. Regret is in
    probability of paying (a uniformly random choice: 0.375).
  - Regret: 0.190 +- 0.010 over the first 250 moments, 0.055 +- 0.016 over the next, then 0.03-0.05 to the end.
  - Calm moments: 7% in the critical period, then 82-93%. The arousal level falls from 7.8 to about 1.1.
  - A test with two contexts and deterministic rewards: every one of the last 499 moments right.
- **5.5, the reversal** (the same, each context's best and worst arms swapping at moment 2 000):
  - Regret: 0.042 over the 250 moments before the swap, 0.230 +- 0.023 over the 250 after, then 0.085, and back to
    0.065-0.075 by moment 4 000.
  - Calm moments: 87% before, 40% after the swap, back to 94-97%.
  - The arousal level: 1.13 before, 1.71 after the swap, back to 1.08.
- **5.5, trace pinning** (the bandit, where history tells nothing; context-trace gains 0, 1 and 8; 40 lives; the
  last 2 000 moments). The rate of repeating the last action when the context's best arm changed detects it:

  | | gain 0 | gain 1 (default) | gain 8 |
  |---|---|---|---|
  | memory on: regret | 0.025 | 0.046 | 0.047 |
  | memory on: repeat rate | 0.05 | 0.11 | 0.10 |
  | actor alone (memory gain 0): regret | 0.034 | 0.141 | 0.321 |
  | actor alone: repeat rate | 0.15 | 0.60 | 0.87 |

  The default trace already pins the actor partly, and a strong one nearly always repeats. The memory's direct drive
  on the readout masks most of it, because the trace acts on the hidden layer, which the memory bypasses.
- **With the document's own defaults** (decision 21):
  - The bandit's regret plateaued at 0.11, because the agent was calm for good from moment 100.
  - The reversal never recovered (0.53 to the end).
  - A faster actor (Adam 1e-2) collapsed onto one action whatever the context (repeat rate 0.85-0.93), even when always
    aroused.
  - Always aroused, the actor alone reaches regret 0.017 by moment 4 000 at Adam 3e-4.
- **What the gate does not do**: it detects change, not suboptimality. Disappointment is relative to a long-run level
  that adapts, so a lasting drop eventually reads as normal, and calm moments never explore. With the tuned values the
  learners relearn within the aroused stretch a change provokes, but there is a floor (regret about 0.03-0.05 in the
  bandit, a little higher after a reversal) where a still-imperfect policy no longer surprises the agent.

## 6. Decisions and open questions

**Decided**

1. **Order.** After the MNIST MLP, **transformers are built first and settling networks second.** Consequence: the
   kernels both need - the batch-1 GEMV path, the vectorized row reductions and softmax, and fast `exp`/`tanh` (section 3,
   items 1, 4, 6) - are built for the transformer round; the kernels only settling needs (`Ger` and the fused eligibility
   form, masked updates, absolute sums, top-k, the sparse-code products, CSR) follow in the second round.
2. **FPGA target: a PYNQ-Z2 board** (Zynq-7020: dual Cortex-A9 and programmable logic with about 53 k LUTs, 220 DSP
   slices and 140 BRAM36 blocks, about 4.9 Mbit; Vivado and Vitis HLS; PYNQ overlays driven from the ARM cores).
   Consequences in section 4: 16-bit fixed point on certified circuits, a 64-lane descriptor-driven engine, at most
   about 197 k resident 16-bit weights, learning on the ARM cores by default.
3. **Spiking set up from the start** (the user, answering open question 4: "We want spiking set up"): the neuron model
   is a closed enum with a working leaky integrate-and-fire variant (4.6), aimed at the PYNQ-Z2.

**Taken at implementation (phase 1, 2026-10-09)** - details within the design, each recorded where it applies:

4. **Where it lives**: `circuit.olang` inside oann (open question 2, as recommended).
5. **The gain counts recurrent synapses only** (1.2): a folded input block is a constant of the map, so it neither
   enters `kappa` nor is scaled by `Certify`, which multiplies every recurrent block by one factor. (The document had
   counted input blocks too; that bounds `|v|` for the board, which needs it as a separate check, but it is not the
   contraction's constant, and certifying with it would have shrunk the 784-wide input blocks to near nothing.)
6. **A phase nudged with `beta > 0` steps at `dt / (1 + beta lambda)`** (1.2), `lambda` the force's largest self-slope
   (`1 / (4 T)` cross-entropy, 1 squared error) - without it the default lesson (`beta = 0.5`, `T = 0.25`) refused
   every plus phase on XOR. `Rate(beta)` and `Budget(tolerance, beta)` use that step.
7. **Input regions** are linear by default (the drive is the activity) and have no bias; any region may take a drive.
8. **Handles and recording follow `Graph`**: `Plan(r)` draws the weights (Glorot-uniform, masked by a Bernoulli draw
   where `density < 1`), `Project` takes no generator; the method making a hidden region is `Hidden` (the sketch's
   `Region` would read as a constructor of the handle type).
9. **The double buffer is per phase, committed and candidate** (2.10); a sweep runs in place in the candidate.
10. **Damping continues** from where an attempt ended rather than restarting from the warm start.
11. **Tolerances, nudges and certificate targets are `F64`**, whatever `T` is, as std/linalg's `alpha`/`beta` are.
12. **`LessonTolerance` holds `kappa` at most 0.9** and `Budget` is twice the certified need (q at most 0.95), at least 128.
13. **The nudge** is per phase and applies to every readout by its own `Cost`; a cross-entropy readout reads
    `LabelData`, a squared-error one `TargetData`. The temperature lives in the readout's cost.
14. *(Superseded by decision 17.)* **`circuit.Adam`** steps the parameter region until oann's optimizers take a flat region.
15. **`DriveGrad`** is the gradient of the cost averaged over the rows, as `Contrast`'s is.
16. **Spiking** (4.6): rates as spike counts over windows; residual = a rate's change from the last window; the
    nudge reads a per-neuron rate trace (time constant a quarter window); reset by subtraction; a synaptic current
    per neuron; `LifRate` as the rate model; the default `LifNeuron(leak 0.1, threshold 1, synapse 0.25, window 256)`.

**Taken in phase 2 (2026-10-09)**

17. **optim takes flat regions**: `FlatAdamW` and `FlatSgd` step any `params`/`grads` pair of arrays, sharing their
    loops with `AdamW` and `Sgd` (decision 14's follow-up); `circuit.Adam` is gone, circuits expose `ParamData()` and
    `GradData()`.
18. **The agent's circuit** (2.11): observation and context trace one-way into the first hidden region, a reciprocal
    chain, the memories' read a drive on the readout, the critic on every hidden activity.
19. **One stream per agent** (open question 5): the moment loop needs no per-stream eligibility; many lives are many
    agents.
20. **The fast memory's decay is lazy**, so calm moments touch no memory (5.3.4). The context trace follows every
    moment (it is activity, not learning). The critic's trace only decays on calm moments.
21. **Tuned defaults**: actor's Adam `3e-4` (was `1e-3`), `c_0 = 0.2` (was 0.02), `alpha_s = 0.003` (was 0.01),
    `theta = 1.5` (was 2), measured on the reversal task (5.8). With the starting values the agent stopped learning at
    the end of its critical period and never recovered from a reversal.
22. **Rewards in 5.5 are -1 or +1**, the range 1.5 asks for. With 0 or 1 every tried arm looked better than an untried
    one (the memory stores rewards, an untried arm reads 0), which favoured repeating.
23. **A terminal outcome** clears the eligibility and the critic's trace after learning from it; the context trace
    persists.
24. **Arousal reads the clipped TD error**, and `V(x_prev)` is evaluated with the critic as it is when the outcome
    arrives.
25. **Refusals** (2.11): a refused free settle refuses the moment; a refused nudged phase or re-settle keeps the
    moment, counted.
26. **The agent owns its generator** (made from a seed), so a checkpoint can carry it as the seed and the numbers drawn.
    std/rand exposes no state, so a restore replays the draws and goes into a new agent of the same shape and seed.
27. **Checkpoints** are written in the safetensors layout, with every tensor stored as F64 (2.11).
28. **The critic's weights are F64** whatever `T` is.
29. **The circuit is made by a function**, working around repro/ctorpush.
30. **Pinning is detected** by the rate of repeating the last action when the best arm changed (5.8).

**Open questions**

1. **Scope and order within settling networks.** Recommended: the circuit and the agent (phases 1-2) first; the
   sparse-code store, readout groups and sparse connectivity next; the consolidator and sleep last, on the existing
   `Graph`.
2. *(Taken: inside oann - decision 4.)* **Where it lives.** Modules inside oann (`circuit`, `agent`, `memory`,
   `arousal`) sharing `Matrix` and the planner, or a separate package depending on oann? Recommended: inside oann.
3. **Vector math in olang.** Write fast `tanh`/`exp` approximations in the matrix library (no language change), or take up
   vectorized math functions as an olang question (X8 now forbids replacing a library call)? Recommended: library now;
   the transformer round is the first real test, and the olang question is raised if the library version is not fast
   enough there.
4. *(Answered: yes - decision 3.)* **Spiking.** Should the neuron model be a pluggable enum from the first version
   (recommended; costs one `match`), and is a LIF model with spike-count contrasts the spiking direction you mean?
5. *(Taken: `B = 1` - decision 19.)* **Batched lives.** Is a single continuing stream enough at first, or are
   batched streams (`B` rows sharing weights, per-row eligibility) needed early for faster experiments? Recommended:
   `B = 1` first; the arena already plans `B` rows.
6. **Learning on the board.** PS-only learning with quantized copies pushed to the PL (the default of 4.4), or the fused
   on-chip update pass for small cores from the start? Recommended: PS-only first.
7. **ARM-side software.** Cross-compile olang to the board's ARM cores (an olang item: `TargetArch` is the host's today),
   or keep the PS side a thin driver with all logic in the exported plan? Recommended: raise the olang item when the
   board work starts; thin driver until then.
8. **Exploration in calm moments** (from 5.8). The arousal gate detects change, not suboptimality, so a policy that is
   still imperfect when it stops surprising the agent stays so: a regret floor of about 0.03-0.05 in the bandit. Two
   ways out, both changes to the model: sample rather than act greedily in calm moments (exploration without learning,
   at no cost to calm's "no synapse touched"), or let arousal also rise with the policy's entropy. Recommended: try the
   first. Default: as specified (greedy when calm).

---

## References

Settling and energy
- Hopfield, J. J. (1984). Neurons with graded response have collective computational properties like those of two-state
  neurons. *PNAS* 81.
- Scellier, B., Bengio, Y. (2017). Equilibrium propagation: bridging the gap between energy-based models and
  backpropagation. *Frontiers in Computational Neuroscience* 11.
- Laborieux, A., Ernoult, M., Scellier, B., Bengio, Y., Grollier, J., Querlioz, D. (2021). Scaling equilibrium
  propagation to deep ConvNets by drastically reducing its gradient estimator bias. *Frontiers in Neuroscience* 15.
- O'Connor, P., Gavves, E., Welling, M. (2019). Training a spiking neural network with equilibrium propagation. *AISTATS*.
  (to be checked)
- Martin, E. et al. (2021). EqSpike: spike-driven equilibrium propagation for neuromorphic implementations. *iScience* 24.
  (to be checked)
- Glorot, X., Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.
- Kingma, D. P., Ba, J. (2015). Adam: a method for stochastic optimization. *ICLR*.

Reward learning
- Sutton, R. S., Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.
- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning* 3.
- Barto, A. G., Sutton, R. S., Anderson, C. W. (1983). Neuronlike adaptive elements that can solve difficult learning
  control problems. *IEEE Trans. Systems, Man, and Cybernetics* 13.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning.
  *Machine Learning* 8.
- Konda, V. R., Tsitsiklis, J. N. (2000). Actor-critic algorithms. *NIPS* 12.
- Frémaux, N., Gerstner, W. (2016). Neuromodulated spike-timing-dependent plasticity, and theory of three-factor
  learning rules. *Frontiers in Neural Circuits* 9.
- Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of STDP and dopamine signaling. *Cerebral
  Cortex* 17.
- Schultz, W., Dayan, P., Montague, P. R. (1997). A neural substrate of prediction and reward. *Science* 275.
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature* 518.

Memory, arousal, consolidation
- Kohonen, T. (1972). Correlation matrix memories. *IEEE Trans. Computers* C-21. Anderson, J. A. (1972). A simple neural
  network generating an interactive memory. *Mathematical Biosciences* 14. Widrow, B., Hoff, M. E. (1960). Adaptive
  switching circuits. *IRE WESCON Convention Record*.
- Blundell, C. et al. (2016). Model-free episodic control. arXiv.
- Elman, J. L. (1990). Finding structure in time. *Cognitive Science* 14.
- Marr, D. (1969). A theory of cerebellar cortex. *J. Physiology* 202. Albus, J. S. (1971). A theory of cerebellar
  function. *Mathematical Biosciences* 10. Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
- Dasgupta, S., Stevens, C. F., Navlakha, S. (2017). A neural algorithm for a fundamental computing problem. *Science* 358.
- McClelland, J. L., McNaughton, B. L., O'Reilly, R. C. (1995). Why there are complementary learning systems in the
  hippocampus and neocortex. *Psychological Review* 102.
- McGaugh, J. L. (2004). The amygdala modulates the consolidation of memories of emotionally arousing experiences.
  *Annual Review of Neuroscience* 27.
- Pearce, J. M., Hall, G. (1980). A model for Pavlovian learning: variations in the effectiveness of conditioned but not
  of unconditioned stimuli. *Psychological Review* 87.
- Yu, A. J., Dayan, P. (2005). Uncertainty, neuromodulation, and attention. *Neuron* 46. Aston-Jones, G., Cohen, J. D.
  (2005). An integrative theory of locus coeruleus-norepinephrine function. *Annual Review of Neuroscience* 28. Doya, K.
  (2002). Metalearning and neuromodulation. *Neural Networks* 15. Bear, M. F., Singer, W. (1986). Modulation of visual
  cortical plasticity by acetylcholine and noradrenaline. *Nature* 320.
- Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science* 7. Wilson, M. A.,
  McNaughton, B. L. (1994). Reactivation of hippocampal ensemble memories during sleep. *Science* 265. Hinton, G. E.,
  Plaut, D. C. (1987). Using fast weights to deblur old memories. *Proc. Cognitive Science Society*.
- Cho, K. et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation.
  *EMNLP*. Bradbury, J. et al. (2017). Quasi-recurrent neural networks. *ICLR*. Werbos, P. J. (1990). Backpropagation
  through time: what it does and how to do it. *Proc. IEEE* 78.

Hardware
- Xilinx/AMD. Zynq-7000 SoC Data Sheet: Overview (DS190); Zynq-7000 SoC Technical Reference Manual (UG585); 7 Series DSP48E1
  Slice User Guide (UG479); 7 Series FPGAs Memory Resources User Guide (UG473).
