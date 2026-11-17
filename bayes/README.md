# bayes

Bayesian inference over Gaussian beliefs: sequential updates after each event (rating models, state-space filters),
quadrature, variational inference, MCMC, empirical Bayes and model checking. Everything works in F64 on plain arrays;
the gradient-based methods (ADVI, HMC) take their gradients from oann's recorded graph (`ad/`) and step with
`nn/optim`'s Adam.

| module | what it does |
|---|---|
| `dense` | small dense matrices as row-major `Array<F64>`: Cholesky factor, triangular solves, products, inverse, log determinant |
| `gauss` | `Gaussian` (mean, full covariance) and `Diag` (mean, variances - for large models): blocks put and taken (`Put`, `Take`, `Marginal`), conditioning on components (`Condition`), a linear observation's Kalman update (`Observe`), sampling, log density, entropy, KL; the normal cdf, log cdf, Mills ratio and quantile |
| `laplace` | `Likelihood` of a scalar predictor - normal, logit, probit, binomial, Poisson, ordered logistic - with its log density, slope and curvature; `Update`: the Laplace approximation of the posterior after observations of a `Predictor` (linear `Rows`, or any smooth link), by damped Newton steps on the parameters they touch; `UpdateDiag`: one event of a factorised belief (assumed-density filtering); `Predictive`: an observation's log predictive density; `SigmoidMean`: a sigmoid's mean under a Gaussian by the probit approximation |
| `state` | `Drift`: a random walk or an AR(1) pull, its variance growing with the time between events; `Model`: a linear-Gaussian state-space model with its Kalman `Filter` (and the data's log likelihood) and RTS `Smooth` |
| `quadrature` | Gauss-Hermite rules, 1-d (`Rule.Expect`) and tensor-product n-d under a correlated Gaussian (`Expect`); the unscented transform (`Unscented`) |
| `target` | `Differentiable`: a log joint and its gradient; `GraphTarget`: one from an `ad/graph` graph whose parameters are the point and whose loss is the negated log joint; `Logistic`: Bayesian logistic regression recorded on a graph |
| `vi` | ADVI: `MeanField` and `FullRank` Gaussian approximations, reparameterised gradients through a `Differentiable`, Adam on the variational parameters |
| `mcmc` | random-walk `Metropolis` over any `Density`; `Hmc` over any `Differentiable` (dual-averaged step, diagonal metric from warm-up, trajectory length by the no-U-turn criterion during warm-up); `Ess` and `SplitRHat` |
| `empirical` | `Events` and `Prequential`: the sequential predictive log score of an assumed-density filter; `Golden` and `Coordinate` to choose hyperparameters by it (or by a Kalman filter's `LogLik`) |
| `check` | `Calibration`: coverage of the central 50% and 90% predictive intervals, the PIT histogram and its chi-square from flat, the mean log score and the Brier score |

## A rating filter

Each item's latent strength is a component of a `gauss.Diag`. An event (a pairwise comparison, any observation) is a linear predictor over the
components it touches, `eta = coef . x[idx] + offset`, observed under a likelihood:

```
import "github.com/osarv/oann/bayes/gauss"
import "github.com/osarv/oann/bayes/laplace"
import "github.com/osarv/oann/bayes/state"

b := gauss.Diag.Isotropic(items, 1.0)
walk := state.Drift.RandomWalk(0.01)                     # variance added per unit of time
probit := laplace.Likelihood(laplace.Family.Probit)
# item a preferred to item c: move both by the time since each was last observed, then update
walk.ApplyDiag(b, a, sinceA)
walk.ApplyDiag(b, c, sinceC)
logp := try laplace.UpdateDiag(b, Array<I64>[a, c], Array<F64>[1.0, -1.0], 0.0, probit, 1.0)
```

`UpdateDiag` joins the touched components into one Gaussian, finds the posterior's mode by Newton's method with the
likelihood's exact curvature (one step for the normal likelihood, where it is the Kalman update), takes the covariance
at the mode (Laplace), and writes the marginals back. Each Newton step costs one linearization of the predictor and one
Kalman update of `n` pseudo-observations, O(d n (d + n)) with the Jacobian's zeros skipped; the prior's precision is
never formed (every iterate is `m + P u`, so the prior's quadratic form is `u . (z - m)`). `bench/laplace.olang` times
a 38-component update of two binomial observations. With a full `gauss.Gaussian` and `laplace.Update` the
correlations are kept. `empirical.Prequential` runs the same filter over an `Events` sequence and returns the sum of
the log predictive densities; maximising it with `empirical.Golden` chooses the drift's rate.

## Gradient-based inference

A model is a graph whose parameter region is the point and whose loss is the negated log joint:

```
g := graph.Graph<F64>(1)
nodes := target.Logistic.Record(g, n, d, 4.0)            # or any model built from the graph's operations
try g.Plan(r)
nodes.Set(g, x, y)
t := target.GraphTarget(g, nodes.Loss)
fit := try vi.Fit.FullRank(t, r, steps = 3000, samples = 4)   # fit.Q, fit.Elbo
chain := mcmc.Chain.Hmc(t, fit.Q.Mean, 2000, 1000, r)    # chain.Mean(i), chain.Sd(i), mcmc.Ess(chain.Column(i))
```

## Agreement checks

Where an exact answer exists the tests hold each method to it: the Kalman filter's last belief, RTS smoother and log
likelihood against the batch Gaussian posterior over every state; the normal likelihood's update against the conjugate
posterior; every family's Laplace mode and curvature against the stationarity conditions; the prequential score of a
normal filter against the Kalman log likelihood; Laplace and ADVI against HMC on a small logistic regression; HMC and
Metropolis against a correlated normal; full-rank ADVI against a Gaussian target.

```
olang test bayes/*.olang
```
