# Internal API (Unexported)

```@setup internal
using Optim
```

The following symbols are internal helpers or implementation details. They are documented here for
contributors and advanced users. These APIs may change without notice.

```@docs
PyBayesOpt._toscale!
PyBayesOpt._to01!
PyBayesOpt.dict2vec
PyBayesOpt.vec2pairs
PyBayesOpt.bounds2pairs
PyBayesOpt.x00i
PyBayesOpt.get
PyBayesOpt.BayesianOptimizationResult
PyBayesOpt.AbstractBenchmarkFunction
Optim.optimize
Optim.summary
```
