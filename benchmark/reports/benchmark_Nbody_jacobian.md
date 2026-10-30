# Benchmarking N-body Jacobian


## N-body Jacobian with ForwardDiff

```julia
@benchmark HighFidelityEphemerisModel.eom_jacobian_fd(HighFidelityEphemerisModel.eom_Nbody, x0, 0.0, parameters, 0.0)
```
```
BenchmarkTools.Trial: 10000 samples with 4 evaluations per sample.
 Range (min … max):  7.600 μs …  1.205 ms  ┊ GC (min … max): 0.00% … 97.92%
 Time  (median):     7.800 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   8.266 μs ± 17.485 μs  ┊ GC (mean ± σ):  3.91% ±  2.32%

  ▂▅▇██▇▆▅▃▁                                                 ▂
  ████████████▇███▇█▆▇▆▄▅▄▄▆▅▅▄▆▃▅▅▄▅▆▆▇▇▇▇▇▇▇█▇▆▆▆▇▇▆▆▄▄▃▄▅ █
  7.6 μs       Histogram: log(frequency) by time     10.5 μs <

 Memory estimate: 6.83 KiB, allocs estimate: 53.
```

