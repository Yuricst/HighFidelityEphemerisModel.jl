# Benchmarking N-body Jacobian


## N-body Jacobian with analytical method

```julia
@benchmark HighFidelityEphemerisModel.dfdx_Nbody_SPICE([1.0, 0.0, 0.3, 0.5, 1.0, 0.0], 0.0, , 0.0)
```
```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  15.208 μs …  38.875 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     15.417 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   15.473 μs ± 318.306 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

         ▄  █  ▆ ▄  ▂                                           
  ▂▁▃▁▁▆▁█▁▁█▁▁█▁█▁▁█▁█▁▁▆▁▁▅▁▄▁▁▃▁▃▁▁▃▁▁▂▁▂▁▁▂▁▂▁▁▂▁▁▂▁▂▁▁▂▁▂ ▃
  15.2 μs         Histogram: frequency by time         16.2 μs <

 Memory estimate: 1.88 KiB, allocs estimate: 46.
```


## N-body Jacobian with ForwardDiff

```julia
@benchmark HighFidelityEphemerisModel.eom_jacobian_fd(HighFidelityEphemerisModel.eom_Nbody_SPICE, [1.0, 0.0, 0.3, 0.5, 1.0, 0.0], 0.0, , 0.0)
```
```
BenchmarkTools.Trial: 10000 samples with 4 evaluations.
 Range (min … max):  7.271 μs …  1.713 ms  ┊ GC (min … max): 0.00% … 99.03%
 Time  (median):     7.729 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   8.108 μs ± 23.768 μs  ┊ GC (mean ± σ):  4.12% ±  1.40%

                    ▂█▂                                       
  ▁▁▂▂▅▅▆▆▆▇▄▃▂▂▃▅▆█████▆▄▄▂▂▂▃▃▄▄▅▇▄▄▄▃▃▂▂▂▁▂▁▂▂▂▂▁▁▁▁▁▁▁▁▁ ▃
  7.27 μs        Histogram: frequency by time         8.6 μs <

 Memory estimate: 6.83 KiB, allocs estimate: 53.
```

