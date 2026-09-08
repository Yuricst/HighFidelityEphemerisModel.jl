# `HighFidelityEphemerisModel.jl`: High-Fidelity Ephemeris Model for Astrodynamics

`HighFidelityEphemerisModel.jl` is a minimal implementation of high-fidelity ephemeris model dynamics compatible with the [`OrdinaryDiffEq.jl`](https://github.com/SciML/OrdinaryDiffEq.jl) ecosystem[...]

![Lunar NRHO illustration](https://raw.githubusercontent.com/Yuricst/HighFidelityEphemerisModel.jl/main/demos/plots/demo_NRHO_deviations.png)

What `HighFidelityEphemerisModel.jl` contains:
- full-ephemeris equations of motion relevant for astrodynamics
- callback conditions for common astrodynamics events (e.g. detection of osculating true anomaly)
- SPICE, Ephemerides.jl, and legacy interpolated-ephemeris parameter backends

The preferred propagation API uses `SpiceParameters`, `EphemeridesParameters`, or `InterpParameters` together with generic EOM names such as `eom_Nbody!` and `eom_NbodySH!`. `HighFidelityEphemeris[...]

What `HighFidelityEphemerisModel.jl` is *not*:
- not an integrator, i.e. there are no integration schemes (e.g. Runge-Kutta algorithms, step-correction, event detection features, etc.) impemented (at least for now)

We strive for minimal dependencies (listed in `Project.toml`), consisting of: `Dierckx`, `Ephemerides`, `ForwardDiff`, `FrameTransformations`, `Interpolations`, `LinearAlgebra`, `OrdinaryDiffEq`, [...]


## Install

### From the Registry

```julia
] add HighFidelityEphemerisModel
```

### Checkout the repo

1. `git clone` this repositiory
2. In your project directory, add:

```julia-repl
pkg> dev ./path/to/HighFidelityEphemerisModel.jl
```

3. To run tests, `cd` to the root of this repository, then

```julia-repl
(@v1.10) pkg> activate .
(HighFidelityEphemerisModel) pkg> test
```

Documentation is built and deployed to [GitHub Pages](https://yuricst.github.io/HighFidelityEphemerisModel.jl/) by the [docs workflow](https://github.com/Yuricst/HighFidelityEphemerisModel.jl/acti[...]

```julia-repl
pkg> activate docs
(docs) pkg> instantiate
julia> include("make.jl")
```


## Tutorials

- [Overview](@ref)
- [Basics](@ref)
- [Perturbations](@ref)
- [Jacobians & Hessians](@ref)
- [ODE Solutions to SPK Files](@ref)
