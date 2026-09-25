# `HighFidelityEphemerisModel.jl`: High-Fidelity Ephemeris Model for Astrodynamics

`HighFidelityEphemerisModel.jl` is a minimal implementation of high-fidelity ephemeris model dynamics compatible with the [`OrdinaryDiffEq.jl`](https://github.com/SciML/OrdinaryDiffEq.jl) ecosystem.

![Lunar NRHO illustration](https://raw.githubusercontent.com/Yuricst/HighFidelityEphemerisModel.jl/main/demos/plots/demo_NRHO_deviations.png)

What `HighFidelityEphemerisModel.jl` contains:
- full-ephemeris equations of motion relevant for astrodynamics
- Cartesian and Gauss variational equation propagation in modified equinoctial, classical Keplerian, and ordinary equinoctial elements
- callback conditions for common astrodynamics events (e.g. detection of osculating true anomaly)
- SPICE, Ephemerides.jl, and legacy interpolated-ephemeris parameter backends

The preferred propagation API uses `SpiceParameters`, `EphemeridesParameters`, or `InterpParameters` together with generic EOM names such as `eom_Nbody!` and `eom_NbodySH!`.

Cartesian states remain the general/default representation. Gauss variational
equations are available when propagation in orbital elements is useful; see
[Gauss variational equations](@ref) for conventions, domains, backend support,
and examples.

What `HighFidelityEphemerisModel.jl` is *not*:
- not an integrator, i.e. there are no integration schemes (e.g. Runge-Kutta algorithms, step-correction, event detection features, etc.) impemented (at least for now)


## Install

The package is available on Julia registry: 

```julia
] add HighFidelityEphemerisModel
```


## Tutorials

- [Overview](@ref)
- [Basics](@ref)
- [Perturbations](@ref)
- [Gauss variational equations](@ref)
- [Jacobians & Hessians](@ref)
- [ODE Solutions to SPK Files](@ref)
