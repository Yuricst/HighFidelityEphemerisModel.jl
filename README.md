<p align="center">
  <img src="docs/src/assets/logo.png" alt="HighFidelityEphemerisModel.jl Logo" width="75%"/>
</p>


<p align="center">
  <a href="https://yuricst.github.io/HighFidelityEphemerisModel.jl/">
    <img src="https://github.com/Yuricst/HighFidelityEphemerisModel.jl/actions/workflows/docs.yml/badge.svg" alt="docs workflow"/>
  </a>
  <a href="https://github.com/Yuricst/HighFidelityEphemerisModel.jl/actions/workflows/test.yml">
    <img src="https://github.com/Yuricst/HighFidelityEphemerisModel.jl/actions/workflows/test.yml/badge.svg" alt="test workflow"/>
  </a>
  <a href="https://app.codecov.io/gh/Yuricst/HighFidelityEphemerisModel.jl">
    <img src="https://codecov.io/gh/Yuricst/HighFidelityEphemerisModel.jl/branch/main/graph/badge.svg" alt="codecov"/>
  </a>
  <!-- <a href="https://yuricst.github.io/HighFidelityEphemerisModel.jl/">📚Read the docs📚</a> -->
</p>

`HighFidelityEphemerisModel.jl` provides equations of motion for high-fidelity ephemeris model dynamics compatible with the [`OrdinaryDiffEq.jl`](https://github.com/SciML/OrdinaryDiffEq.jl) ecosystem.

What `HighFidelityEphemerisModel.jl` contains:
- full-ephemeris equations of motion relevant for astrodynamics
- Cartesian and Gauss variational equation propagation using modified equinoctial, classical Keplerian, and ordinary equinoctial elements
- callback conditions for common astrodynamics events (e.g. detection of osculating true anomaly)
- SPICE-free ephemerides via [`Ephemerides.jl`](https://github.com/JuliaSpaceMissionDesign/Ephemerides.jl) (`EphemeridesParameters`), to define equations of motion compatible with `EnsembleThreads` & automatic differentiation, e.g. `ForwardDiff`

What `HighFidelityEphemerisModel.jl` is *not*:
- not an integrator, i.e. there are no integration schemes (e.g. Runge-Kutta algorithms, step-correction, event detection features, etc.) implemented (at least for now)

We strive for minimal dependencies (listed in `Project.toml`), consisting of: `AstrodynamicsCore`, `Ephemerides`, `ForwardDiff`, `FrameTransformations`, `LinearAlgebra`, `OrdinaryDiffEq`, `Printf`, `SPICE`.

Cartesian propagation remains the general formulation. The GVE interfaces reuse
the same inertial HFEM force models and are available for N-body and N-body plus
spherical-harmonic propagation with SPICE and Ephemerides parameter
backends. See the [Gauss variational equations tutorial](examples/gve.ipynb)
for state conventions, supported domains, examples, and formulation-selection
guidance.



## Install

### From the Registry

```julia
] add HighFidelityEphemerisModel
```

### Checkout the repo

1. `git clone` this repositiory
2. In your project directory, add:

```julia-repl
] dev ./path/to/HighFidelityEphemerisModel.jl
```

3. To run tests, `cd` to the root of this repository, then

```julia-repl
] activate .
test
```


## Examples

For further details, see the [docs](https://yuricst.github.io/HighFidelityEphemerisModel.jl/).

### N-body Dynamics

```julia
using HighFidelityEphemerisModel
using LinearAlgebra
using OrdinaryDiffEq
using SPICE

# load SPICE kernels
spice_dir = ENV["SPICE"]
furnsh(joinpath(spice_dir, "lsk", "naif0012.tls"))
furnsh(joinpath(spice_dir, "spk", "de440.bsp"))
furnsh(joinpath(spice_dir, "pck", "gm_de440.tpc"))
furnsh(joinpath(spice_dir, "pck", "moon_pa_de440_200625.bpc"))
furnsh(joinpath(spice_dir, "fk", "moon_de440_250416.tf"))

naif_ids = ["301", "399", "10"]        # NAIF IDs of bodies to be included; first ID is of the central body
GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]      # in km^3/s^2
naif_frame = "J2000"
abcorr = "NONE"
DU = 1e5                               # canonical distance unit, in km

nmax = 4                               # using up to 4-by-4 spherical harmonics
filepath_spherical_harmonics = joinpath(pkgdir(HighFidelityEphemerisModel), "data", "luna", "gggrx_1200l_sha_20x20.tab")

et0 = str2et("2026-01-05T00:00:00")    # reference epoch

# SpiceParameters queries SPICE at runtime; for a SPICE-free, thread-safe and
# ForwardDiff-compatible backend use EphemeridesParameters instead, adding e.g.
# `ephemerides_files = [joinpath(spice_dir, "spk", "de440.bsp"), joinpath(spice_dir, "pck", "moon_pa_de440_200625.bpc")]`
parameters = HighFidelityEphemerisModel.SpiceParameters(
    et0, DU, GMs, naif_ids, naif_frame, abcorr;
    filepath_spherical_harmonics = filepath_spherical_harmonics,
    nmax = nmax,
    frame_PCPF = "MOON_PA",
    include_srp = true,
    srp_Cr = 1.15,
    srp_Am = 0.002,
    srp_P0 = 4.56e-6,
)

# construct & solve ODE problem
x0 = [1.0, 0.0, 0.3, 0.5, 1.0, 0.0]         # initial state in DU & DU/TU
tspan = (0.0, 7*86400/parameters.TU)
prob = ODEProblem(HighFidelityEphemerisModel.eom_NbodySH!, x0, tspan, parameters)   # dispatches on the parameter type
sol = solve(prob, Vern8(), reltol=1e-14, abstol=1e-14)

# propagate both state & STM
x0_stm = [x0; reshape(I(6),36)]
prob = ODEProblem(HighFidelityEphemerisModel.eom_stm_NbodySH_SPICE_fd!, x0_stm, tspan, parameters)   # or eom_stm_NbodySH_Ephemerides_fd! with EphemeridesParameters
sol = solve(prob, Vern8(), reltol=1e-14, abstol=1e-14)
```