# Basics

We will now go over how to use the equations of motion to propagate some initial state until some future time, i.e. 

```math
\boldsymbol{x}(t_f)
= 
\boldsymbol{x}(t_0) + 
\int_{t_0}^{t_f} \boldsymbol{f}(x(t),t) \mathrm{d}t
```

!!! tip

    Before we get started, make sure you have relevant SPICE kernels downloaded locally in your system. 
    Most generic kernels are available on the JPL NAIF website: [https://naif.jpl.nasa.gov/pub/naif/generic_kernels/](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/).


## Initializing parameters

We first need to define the parameter struct to be parsed as argument to the equations of motion.

Below is a SPICE-backed example using the generic EOM names. The concrete parameter type selects the backend.

```julia
using OrdinaryDiffEq
using HighFidelityEphemerisModel

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
filepath_spherical_harmonics = "HighFidelityEphemerisModel.jl/data/luna/gggrx_1200l_sha_20x20.tab"

et0 = str2et("2026-01-05T00:00:00")    # reference epoch
parameters = HighFidelityEphemerisModel.SpiceParameters(
    et0, DU, GMs, naif_ids, naif_frame, abcorr;
    filepath_spherical_harmonics = filepath_spherical_harmonics,
    nmax = nmax,
    frame_PCPF = "MOON_PA",
    include_srp = true,
    srp_Cr = 1.15,                     # SRP radiation pressure coefficient
    srp_Am = 0.002,                    # SRP area-to-mass ratio in m^2/kg
)
```

!!! note

    - NAIF body IDs are defined according to: [https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html)
    - use `SpiceParameters`, `EphemeridesParameters`, or `InterpParameters` to choose the backend
    - use `eom_Nbody!`/`eom_Nbody` or `eom_NbodySH!`/`eom_NbodySH` for normal propagation
    - `HighFidelityEphemerisModelParameters(...)` remains available only as a backward-compatible constructor
    - if using `Nbody` dynamics instead of `NbodySH`, you do not need to provide `filepath_spherical_harmonics`, `nmax`, and `frame_PCPF`
    - if excluding SRP terms, set `include_srp = false` (then, `srp_Cr` and `srp_Am` won't be used, so they can be removed too)


## Solving an Initial Value Problem

The integration is done with the `OrdinaryDiffEq.jl` library (or equivalently with `DifferentialEquations.jl`).

```julia
# initial state (in canonical scale)
x0 = [1.05, 0.0, 0.3, 0.5, 1.0, 0.0]

# time span (in canonical scale)
tspan = (0.0, 6 * 3600/parameters.TU)

# solve
prob = ODEProblem(HighFidelityEphemerisModel.eom_NbodySH!, x0, tspan, parameters)
sol = solve(prob, Vern8(), reltol=1e-14, abstol=1e-14)
```

## Propagate the STM

Suppose we now also want to integrate the STM, i.e. we want to also solve the Matrix-valued IVP

```math
\begin{aligned}
&\dot{\boldsymbol{\Phi}}(t,t_0) = \dfrac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}} \boldsymbol{\Phi}(t,t_0)
\\
&\boldsymbol{\Phi}(t_0,t_0) = \boldsymbol{I}_6
\end{aligned}
```

Then, we just need to do

```julia
x0_stm = [x0; reshape(I(6),36)]  # initial augmented state, flattened

# solve with SPICE
prob_spice = ODEProblem(HighFidelityEphemerisModel.eom_stm_NbodySH_SPICE_fd!, x0_stm, tspan, parameters)
sol_spice = solve(prob_spice, Vern8(), reltol=1e-14, abstol=1e-14)
```

Now, to extract the STM that maps from `tspan[1]` to `tspan[2]`, 

```julia
x_aug_tf = sol_spice.u[end]             # final state + flattened STM
x_tf   = x_aug_tf[1:6]                  # final state
STM_tf = reshape(x_aug_tf[7:42],6,6)   # final STM
```