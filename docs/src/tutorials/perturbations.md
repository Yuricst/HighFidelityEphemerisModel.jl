# Perturbations

`HighFidelityEphemerisModel.jl` augments the central two-body term with optional perturbations:

- third-body gravity from other massive bodies
- spherical harmonics of the central body
- solar radiation pressure (SRP)
- atmospheric drag

Mathematical definitions appear in the [Overview](@ref "Overview" overview.md). This page shows how to enable each term in `HighFidelityEphemerisModelParameters` and propagate with `OrdinaryDiffEq.jl`.

!!! tip

    Before running the examples, download SPICE kernels and set `ENV["SPICE"]` to your kernel directory, or `furnsh` the files under `spice/test` in this repository. Generic kernels: [https://naif.jpl.nasa.gov/pub/naif/generic_kernels/](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/).

!!! note

    The **first** entry in `naif_ids` is always the central body. All positions and velocities in the state vector are expressed in canonical units (`DU`, `DU/TU`) relative to that body.


## Shared setup

The snippets below assume the following preamble (load once per Julia session):

```julia
using SPICE
using OrdinaryDiffEq
using HighFidelityEphemerisModel

# load SPICE kernels
spice_dir = ENV["SPICE"]
furnsh(joinpath(spice_dir, "lsk", "naif0012.tls"))
furnsh(joinpath(spice_dir, "spk", "de440.bsp"))
furnsh(joinpath(spice_dir, "pck", "gm_de440.tpc"))
furnsh(joinpath(spice_dir, "pck", "moon_pa_de440_200625.bpc"))
furnsh(joinpath(spice_dir, "fk", "moon_de440_250416.tf"))

naif_frame = "J2000"
abcorr = "NONE"
et0 = str2et("2026-01-05T00:00:00")

# typical initial state and short propagation window (canonical units)
x0 = [1.05, 0.0, 0.3, 0.5, 1.0, 0.0]
```


## Third-body perturbations

Third-body accelerations are included automatically for every body listed in `naif_ids` after the central body. Ephemerides are queried from SPICE (`_SPICE` EOMs) or from pre-interpolated tables (`_Interp` EOMs).

```julia
naif_ids = ["301", "399", "10"]   # Moon, Earth, Sun (301 = central body)
GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
DU = 1e5

parameters = HighFidelityEphemerisModelParameters(
    et0, DU, GMs, naif_ids, naif_frame, abcorr,
)

tspan = (0.0, 6 * 3600 / parameters.TU)
prob = ODEProblem(eom_Nbody_SPICE!, x0, tspan, parameters)
sol = solve(prob, Vern7(), reltol=1e-12, abstol=1e-12)
sol.u[end]
```

!!! note

    NAIF body IDs: [https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html)


## Spherical harmonics

Use the `NbodySH` equations of motion when the gravity field of the central body is not spherical. Provide a gravity model file, the maximum degree `nmax`, and the planet-centered planet-fixed (PCPF) frame name.

```julia
naif_ids = ["301", "399", "10"]
GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
DU = 1e5
nmax = 4
filepath_spherical_harmonics = joinpath(
    pkgdir(HighFidelityEphemerisModel),
    "data", "luna", "gggrx_1200l_sha_20x20.tab",
)

parameters = HighFidelityEphemerisModelParameters(
    et0, DU, GMs, naif_ids, naif_frame, abcorr;
    filepath_spherical_harmonics = filepath_spherical_harmonics,
    nmax = nmax,
    frame_PCPF = "MOON_PA",
)

tspan = (0.0, 6 * 3600 / parameters.TU)
prob = ODEProblem(eom_NbodySH_SPICE!, x0, tspan, parameters)
sol = solve(prob, Vern7(), reltol=1e-12, abstol=1e-12)
sol.u[end]
```

!!! note

    Spherical harmonics require `eom_NbodySH_*` functions. The `Nbody` variants do not evaluate harmonic terms.


## Solar radiation pressure

Enable the cannonball SRP model with `include_srp = true` and spacecraft properties `srp_Cr`, `srp_Am`. The Sun (NAIF ID `"10"`) **must** appear in `naif_ids` so its ephemeris is available.

```julia
naif_ids = ["301", "399", "10"]
GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
DU = 1e5

parameters = HighFidelityEphemerisModelParameters(
    et0, DU, GMs, naif_ids, naif_frame, abcorr;
    include_srp = true,
    srp_Cr = 1.15,
    srp_Am = 0.002,
    srp_P0 = 4.56e-6,
)

tspan = (0.0, 6 * 3600 / parameters.TU)
prob = ODEProblem(eom_Nbody_SPICE!, x0, tspan, parameters)
sol = solve(prob, Vern7(), reltol=1e-12, abstol=1e-12)
sol.u[end]
```

!!! warning

    If `include_srp = true` but `"10"` is missing from `naif_ids`, parameter construction throws an error.


## Atmospheric drag

Drag uses a quadratic law with density supplied by a callback `f_density(et, r_km)` returning $\rho$ in kg/m³. The atmosphere is modeled as co-rotating with angular rate `omega_atm` (default: Earth sidereal rate about the $z$-axis).

A built-in Harris–Priester table is available via `harris_priester_f_density(R_earth_km)`. The argument `R_earth_km` is the **Earth reference radius in km** used to form altitude as `norm(r_km) - R_earth_km`; it is not the canonical distance unit `DU` (those may differ if you choose a different `DU` for scaling). Drag is supported on all `eom_Nbody_*` and `eom_NbodySH_*` variants.

```julia
naif_ids = ["399", "10"]   # Earth-centered; Sun for optional third-body / SRP
GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
DU = 6378.0
R_earth_km = 6378.0        # Earth radius in km (for Harris–Priester altitude; not DU)

f_density = harris_priester_f_density(R_earth_km; use_min=true)

parameters = HighFidelityEphemerisModelParameters(
    et0, DU, GMs, naif_ids, naif_frame, abcorr;
    include_drag = true,
    drag_Cd = 2.2,
    drag_Am = 0.01,
    f_density = f_density,
)

x0_earth = [1.05, 0.0, 0.01, 0.0, 1.0, 0.0]
tspan = (0.0, 2 * 86400 / parameters.TU)
prob = ODEProblem(eom_Nbody_SPICE!, x0_earth, tspan, parameters)
sol = solve(prob, Vern7(), reltol=1e-12, abstol=1e-12)
sol.u[end]
```

Custom density models plug in the same way:

```julia
f_density = (et, r_km) -> 1e-12   # constant density, kg/m^3
```

!!! note

    `include_drag = true` requires `f_density` to be provided. Coefficient `k_drag` is computed internally from `drag_Cd` and `drag_Am` in SI units.


## Combining perturbations

Flags compose in a single `HighFidelityEphemerisModelParameters` instance. The example below uses interpolated ephemerides (compatible with `EnsembleThreads` and `ForwardDiff`) and enables spherical harmonics, SRP, and drag together.

```julia
naif_ids = ["399", "301", "10"]     # Earth-centered inertial frame
GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
DU = 1e3
nmax = 4
filepath_spherical_harmonics = joinpath(
    pkgdir(HighFidelityEphemerisModel),
    "data", "luna", "gggrx_1200l_sha_20x20.tab",
)

etf = et0 + 7 * 86400.0
interpolate_ephem_span = [et0, etf]
interpolation_time_step = 1000.0

parameters = HighFidelityEphemerisModelParameters(
    et0, DU, GMs, naif_ids, naif_frame, abcorr;
    interpolate_ephem_span = interpolate_ephem_span,
    interpolation_time_step = interpolation_time_step,
    filepath_spherical_harmonics = filepath_spherical_harmonics,
    nmax = nmax,
    frame_PCPF = "MOON_PA",
    include_srp = true,
    srp_Cr = 1.15,
    srp_Am = 0.002,
    include_drag = true,
    drag_Cd = 2.2,
    drag_Am = 0.01,
    f_density = harris_priester_f_density(6378.0; use_min=true),
)

tspan = (0.0, 6 * 3600 / parameters.TU)
prob = ODEProblem(eom_NbodySH_Interp!, x0, tspan, parameters)
sol = solve(prob, Vern8(), reltol=1e-12, abstol=1e-12)
sol.u[end]
```

!!! warning

    For Earth-orbit drag with Harris–Priester, use an Earth-centered `naif_ids` (e.g. `["399", "10"]`) and pass the physical Earth radius in km to `harris_priester_f_density`. The combined example above uses a small constant density as a placeholder for lunar-centric demonstration only.


## Choosing an equation of motion

| Model | Perturbations included | SPICE at runtime | `EnsembleThreads` / AD-friendly |
|-------|------------------------|------------------|----------------------------------|
| `Nbody` | third-body, optional SRP & drag | `_SPICE` yes | `_Interp` yes |
| `NbodySH` | above + spherical harmonics | `_SPICE` yes | `_Interp` yes |

See the full function list and STM options in the [Overview](@ref "Overview" overview.md).

| Use case | Recommended EOM |
|----------|-----------------|
| High accuracy, few propagations | `eom_NbodySH_SPICE!` |
| Many trajectories / sensitivities | `eom_NbodySH_Interp!` |
| No harmonics needed | `eom_Nbody_SPICE!` or `eom_Nbody_Interp!` |

For IVP setup and STM propagation, see [Basics](@ref "Basics" tutorials/basics.md). For Jacobians and Hessians, see [Jacobians & Hessians](@ref "Jacobians & Hessians" tutorials/jacobians_hessians.md).
