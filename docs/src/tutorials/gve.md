# Gauss variational equations

HFEM provides Gauss variational equations (GVE) for modified equinoctial
elements (MEE), classical Keplerian elements, and ordinary equinoctial
elements. Cartesian propagation remains the general/default formulation. Use
an element formulation when its coordinates are useful and the trajectory
remains inside its domain.

The GVE interfaces dispatch on `SpiceParameters`, `InterpParameters`, and
`EphemeridesParameters` in the same way as `eom_Nbody!` and `eom_NbodySH!`.
Both mutating and nonmutating forms are available; see the [GVE API](@ref
gve-api).

## Force-model architecture

The production GVE implementation reuses the existing inertial Cartesian HFEM
force models:

1. reconstruct Cartesian position and velocity from the element state;
2. evaluate `eom_Nbody` or `eom_NbodySH` using the supplied parameters;
3. remove the central two-body acceleration;
4. project the total remaining perturbation once into RTN;
5. apply the selected Gauss variational equations.

This preserves the Cartesian model's central-body-relative geometry,
third-body indirect terms, spherical harmonics, optional solar-radiation
pressure and drag, frames, units, and canonical scaling. Third-body forces are
not evaluated directly in RTN.

The central point-mass contribution is included analytically in the element
drift. The RTN acceleration supplied to the GVE contains only the summed
non-Keplerian perturbation.

## State conventions

All angles are radians. State values, time, acceleration, and gravitational
parameters must use the units of the selected parameter object.

### Modified equinoctial elements

The prograde MEE state is

```math
\boldsymbol{x}_{\mathrm{MEE}}=[p,f,g,h,k,L],
```

where

```math
\begin{aligned}
p &= a(1-e^2), & f &= e\cos(\Omega+\omega),
& g &= e\sin(\Omega+\omega),\\
h &= \tan(i/2)\cos\Omega, & k &= \tan(i/2)\sin\Omega,
& L &= \Omega+\omega+\nu.
\end{aligned}
```

This is the Walker modified-equinoctial convention used by
AstrodynamicsCore.jl.

### Classical Keplerian elements

Classical Keplerian states use AstrodynamicsCore's ordering

```math
[a,e,i,\Omega,\omega,\nu].
```

The sixth component is true anomaly, not mean anomaly.

### Ordinary equinoctial elements

Ordinary equinoctial states use

```math
[a,f,g,h,k,\lambda],
```

with

```math
\lambda=\Omega+\omega+M.
```

Here `a` is semimajor axis and `lambda` is instantaneous mean longitude. This
elliptic convention is distinct from Walker MEE, which use semi-latus rectum
`p` and true longitude `L`.

## Choosing a formulation

| Formulation | Supported domain | Guidance |
|---|---|---|
| Cartesian | General states; no orbital-element coordinate singularities | Use for the broadest robustness and generality. The point-mass force model remains singular at collision or body-coincidence configurations. |
| MEE | Elliptic, parabolic, and hyperbolic osculating states; supports crossing `e = 1` | Preferred GVE for broad eccentricity coverage. The standard prograde chart is singular exactly at `i = pi`, an uncommon retrograde-planar configuration. |
| Classical Keplerian GVE | Noncircular, nonequatorial states away from `e = 1` | Useful for completeness, reference, validation, and applications known to remain safely inside its domain. It becomes ill-conditioned at `e = 1` and cannot cross that boundary smoothly. |
| Ordinary equinoctial GVE | Elliptic states with `a > 0` and `e < 1`; supports circular and prograde-equatorial states | Useful for bound elliptic problems. It cannot cross `e = 1`, and the standard prograde inclination coordinates are singular exactly at `i = pi`. |
| Keplerian two-body | Classical element state under central point-mass drift | Useful for reference, validation, simplified analysis, and practical propagation when one central body dominates. |

Ordinary equinoctial propagation is well suited to bound debris, satellites,
asteroids, planetary trajectories, and similar elliptic applications. An
optimization iterate can nevertheless become hyperbolic even when the desired
final solution is elliptic. The `e < 1` representation can therefore be
restrictive in optimization unless the iterate domain is controlled.

## Constructing parameters

The GVE functions use the same parameter constructors as Cartesian HFEM. For
example, after furnishing the required SPICE kernels:

```julia
using AstrodynamicsCore
using HighFidelityEphemerisModel
using OrdinaryDiffEq
using SPICE

naif_ids = ["301", "399", "10"]
GMs = [bodvrd(id, "GM", 1)[1] for id in naif_ids]
et0 = str2et("2020-01-01T00:00:00")
DU = 3000.0

params = SpiceParameters(et0, DU, GMs, naif_ids, "J2000", "NONE")
```

`params.mus[1]` is the central-body gravitational parameter in the parameter
object's working units. `InterpParameters` and `EphemeridesParameters` may be
used in place of `SpiceParameters` without changing the generic GVE function
names.

## N-body propagation examples

The following elliptic, inclined initial condition is valid for all four state
representations:

```julia
mu = params.mus[1]
kep0 = [1.5, 0.1, 0.4, 0.2, 0.3, 0.6]
rv0 = kep2rv(kep0, mu)
mee0 = rv2mee(rv0, mu)
eq0 = rv2eq(rv0, mu)
tspan = (0.0, 2.0)
```

Cartesian propagation:

```julia
prob_rv = ODEProblem(eom_Nbody!, rv0, tspan, params)
sol_rv = solve(prob_rv, Vern9(); reltol = 1e-12, abstol = 1e-12)
```

MEE propagation and Cartesian recovery:

```julia
prob_mee = ODEProblem(gve_mee_Nbody!, mee0, tspan, params)
sol_mee = solve(prob_mee, Vern9(); reltol = 1e-12, abstol = 1e-12)
rvf_mee = mee2rv(sol_mee.u[end], mu)
```

Classical Keplerian GVE propagation:

```julia
prob_kep = ODEProblem(gve_kep_Nbody!, kep0, tspan, params)
sol_kep = solve(prob_kep, Vern9(); reltol = 1e-12, abstol = 1e-12)
rvf_kep = kep2rv(sol_kep.u[end], mu)
```

Ordinary-equinoctial GVE propagation:

```julia
prob_eq = ODEProblem(gve_eq_Nbody!, eq0, tspan, params)
sol_eq = solve(prob_eq, Vern9(); reltol = 1e-12, abstol = 1e-12)
rvf_eq = eq2rv(sol_eq.u[end], mu)
```

The same scalar `abstol` does not impose the same physical error weighting on
Cartesian and element components. For sensitive comparisons, use tolerances
appropriate to each state representation and compare Cartesian states at
explicitly shared output epochs.

## N-body plus spherical harmonics

Construct parameters with the same spherical-harmonic options used by the
Cartesian model:

```julia
filepath_SH = joinpath(
    pkgdir(HighFidelityEphemerisModel),
    "data", "luna", "gggrx_1200l_sha_20x20.tab",
)

params_SH = SpiceParameters(
    et0, DU, GMs, naif_ids, "J2000", "NONE";
    filepath_spherical_harmonics = filepath_SH,
    nmax = 8,
    frame_PCPF = "MOON_PA",
)

prob_mee_SH = ODEProblem(gve_mee_NbodySH!, mee0, tspan, params_SH)
sol_mee_SH = solve(prob_mee_SH, Vern9(); reltol = 1e-12, abstol = 1e-12)
```

Use `gve_kep_NbodySH!` or `gve_eq_NbodySH!` for the corresponding element
states. The nonmutating forms omit the trailing `!`, for example
`gve_mee_NbodySH(mee, params, t)`.

## Two-body Keplerian reference

`eom_kep_twobody!` propagates only the two-body true-anomaly drift in the
classical state. The ODE parameter is the central gravitational parameter:

```julia
prob_twobody = ODEProblem(eom_kep_twobody!, kep0, tspan, mu)
sol_twobody = solve(prob_twobody, Vern9(); reltol = 1e-12, abstol = 1e-12)
```

This model is useful for reference, validation, simplified analysis, and
practical propagation in regimes dominated by one central body. This can
include some low-thrust interplanetary cruise phases where two-body dynamics
are a close approximation. It omits third bodies, spherical harmonics,
solar-radiation pressure, drag, and all other perturbations. Use N-body gravity
when multiple celestial bodies have meaningful influence, and spherical
harmonics when operating close enough to a nonspherical body that point-mass
gravity is inadequate.

## RTN convention and helpers

RTN component order is `[radial, transverse, normal]`. The radial axis points
along position, the normal axis along `cross(r,v)`, and the transverse axis is
`normal x radial`.

```julia
T_inr2rtn = pxform_inr2rtn(rv0[1:3], rv0[4:6])
T_rtn2inr = pxform_rtn2inr(rv0[1:3], rv0[4:6])

a_inr = [0.01, -0.02, 0.03]
a_rtn = zeros(3)
project_inr_to_rtn!(a_rtn, a_inr, rv0[1:3], rv0[4:6])
```

The derivative helpers `gve_mee_derivs!`, `gve_kep_derivs!`, and
`gve_eq_derivs!` accept an element state, a non-Keplerian RTN acceleration, and
the central gravitational parameter. Most users should call the N-body or
N-body+SH interfaces instead.

## Backend and threading notes

| Parameters | GVE support | Threading status |
|---|---|---|
| `SpiceParameters` | N-body and N-body+SH | Supported for serial use. Concurrent SPICE/CSPICE evaluation was not tested and must not be assumed thread-safe. |
| `EphemeridesParameters` | N-body and N-body+SH | Serial/threaded RHS and `EnsembleThreads` comparisons passed for the tested GVE formulations. |
| `InterpParameters` | N-body and N-body+SH | Serial/threaded RHS and `EnsembleThreads` comparisons passed for the tested GVE formulations; retained as a legacy compatibility backend. |

Each independent threaded trajectory must own its state and output arrays. One
ODE solve does not automatically use all Julia threads; `EnsembleThreads` is
for independent trajectories. For parallel SPICE propagation, prefer separate
Julia processes with serial kernel loading unless application-level
synchronization is provided.

## Benchmark

From the package root, run the finalized benchmark with:

```sh
julia --project=. benchmark/gve_vs_cartesian.jl
```

The root environment and script load/parse were verified together. The script
warms each case, then measures Cartesian and GVE RHS time, allocations, memory,
RTN projection, integrated propagation time, and Cartesian trajectory error
for N-body and N-body+SH cases. It writes
`benchmark/reports/gve_vs_cartesian.md` unless `GVE_BENCHMARK_OUTPUT` is set.

Benchmark results depend on the trajectory, backend, solver tolerances,
hardware, Julia version, and compilation state. A timing from one formulation
or case should not be treated as a universal performance ranking.