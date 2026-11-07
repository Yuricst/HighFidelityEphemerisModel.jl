---
title: '`HighFidelityEphemerisModel.jl`: A Julia package for high-fidelity astrodynamics'
tags:
  - Julia
  - astrodynamics
  - orbital mechanics
  - spacecraft dynamics
  - ephemerides
  - orbit propagation
authors:
  - name: Yuri Shimane
    orcid: 0000-0001-7728-3780
    corresponding: true
    affiliation: 1
  - name: Derek Phung
    orcid: 0009-0002-3063-5595
    affiliation: 1
affiliations:
  - name: Department of Mechanical and Aerospace Engineering, University of California, Irvine, United States
    index: 1
    ror: 04gyf1771
date: 30 October 2026
bibliography: paper.bib
---
# Summary

`HighFidelityEphemerisModel.jl` is an open-source Julia [@bezanson2017julia] package for modeling high-fidelity translational dynamics of spacecraft. The package provides equations of motion that account for the gravitational influence of multiple celestial bodies using ephemeris data from the Jet Propulsion Laboratory's SPICE ecosystem [@acton1996spice], together with configurable perturbations from a spherical harmonics model of gravitational irregularities, solar radiation pressure, and atmospheric drag. It is designed to be fully compatible with Julia's `OrdinaryDiffEq.jl` ecosystem [@rackauckas2017differentialequations].

The package supports trajectory propagation in Cartesian coordinates and Gauss variational equations formulated in modified equinoctial, classical Keplerian, and ordinary equinoctial elements. Additional capabilities include state-transition matrix propagation, Jacobian and Hessian evaluation, event-condition utilities, and export of propagated trajectories to SPICE spacecraft kernels. Ephemeris and reference-frame evaluations are supported through the SPICE  kernels ecosystem. These capabilities enable researchers to incorporate high-fidelity spacecraft dynamics into trajectory analysis, sensitivity analysis, and optimization workflows, while also leveraging Julia's scientific computing libraries.

# Statement of need

Accurate spacecraft trajectory propagation is essential for mission design, trajectory optimization, and dynamical analysis. Simplified models often provide valuable insight into spacecraft motion, but are insufficient in designing flyable nominal trajectories or in assessing realistic performance of guidance, navigation, and control (GNC) algorithms. While mature numerical integration libraries provide suitable algorithms for solving general initial value problems (IVP), these must be paired with efficient implementation of these high-fidelity dynamics models, often involving not only the equations of motion but also the variational equations that yield sensitivity information, commonly used by  workflows in Astrodynamics involving numerical optimization, filtering, and sensitivity analysis. This software addresses this need by providing configurable, ephemeris-based spacecraft dynamics within Julia's scientific computing ecosystem. By separating the equations of motion from the numerical integration algorithms, the package enables researchers and mission analysis engineers to combine high-fidelity dynamical models with existing differential-equation solvers and optimization tools.
Figure \\ref{fig:hfem-hero} demonstrates some of the capabilities of `HighFidelityEphemerisModel.jl`. 

![Demonstration of `HighFidelityEphemerisModel.jl` capabilities.\label{fig:hfem-hero}](figures/hero_figure.png){width="100%"}

# State of the field

Several software packages provide high-fidelity spacecraft dynamics and trajectory propagation capabilities in languages other than Julia. Orekit [@orekit] is a Java-based flight dynamics library offering comprehensive orbital propagation, reference-frame transformations, and mission analysis functionality. NASA's General Mission Analysis Tool (GMAT) [@gmat] provides an integrated environment for trajectory design, optimization, and navigation through graphical, scripting, and programmatic interfaces. Basilisk [@Kenneally2020-zn] is a modular simulation framework that combines spacecraft dynamics with guidance, navigation, and control models, exposed as a python package. Nyx [@nyxspace] provides a Rust-based astrodynamics toolkit for trajectory propagation and orbit determination, while Brahe [@eddy2026brahe] offers a composable astrodynamics library with Rust and Python interfaces, particularly for satellite dynamics and mission operations.

While these tools offer extensive capabilities, `HighFidelityEphemerisModel.jl` addresses a more focused requirement: providing configurable, ephemeris-based equations of motion that can be directly incorporated into Julia's scientific computing workflows. 
Rather than providing a comprehensive mission analysis toolkit or implementing separate numerical integration algorithms, `HighFidelityEphemerisModel.jl` exposes reusable dynamical models compatible with `OrdinaryDiffEq.jl`. This enables researchers to combine efficient, high-fidelity force models with numerical integration, sensitivity analysis, and trajectory optimization methods within the same programming environment.

Within the Julia ecosystem, there are a number of mature spacecraft-related packages, albeit focused on different scopes. `SatelliteToolbox.jl` offers satellite-analysis tools, including analytical orbit propagators and terrestrial reference-frame utilities [@satellitetoolbox]. `GeneralAstrodynamics.jl` provides common astrodynamics calculations and solvers for Keplerian, Lambert, and halo-orbit problems [@generalastrodynamics]. `AstroModels.jl` develops reusable gravitational and other astrodynamical force models [@astromodels]. 
`HighFidelityEphemerisModel.jl` complements these projects by packaging ephemeris-based spacecraft equations of motion, multiple ephemeris backends, and Cartesian and Gauss variational formulations for direct use with Julia differential-equation solvers.

# Software design

## Dynamics models

`HighFidelityEphemerisModel.jl` formulates spacecraft motion relative to a central celestial body, such as a planet, a moon, or the Sun, with additional, configurable perturbations due to gravitational irregularities of the central body modeled via spherical harmonics, third-body point-mass perturbations by other celestial bodies, solar radiation pressure, and atmospheric drag.
The package supports propagation of the spacecraft state in terms of its inertial position and velocity vectors, as well as in terms of orbital elements throug Gauss variational equations. 
Their implementation support rescaling the dynamics via canonical scales for improved numerical stability, selected by a user defined distance unit $\mathrm{DU}$, and an automatically scaled velocity unit $\mathrm{VU} = \sqrt{\mu / \mathrm{DU}}$, where $\mu$ is the central body's gravitational parameter.

## Modular implementation

A central architectural decision is the separation of dynamical models from ephemeris evaluation and numerical integration. `HighFidelityEphemerisModel.jl` provides two parameter backends: `SpiceParameters`, which queries NAIF SPICE directly [@acton1996spice]; and `EphemeridesParameters`, which uses the Julia-native `Ephemerides.jl` and `FrameTransformations.jl` libraries [@jsmdEphemerides; @jsmdFrameTransformations].
Julia's multiple dispatch allows the same high-level equations-of-motion functions, `eom_Nbody!` and `eom_NbodySH!`, to operate with these different parameter types. Direct SPICE evaluation provides an established reference implementation, while the Julia-native backend supports automatic differentiation and tested multithreaded propagation workflows without relying on direct CSPICE calls during each force evaluation.

Numerical integration is delegated to `OrdinaryDiffEq.jl` [@rackauckas2017differentialequations]. This self-contained example propagates a Moon-centered state with Earth and Sun point-mass perturbations. 
<!-- Run it from the `HighFidelityEphemerisModel.jl` repository using `julia --project=.` after installing its declared dependencies. Download the required NAIF kernels beforehand and set the `SPICE` environment variable to their root directory, with `lsk/`, `spk/`, and `pck/` subdirectories. -->

The package also provides automatic-differentiation-based Jacobian and Hessian calculations through `ForwardDiff.jl` [@revels2016forwarddiff], together with state-transition matrix propagation for sensitivity analysis. Propagated trajectories can be exported to SPICE type-13 ephemeris kernels, with utilities supporting multiple trajectory segments and maneuver metadata. Together, these capabilities facilitate the integration of high-fidelity spacecraft dynamics into trajectory optimization, GNC algorithm designs, and reproducible mission-analysis workflows.


Figure \\ref{fig:hfem-architecture} summarizes these interfaces.

![`HighFidelityEphemerisModel.jl` architecture. Cartesian forces support direct propagation and Gauss variational equations (GVE). Jacobians drive Cartesian state-transition matrix (STM) evolution; Hessians are separate. Element solutions require Cartesian conversion before SPICE export. \label{fig:hfem-architecture}](figures/hfem_architecture.png){width="100%"}


## Validation

The package has been evaluated through numerical comparisons with established astrodynamics libraries, including GMAT, Orekit, Nyx, and Basilisk. 
Three representative cases encompass a seven-day near-rectilinear halo orbit (H7), a low lunar orbit (LLO-MESH4), and a low Earth orbit (LEO-EMSH4). Each case includes third-body gravitational perturbations and spherical harmonics terms up to degree/order-four.
Residuals use documented `HighFidelityEphemerisModel.jl` SPICE trajectories, with tool-specific numerical mirrors for Nyx and LLO Orekit; they measure agreement, not absolute accuracy. These comparisons provide quantitative evidence of numerical consistency across different propagation frameworks while accounting for differences in integrators, ephemeris evaluation, and model implementation.

Table \\ref{tbl:validation-cases}, Figure \\ref{fig:validation-position}, and Table \\ref{tbl:position-maxima} summarize these comparisons. GMAT H7 uses shared MOON_PA orientation with an inertial-frame correction. GMAT LLO/LEO endpoint offsets from nominal grids reach 2.70 microseconds. Basilisk uses 1 s communication/task updates with adaptive integration; Orekit uses shared frame orientations.


| Case      | Central body | Third bodies | Spherical harmonics frame  | Integration time |
| --------- | ------------ | ------------ | ------------ | ------- |
| H7        | Moon         | Earth + Sun  | `MOON\_PA`   | 7 d     |
| LLO-MESH4 | Moon         | Earth + Sun  | `MOON\_PA`   | 2.064 h |
| LEO-EMSH4 | Earth        | Moon + Sun   | `IAU\_EARTH` | 1.978 h |

: Overview of validation cases; LLO/LEO each span one orbital period. \\label{tbl:validation-cases}


| Implementation   | H7 \[mm\]  | LLO-MESH4 \[mm\] | LEO-EMSH4 \[mm\] |
| ---------------- | ---------- | ---------------- | ---------------- |
| `EphemeridesParameters` | 0.337568   | 0.0349237        | 0.148159         |
| GMAT             | 136.282    | 3.07436          | 32.1019          |
| Orekit           | 5.31215    | 1.51733e-05      | 0.000104418      |
| Nyx              | 0.00121614 | 0.221886         | 1.02998          |
| Basilisk         | 0.347638   | 0.0262097        | 101.602          |

: Maximum sampled position residuals against documented `HighFidelityEphemerisModel.jl` with `SpiceParameters` or numerical mirrors, not absolute trajectory errors. First row uses `HighFidelityEphemerisModel.jl` with `EphemeridesParameters`.\\label{tbl:position-maxima}

![Position residuals against `HighFidelityEphemerisModel.jl` SPICE trajectories or tool-specific numerical mirrors, not absolute trajectory accuracy. Initialization ($t=0$) is omitted; later stored zeros appear as logarithmic gaps without flooring. Output sampling differs from integration steps. \label{fig:validation-position}](figures/validation_position_errors.png){width="100%"}

Additional benchmarks evaluate `HighFidelityEphemerisModel.jl`'s alternative ephemeris backends. The Julia-native `Ephemerides.jl` backend reduced propagation time relative to direct SPICE evaluation from 0.2473 to 0.1236 s for a point-mass near-rectilinear halo orbit case, and from 0.8677 to 0.7031 s with lunar spherical harmonics. These timings are configuration-specific, not universal speedups.

Table \\ref{tbl:warm-performance} reports end-to-end propagation at `HighFidelityEphemerisModel.jl` source commit `2aca3a70ffffad4e61db28f68c8c9d6d7630f1e9` (Julia 1.10.11). For each case/backend, an initial untimed solve excludes the initial Julia compilation run; a separate allocation-measurement solve also precedes the ten timed repetitions. No compilation time is subtracted from measured runs. Timings include ODE construction, Vern8 propagation and in-memory output with initialized backends; backend setup and serialization are excluded.


| Case | SPICE \[s\] | Ephemerides \[s\] | Time ratio |
| ---- | ----------- | ----------------- | ---------- |
| P3   | 0.2472955   | 0.1236207         | 0.4999     |
| H7   | 0.8677467   | 0.7031303         | 0.8103     |


: Median propagation (ten repetitions, one Julia thread). Ratio: Ephemerides/SPICE. P3 uses point masses; H7 adds lunar 4x4. \\label{tbl:warm-performance}

The package is distributed through Julia's package registry and provides documented examples, automated tests, and continuous integration to support reproducible research and further software development.


# Example

The following code provides a short example solving an IVP for an initial state in a Moon-centered J2000 inertial frame over the course of 6 hours.
The dynamics model accounts for third-body perturbations of the Earth and the Sun.

```julia
using HighFidelityEphemerisModel, OrdinaryDiffEq, SPICE

furnsh(joinpath(ENV["SPICE"], "lsk", "naif0012.tls"))
furnsh(joinpath(ENV["SPICE"], "spk", "de440.bsp"))
furnsh(joinpath(ENV["SPICE"], "pck", "gm_de440.tpc"))

ids = ["301", "399", "10"]                    # Moon first, then Earth and Sun
GMs = [bodvrd(id, "GM", 1)[1] for id in ids]  # gravitational parameters, km^3/s^2
et0 = str2et("2030-01-05T00:00:00")           # initial epoch, UTC
DU = 1e5                                      # canonical scale, km
params = SpiceParameters(et0, DU, GMs, ids, "J2000", "NONE")

x0 = [1.05, 0.0, 0.3, 0.5, 1.0, 0.0]          # canonical units DU, DU/TU
tspan = (0.0, 6 * 3600 / params.TU)           # canonical unit TU
prob = ODEProblem(eom_Nbody!, x0, tspan, params)
sol = solve(prob, Vern8(); reltol=1e-12, abstol=1e-12)
@assert sol.t[end] == tspan[2] && all(u -> all(isfinite, u), sol.u)
kclear()
```

# Research impact

`HighFidelityEphemerisModel.jl` has supported several research on high-fidelity trajectory design and GNC in cisluar space [@shimane2026optimizationbasedformationflightlibration; @Shimane2026-qj; @phung2026verticalorbits]. This application demonstrates the integration of ephemeris-based spacecraft dynamics, sensitivity analysis, and trajectory propagation into larger numerical optimization and mission-analysis workflows.

# AI usage disclosure

AI was used to help develop the code and to draft and edit parts of this manuscript.
All text produced with AI assistance was reviewed, revised, and verified by the authors before
inclusion in the code or paper.

# Acknowledgements

The authors acknowledge the developers and maintainers of the Julia scientific computing ecosystem and NASA's Navigation and Ancillary Information Facility (NAIF), whose software and documentation provide important foundations for this work.

# References