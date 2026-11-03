---
title: 'HighFidelityEphemerisModel.jl: A Julia package for high-fidelity astrodynamics'
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

HighFidelityEphemerisModel.jl is an open-source Julia [@bezanson2017julia] package for modeling spacecraft motion in high-fidelity, time-dependent dynamical environments. The package provides equations of motion that account for the gravitational influence of multiple celestial bodies using ephemeris data, together with configurable perturbations from spherical harmonics, solar radiation pressure, and atmospheric drag. Rather than implementing numerical integration algorithms, the package provides dynamical models compatible with Julia's OrdinaryDiffEq.jl ecosystem [@rackauckas2017differentialequations].

The package supports trajectory propagation in Cartesian coordinates and Gauss variational equations formulated in modified equinoctial, classical Keplerian, and ordinary equinoctial elements. Additional capabilities include state-transition matrix propagation, Jacobian and Hessian evaluation, event-condition utilities, and export of propagated trajectories to SPICE kernels. Ephemeris and reference-frame evaluations are supported through NASA's SPICE toolkit [@acton1996spice] and Julia-native alternatives. These capabilities enable researchers to incorporate high-fidelity spacecraft dynamics into trajectory analysis, sensitivity analysis, and optimization workflows, including applications in cislunar space.

# Statement of need

Accurate spacecraft trajectory propagation is essential for mission design, trajectory optimization, and dynamical analysis. Simplified models, such as the two-body problem and circular restricted three-body problem, provide valuable insight into spacecraft motion but may be insufficient for applications requiring higher physical fidelity. In multi-body environments, particularly cislunar space, trajectory propagation may require time-dependent planetary and lunar ephemerides, nonspherical gravitational fields, and non-gravitational perturbations such as solar radiation pressure. Incorporating these effects requires consistent treatment of celestial-body positions, reference-frame transformations, and force models.

Developing and validating these dynamical models independently can require substantial effort, especially when they must be repeatedly evaluated within numerical optimization and sensitivity-analysis workflows. While numerical integration libraries provide algorithms for solving differential equations, astrodynamics applications additionally require specialized equations of motion, perturbation models, and derivative information. Providing these components through reusable software can reduce duplicated implementation effort and facilitate reproducible numerical experiments.

HighFidelityEphemerisModel.jl addresses this need by providing configurable, ephemeris-based spacecraft dynamics within Julia's scientific computing ecosystem. By separating the equations of motion from the numerical integration algorithms, the package enables researchers to combine high-fidelity dynamical models with existing differential-equation solvers and optimization tools. It is intended for researchers and engineers developing spacecraft trajectory analysis, design, and optimization methods, particularly those requiring direct programmatic access to the underlying dynamics.

# State of the field

Several open-source software packages provide high-fidelity spacecraft dynamics and trajectory propagation capabilities. Orekit [@orekit] is a Java-based flight dynamics library offering comprehensive orbital propagation, reference-frame transformations, and mission analysis functionality. NASA's General Mission Analysis Tool (GMAT) [@gmat] provides an integrated environment for trajectory design, optimization, and navigation through graphical, scripting, and programmatic interfaces. Basilisk [@Kenneally2020-zn] is a modular simulation framework that combines spacecraft dynamics with guidance, navigation, and control models. Nyx [@nyxspace] provides a Rust-based astrodynamics toolkit for trajectory propagation and orbit determination, while Brahe [@eddy2026brahe] offers a composable astrodynamics library with Rust and Python interfaces, particularly for satellite dynamics and mission operations.

Although these tools offer extensive capabilities, HighFidelityEphemerisModel.jl addresses a more focused requirement: providing configurable, ephemeris-based equations of motion that can be directly incorporated into Julia's scientific computing workflows. Rather than providing a comprehensive mission analysis environment or implementing separate numerical integration algorithms, HFEM exposes reusable dynamical models compatible with OrdinaryDiffEq.jl. This enables researchers to combine high-fidelity force models with numerical integration, sensitivity analysis, and trajectory optimization methods within the same programming environment.

Within the Julia ecosystem, `SatelliteToolbox.jl` offers satellite-analysis tools, including analytical orbit propagators and terrestrial reference-frame utilities [@satellitetoolbox]. `GeneralAstrodynamics.jl` provides common astrodynamics calculations and solvers for Keplerian, Lambert, and halo-orbit problems [@generalastrodynamics]. `AstroModels.jl` develops reusable gravitational and other astrodynamical force models [@astromodels]. HFEM complements these projects by packaging ephemeris-based spacecraft equations of motion, multiple ephemeris backends, and Cartesian and Gauss variational formulations for direct use with Julia differential-equation solvers.

Developing HFEM as a dedicated Julia package provides a focused interface to high-fidelity spacecraft dynamics without requiring researchers to integrate larger mission-analysis environments or introduce cross-language interfaces. Its combination of ephemeris evaluation, configurable perturbations, and multiple orbital-state representations supports custom research applications while leveraging existing Julia numerical computing tools. HFEM therefore complements established astrodynamics software rather than attempting to replace its broader mission-analysis and operational capabilities.

# Software design

## Dynamical model

`HighFidelityEphemerisModel.jl` formulates spacecraft motion relative to a central body using ephemeris-based gravitational dynamics and configurable perturbation models. The Cartesian equations of motion are expressed as

$$
\dot{\boldsymbol{x}} =
\begin{bmatrix}
\boldsymbol{v} \\
-\dfrac{\mu}{\lVert\boldsymbol{r}\rVert^3}\boldsymbol{r}
+\displaystyle\sum_i \boldsymbol{a}_{\mathrm{3bd},i}
+\boldsymbol{a}_{\mathrm{SH}}
+\boldsymbol{a}_{\mathrm{SRP}}
+\boldsymbol{a}_{\mathrm{drag}}
\end{bmatrix},
$$

where $\boldsymbol{x}=[\boldsymbol{r}^{T},\boldsymbol{v}^{T}]^{T}$ is the spacecraft state, $\mu$ is the central-body gravitational parameter, and the remaining terms represent third-body gravity, spherical harmonics, solar radiation pressure, and atmospheric drag, respectively. The perturbation terms are configurable, allowing the same framework to represent point-mass N-body dynamics or more detailed gravitational and non-gravitational environments.

Third-body accelerations are evaluated using ephemeris-derived body positions and Battin's numerically stable formulation [@battin1987]. Spherical-harmonic accelerations are evaluated in the corresponding body-fixed frame and transformed into the propagation frame. The implementation also supports nondimensional state and time scaling through configurable characteristic distance and time units.

## Modular implementation

A central architectural decision is the separation of dynamical models from ephemeris evaluation and numerical integration. HFEM provides three parameter backends: `SpiceParameters`, which queries NAIF SPICE directly [@acton1996spice]; `EphemeridesParameters`, which uses the Julia-native `Ephemerides.jl` and `FrameTransformations.jl` libraries [@jsmdEphemerides; @jsmdFrameTransformations]; and `InterpParameters`, which evaluates pre-interpolated ephemeris and transformation data.

Julia's multiple dispatch allows the same high-level equations-of-motion functions, `eom_Nbody!` and `eom_NbodySH!`, to operate with these different parameter types. Direct SPICE evaluation provides an established reference implementation, while the Julia-native backend supports automatic differentiation and tested multithreaded propagation workflows without relying on direct CSPICE calls during each force evaluation. The interpolated backend is retained for compatibility, although its accuracy depends on the interpolation resolution.

Numerical integration is delegated to Julia's `OrdinaryDiffEq.jl` ecosystem [@rackauckas2017differentialequations]. This separation permits users to select integration algorithms and error tolerances independently of the dynamical model. For example, assuming the required ephemeris data have been loaded and an HFEM parameter object has been initialized, a spacecraft trajectory can be propagated using

```julia
using HighFidelityEphemerisModel
using OrdinaryDiffEq
# Assumes params is a configured HFEM parameter object
x0 = [1.05, 0.0, 0.3, 0.5, 1.0, 0.0]
tspan = (0.0, 6 * 3600 / params.TU)
prob = ODEProblem(eom_NbodySH!, x0, tspan, params)
sol = solve(prob, Vern8(); reltol=1e-12, abstol=1e-12)
```

Here, `x0` and `tspan` are expressed in the canonical units defined by the parameter object, and `eom_NbodySH!` evaluates N-body dynamics with spherical-harmonic perturbations. The numerical solver and tolerances can be changed without modifying the underlying force-model implementation.

Figure \ref{fig:hfem-architecture} summarizes these interfaces.

![HFEM architecture. Cartesian forces support direct propagation and Gauss variational equations (GVE). Jacobians drive Cartesian state-transition matrix (STM) evolution; Hessians are separate. Element solutions require Cartesian conversion before SPICE export. \label{fig:hfem-architecture}](figures/hfem_architecture.png){width="100%"}

## Orbital representations and research interfaces

In addition to Cartesian propagation, HFEM implements Gauss variational equations in modified equinoctial [@walker1985mee], classical Keplerian, and ordinary equinoctial elements.
The associated orbital-element conversions use `AstrodynamicsCore.jl` [@astrodynamicscore]. These formulations reuse the existing Cartesian force models by reconstructing the Cartesian state, evaluating the inertial accelerations, subtracting the central two-body contribution, and projecting the remaining perturbing acceleration into the radial-transverse-normal frame. This avoids maintaining separate perturbation implementations for each orbital representation.

The choice of formulation depends on the trajectory domain. Modified equinoctial elements support elliptic, parabolic, and hyperbolic trajectories, subject to their inclination singularity, while classical Keplerian elements become singular or ill-conditioned for certain circular, equatorial, and parabolic configurations. The implemented ordinary equinoctial formulation is restricted to elliptic trajectories.

The package also provides automatic-differentiation-based Jacobian and Hessian calculations through `ForwardDiff.jl` [@revels2016forwarddiff], together with state-transition matrix propagation for sensitivity analysis. Propagated trajectories can be exported to SPICE type-13 ephemeris kernels, with utilities supporting multiple trajectory segments and maneuver metadata. Together, these capabilities facilitate the integration of high-fidelity spacecraft dynamics into trajectory optimization, guidance, and reproducible mission-analysis workflows.

# Research impact statement

`HighFidelityEphemerisModel.jl` has supported research on high-fidelity trajectory design and station keeping in the Earth–Moon system [@phung2026verticalorbits]. This application demonstrates the integration of ephemeris-based spacecraft dynamics, sensitivity analysis, and trajectory propagation into larger numerical optimization and mission-analysis workflows.

The package has also been evaluated through numerical comparisons with established astrodynamics software, including GMAT, Orekit, Nyx/ANISE, and Basilisk. Three representative cases encompass a seven-day near-rectilinear halo orbit (H7), a low lunar orbit (LLO-MESH4), and a low Earth orbit (LEO-EMSH4). Each case includes third-body gravitational perturbations and degree/order-four spherical harmonics. Residuals use documented HFEM SPICE trajectories, with tool-specific numerical mirrors for Nyx/ANISE and LLO Orekit; they measure agreement, not absolute accuracy. These comparisons provide quantitative evidence of numerical consistency across different propagation frameworks while accounting for differences in integrators, ephemeris evaluation, and model implementation.

Table \ref{tbl:validation-cases}, Figure \ref{fig:validation-position}, and Table \ref{tbl:position-maxima} summarize these comparisons. GMAT H7 uses shared MOON_PA orientation with an inertial-frame correction. GMAT LLO/LEO endpoint offsets from nominal grids reach 2.70 microseconds. Basilisk uses 1 s communication/task updates with adaptive integration; Orekit uses shared frame orientations.

| Case | Central body | Third bodies | 4x4 frame | Span |
|---|---|---|---|---|
| H7 | Moon | Earth + Sun | MOON_PA | 7 d |
| LLO-MESH4 | Moon | Earth + Sun | MOON_PA | 2.064 h |
| LEO-EMSH4 | Earth | Moon + Sun | IAU_EARTH | 1.978 h |

: Validation cases; frames specify harmonic evaluation axes. LLO/LEO each span one period. \label{tbl:validation-cases}

| Implementation | H7 [mm] | LLO-MESH4 [mm] | LEO-EMSH4 [mm] |
|---|---|---|---|
| HFEM Ephemerides | 0.337568 | 0.0349237 | 0.148159 |
| GMAT | 136.282 | 3.07436 | 32.1019 |
| Orekit | 5.31215 | 1.51733e-05 | 0.000104418 |
| Nyx/ANISE | 0.00121614 | 0.221886 | 1.02998 |
| Basilisk | 0.347638 | 0.0262097 | 101.602 |

: Maximum sampled position residuals against documented HFEM SPICE references or numerical mirrors, not absolute trajectory errors. \label{tbl:position-maxima}

![Position residuals against HFEM SPICE trajectories or tool-specific numerical mirrors, not absolute trajectory accuracy. Initialization ($t=0$) is omitted; later stored zeros appear as logarithmic gaps without flooring. Output sampling differs from integration steps. \label{fig:validation-position}](figures/validation_position_errors.png){width="100%"}

Additional benchmarks evaluate the computational performance of HFEM's alternative ephemeris backends. For warm trajectory propagation, the Julia-native `Ephemerides.jl` backend reduced execution time relative to direct SPICE evaluation from 0.2473 to 0.1236 s for a point-mass near-rectilinear halo orbit case, and from 0.8677 to 0.7031 s for a case including lunar spherical harmonics. These results demonstrate the performance characteristics of the alternative backend under the tested configurations, without implying equivalent improvements for all dynamical models or computational environments.

Table \ref{tbl:warm-performance} reports warm end-to-end propagation at HFEM source commit `2aca3a70ffffad4e61db28f68c8c9d6d7630f1e9` (Julia 1.10.11). Timings include ODE construction, Vern8 propagation and in-memory output with initialized backends; setup, compilation and serialization are excluded.

| Case | SPICE [s] | Ephemerides [s] | Time ratio |
|---|---|---|---|
| P3 | 0.2472955 | 0.1236207 | 0.4999 |
| H7 | 0.8677467 | 0.7031303 | 0.8103 |

: Median warm propagation (ten repetitions, one Julia thread). Ratio: Ephemerides/SPICE. P3 uses point masses; H7 adds lunar 4x4. \label{tbl:warm-performance}

The package is distributed through Julia's package registry and provides documented examples, automated tests, and continuous integration to support reproducible research and further software development.

# AI usage disclosure

AI was used to help develop the code and to draft and edit parts of this manuscript.
All text produced with AI assistance was reviewed, revised, and verified by the authors before
inclusion in the code or paper.

# Acknowledgements

The authors acknowledge the developers and maintainers of the Julia scientific computing ecosystem and NASA's Navigation and Ancillary Information Facility (NAIF), whose software and documentation provide important foundations for this work.

# References