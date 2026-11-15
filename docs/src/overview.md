# Overview

There are a number of equations of motion implemented in `HighFidelityEphemerisModel.jl`.

- functions starting with `eom_` integrates the translational state (`[x,y,z,vx,vy,vz]`)
- functions starting with `eom_stm_` integrates both the translational state and the flattened 6-by-6 STM.

!!! note

    The STM is flattened column by column (Julia's memory order), so `reshape(x[7:42], 6, 6)` directly gives the STM, without transposing; for example,

    ```julia
    x_stm_tf = sol.u[end]                       # concatenated state & STM (flattened)
    x_tf     = x_stm_tf[1:6]                    # final state [x,y,z,vx,vy,vz]
    STM_tf   = reshape(sol.u[end][7:42],6,6)    # final 6-by-6 STM
    ```

## Propagation formulations

Cartesian propagation is the general/default HFEM formulation and has no
orbital-element coordinate singularities. HFEM also provides Gauss variational
equations (GVE) that reuse the same Cartesian force models.

Cartesian coordinates do not remove physical force-model singularities. The
point-mass model remains singular at zero central-body distance and when the
spacecraft coincides with another point-mass gravitating body.

| Formulation | State ordering | Domain and intended use |
|---|---|---|
| Cartesian | `[x, y, z, vx, vy, vz]` | Broadest robustness and generality; no orbital-element coordinate singularities. |
| Modified equinoctial (MEE) | `[p, f, g, h, k, L]` | Elliptic, parabolic, and hyperbolic states, including crossings of `e = 1`; the standard prograde chart is singular exactly at `i = pi`. |
| Classical Keplerian | `[a, e, i, Omega, omega, true_anomaly]` | Useful for completeness, reference, validation, and restricted-domain applications; singular for circular and equatorial states and ill-conditioned at `e = 1`, which it cannot cross smoothly. |
| Ordinary equinoctial | `[a, f, g, h, k, lambda]` | Elliptic-only (`a > 0`, `e < 1`); regular at circular and prograde-equatorial states but singular exactly at `i = pi`. |
| Keplerian two-body | `[a, e, i, Omega, omega, true_anomaly]` | Reference, validation, simplified analysis, and practical central-body-dominated propagation. |

Ordinary equinoctial elements are useful for bound satellites, debris,
asteroids, planets, and other elliptic-orbit applications. In optimization,
however, an intermediate iterate can become hyperbolic even when the desired
solution is elliptic; this representation is restrictive unless the iterate
domain is controlled.

The two-body model can be an appropriate approximation when one central body
dominates, including some low-thrust interplanetary cruise phases. Use N-body
gravity when multiple bodies have meaningful influence, and spherical
harmonics when point-mass gravity is inadequate near a nonspherical body.

For every GVE force model, HFEM reconstructs Cartesian position and velocity,
evaluates the perturbing accelerations of the corresponding inertial Cartesian
EOM (all terms except the central two-body acceleration, which is never added,
so that nothing has to be subtracted), projects the total perturbation once
into RTN, and applies the selected variational equations. See
[Gauss variational equations](@ref) for examples and detailed limitations.
Third-body accelerations are not evaluated directly in RTN.

## Dynamics model

In `HighFidelityEphemerisModel.jl`, the dynamcis consists of the central gravitational term, together with the following perturbations:

- third-body perturbations
- spherical harmonics
- solar radiation pressure
- atmospheric drag

```math
\dot{\boldsymbol{x}}(t) = 
\begin{bmatrix}
    \dot{\boldsymbol{r}}(t) \\ \dot{\boldsymbol{v}}(t)
\end{bmatrix} = 
\begin{bmatrix}
    \boldsymbol{v}(t)
    \\ -\dfrac{\mu}{\| \boldsymbol{r}(t) \|_2^3}\boldsymbol{r}(t)
\end{bmatrix}
+
\sum_{i} 
\begin{bmatrix}
    \boldsymbol{0}_{3 \times 1} 
    \\ \boldsymbol{a}_{\mathrm{3bd},i}(t)
\end{bmatrix}
+ 
\begin{bmatrix}
    \boldsymbol{0}_{3 \times 1}
    \\ \boldsymbol{a}_{\mathrm{SH},n_{\max}}(t)
\end{bmatrix}
+ 
\begin{bmatrix}
    \boldsymbol{0}_{3 \times 1}
    \\ \boldsymbol{a}_{\mathrm{SRP}}(t)
\end{bmatrix}
+
\begin{bmatrix}
    \boldsymbol{0}_{3 \times 1}
    \\ \boldsymbol{a}_{\mathrm{drag}}(t)
\end{bmatrix}
```

### Third-body perturbation

The third-body perturbation due to body $i$, $\boldsymbol{a}_{\mathrm{3bd},i}$, is given by

```math
\boldsymbol{a}_{\mathrm{3bd},i}(t)
= -\mu_i \left(
    \dfrac{\boldsymbol{r}(t) - \boldsymbol{r}_i(t)}{\| \boldsymbol{r}(t) - \boldsymbol{r}_i(t) \|_2^3}
    +
    \dfrac{\boldsymbol{r}_i(t)}{\| \boldsymbol{r}_i(t) \|_2^3}
\right)
```

where $\boldsymbol{r}_i$ is the position vector of the perturbing body.
In `HighFidelityEphemerisModel.jl`, this term is implemented using Battin's $F(q)$ function:

```math
\boldsymbol{a}_{\mathrm{3bd},i}(t) =
-\dfrac{\mu_i}{\| \boldsymbol{r}(t) - \boldsymbol{r}_i(t) \|_2^3} (\boldsymbol{r}(t) + F(q_i)\boldsymbol{r}_i(t))
```

where $F(q_i)$ is given by

```math
F(q_i) = q_i \left( \dfrac{3 + 3q_i + q_i^2}{1 + (\sqrt{1 + q_i})^3} \right)
,\quad
q_i = \dfrac{\boldsymbol{r}(t)^T (\boldsymbol{r}(t) - 2\boldsymbol{r}_i(t))}{\boldsymbol{r}_i(t)^T \boldsymbol{r}_i(t)}
```

### Spherical Harmonics

The spherical harmonics perturbation $\boldsymbol{a}_{\mathrm{SH},n_{\max}}$ is given by

```math
\boldsymbol{a}_{\mathrm{SH},n_{\max}} = 
\sum_{n=2}^{n_{\max}} \sum_{m=0}^n \boldsymbol{a}_{\mathrm{SH},nm}
```

where $\boldsymbol{a}_{\mathrm{SH},nm}$ is given by

```math
\boldsymbol{a}_{\mathrm{SH},nm} = 
\begin{bmatrix}
    \ddot{x}_{n m} \\ \ddot{y}_{n m} \\ \ddot{z}_{n m}
\end{bmatrix}
```

where

```math
\begin{aligned}
& \ddot{x}_{n m} =
\begin{cases}
    \frac{G M}{R_{\oplus}^2} \cdot\left\{-C_{n 0} V_{n+1,1}\right\} & m = 0 \\[1.0em]
    \frac{G M}{R_{\oplus}^2} \cdot \frac{1}{2} \cdot\left\{\left(-C_{n m} V_{n+1, m+1}-S_{n m} W_{n+1, m+1}\right) + \frac{(n-m+2)!}{(n-m)!} \cdot\left(+C_{n m} V_{n+1, m-1}+S_{n m} W_{n+1, m-1}\right)\right\} & m > 0
\end{cases}
\\[2.5em]
& \ddot{y}_{n m} = 
\begin{cases}
    \frac{G M}{R_{\oplus}^2} \cdot\left\{-C_{n 0} W_{n+1,1}\right\} & m = 0 \\[1.0em]
    \frac{G M}{R_{\oplus}^2} \cdot \frac{1}{2} \cdot\left\{\left(-C_{n m} \cdot W_{n+1, m+1}+S_{n m} \cdot V_{n+1, m+1}\right) + \frac{(n-m+2)!}{(n-m)!} \cdot\left(-C_{n m} W_{n+1, m-1}+S_{n m} V_{n+1, m-1}\right)\right\} & m > 0
\end{cases}
\\[2.5em]
& \ddot{z}_{n m} = \frac{G M}{R_{\oplus}^2} \cdot\left\{(n-m+1) \cdot\left(-C_{n m} V_{n+1, m}-S_{n m} W_{n+1, m}\right)\right\}
\end{aligned}
```

and 

```math
V_{n m}=\left(\frac{R_{\oplus}}{r}\right)^{n+1} \cdot P_{n m}(\sin \phi) \cdot \cos m \lambda ,
\quad
W_{n m}=\left(\frac{R_{\oplus}}{r}\right)^{n+1} \cdot P_{n m}(\sin \phi) \cdot \sin m \lambda
```

(c.f. Montenbruck & Gill Chapter 3.2)

The expressions above use un-normalized coefficients and functions, which overflow or
underflow `Float64` and lose accuracy at high degree. The implementation therefore works
with the **fully-normalized** quantities $\bar{C}_{nm} = C_{nm}/N_{nm}$,
$\bar{V}_{nm} = N_{nm} V_{nm}$, $\bar{W}_{nm} = N_{nm} W_{nm}$, with
$N_{nm} = \sqrt{(2-\delta_{m0})(2n+1)(n-m)!/(n+m)!}$ (M&G Eq. (3.13)), so that
$C_{nm} V_{nm} = \bar{C}_{nm} \bar{V}_{nm}$. The $\bar{V}_{nm}, \bar{W}_{nm}$ are
evaluated in Cartesian coordinates with the normalized Cunningham (forward-column)
recursion

```math
\bar{V}_{mm} = D_m \left( \tilde{x}\, \bar{V}_{m-1,m-1} - \tilde{y}\, \bar{W}_{m-1,m-1} \right),
\qquad
\bar{V}_{nm} = A_{nm}\, \tilde{z}\, \bar{V}_{n-1,m} - B_{nm}\, \tilde{\rho}\, \bar{V}_{n-2,m}
```

(and likewise for $\bar{W}$), where $\tilde{x} = xR/r^2$, $\tilde{y} = yR/r^2$,
$\tilde{z} = zR/r^2$, $\tilde{\rho} = R^2/r^2$, $\bar{V}_{00} = R/r$,
$A_{nm} = \sqrt{(2n-1)(2n+1)/((n-m)(n+m))}$,
$B_{nm} = \sqrt{(2n+1)(n+m-1)(n-m-1)/((2n-3)(n+m)(n-m))}$ and
$D_m = \sqrt{k_m (2m+1)/(2m)}$ ($k_1 = 2$, otherwise 1). The factorial ratios of the
acceleration formulas are combined with the normalization ratios into precomputed
coefficients. The whole sum is evaluated in a single recursion pass
($\mathcal{O}(n_{\max}^2)$ operations, $\mathcal{O}(n_{\max})$ memory), without
factorials or `BigInt`, is non-singular at the poles, generic over `Real` (ForwardDiff
compatible), and stable to high degree: it is tested up to degree 180 (GGM03S); on a desktop CPU
one evaluation takes ~20 µs at degree 100 and ~60 µs at degree 180. Underflow of sectoral terms near the poles at degrees above
~1500 (Holmes & Featherstone, 2002) is not handled. Gravity files are read in the PDS
SHADR `.tab` format, fully normalized or un-normalized (normalized on load).

!!! warning

    The gravity field must be that of the central body (`naif_ids[1]`) and `frame_PCPF`
    must be the body-fixed frame the field is expressed in, e.g. `data/earth/GGM03S.tab`
    with `"ITRF93"` (or the lower-accuracy `"IAU_EARTH"`) for Earth-centred problems, and
    `data/luna/gggrx_1200l_sha_*.tab` with `"MOON_PA"` for Moon-centred problems.


### Solar Radiation Pressure 

#### Cannonball model

The cannonball model SRP acceleration is given by

```math
\boldsymbol{a}_{\mathrm{SRP}}(t) = 
P_{\odot} \left(\frac{\mathrm{AU}}{\| \boldsymbol{r}_{\odot} \|_2}\right)^2 C_r \frac{A}{m} \dfrac{\boldsymbol{r}_{\odot}}{\| \boldsymbol{r}_{\odot} \|_2}
```

where $\mathrm{AU}$ is the astronomical unit, $P_{\odot}$ is the radiation pressure at $1\,\mathrm{AU}$, $C_r$ is the radiation pressure coefficient, $A/m$ is the area-to-mass ratio, and $\boldsymbol{r}_{\odot}$ is the Sun-to-spacecraft position vector.
In `HighFidelityEphemerisModel`, the above is computed in canonical scales as

```math
\boldsymbol{a}_{\mathrm{SRP}}(t) = k_{\mathrm{SRP}} \dfrac{\boldsymbol{r}_{\odot}}{\| \boldsymbol{r}_{\odot} \|_2^3}
```

where $k_{\mathrm{SRP}}$

```math
k_{\mathrm{SRP}} = 
P_{\odot} \left(\frac{\mathrm{AU}}{\mathrm{DU}}\right)^2 C_r \frac{A/m}{10^3}  \dfrac{\mathrm{TU}^2}{\mathrm{DU}}
```

is pre-computed and stored. No eclipse/shadow model is applied: the spacecraft is always assumed to be in sunlight.


### Atmospheric drag

Quadratic drag in the inertial frame is modeled as

```math
\boldsymbol{a}_{\mathrm{drag}} = -\tfrac{1}{2}\,\rho\,C_d\,\frac{A}{m}\,|\mathbf{v}_{\mathrm{rel}}|\,\mathbf{v}_{\mathrm{rel}}
```

where $\rho$ is atmospheric density (kg/m³) from a user callback, $C_d$ is the drag coefficient, $A/m$ is the area-to-mass ratio, and $\mathbf{v}_{\mathrm{rel}} = \mathbf{v} - \mathbf{v}_{\mathrm{atm}}$ is the spacecraft velocity relative to a co-rotating atmosphere ($\mathbf{v}_{\mathrm{atm}} = T_U\,\boldsymbol{\omega}_{\mathrm{atm}} \times \mathbf{r}$ in canonical units).

The pre-computed coefficient $k_{\mathrm{drag}}$ absorbs $\tfrac{1}{2} C_d (A/m)$ and SI-to-canonical conversion so that

```math
\boldsymbol{a}_{\mathrm{drag}} = -k_{\mathrm{drag}}\,\rho\,|\mathbf{v}_{\mathrm{rel}}|\,\mathbf{v}_{\mathrm{rel}}.
```

The density callback `f_density(et, r_km)` receives the epoch and the spacecraft position
in km in the body-fixed frame `frame_PCPF`. Two Earth models are provided:

- `harris_priester_f_density(...)`: the Harris-Priester model of Montenbruck & Gill
  (*Satellite Orbits*, §3.5.2), with geodetic altitude above an oblate ellipsoid
  (WGS-84 by default), exponential interpolation of the antapex/apex density tables
  $\rho_{\min}(h)$, $\rho_{\max}(h)$ (constant scale height per table interval), and the
  diurnal bulge

  ```math
  \rho = \rho_{\min}(h) + \left(\rho_{\max}(h) - \rho_{\min}(h)\right) \cos^n(\psi/2),
  ```

  where $\psi$ is the angle between the position vector and the bulge apex, located at the
  Sun's declination and 30° east of the Sun in right ascension (≈ 14h local time).
  M&G recommend $n = 2$ for low-inclination and $n = 6$ (the default) for polar orbits.
  By default the Sun comes from a SPICE-free low-precision analytic model (≈ 0.01° from
  ITRF93), so the callback is thread-safe; a user `sun_position(et)` function may be
  supplied instead. Caveats:

  - **`r_km` must be Earth-fixed.** The bulge is placed using the Sun in an Earth-fixed
    frame, so `frame_PCPF` must be an Earth body-fixed frame (`"ITRF93"`, `"IAU_EARTH"`);
    with an inertial `frame_PCPF` the bulge is at the wrong local time (no error is
    raised). A user `sun_position` must return the Sun in the same frame as `r_km`.
  - **Zero density outside 100-1000 km.** As in M&G, the density is zero below 100 km
    (geodetic) and above 1000 km. A re-entering spacecraft therefore loses all drag below
    100 km, where the density drops discontinuously from ≈5e-7 kg/m³ to 0: terminate the
    propagation with an altitude event or use another model there
    (`jacchia_roberts_f_density` throws an error at or below 100 km).
  - **Only C⁰ in altitude.** $\partial\rho/\partial h$ jumps at each table height
    (every 10 km between 120 and 300 km, after a 20 km first step from 100 km; 20-40 km above). ForwardDiff STMs are still the model's
    derivative almost everywhere, but on orbits that cross table heights they degrade
    against finite-difference STMs (≈4e-4 discrepancy observed over a few hours, vs
    ≲1e-5 within one interval), and gradient-based optimization may see a non-smooth
    objective.
- `jacchia_roberts_f_density(...)`: the Jacchia-Roberts model (GMAT heritage) with
  constant F10.7, F10.7a and Kp. With the default `sun_model = :spice` it calls SPICE in
  every evaluation and is **not thread-safe**; use `sun_model = :analytic` or a
  `sun_position` function for a SPICE-free, thread-safe callback.


## Parameters and equations of motion

Backends are selected by the parameter type:

- `SpiceParameters`: query SPICE at runtime
- `EphemeridesParameters`: use Ephemerides.jl/FrameTransformations.jl at runtime

The preferred EOM names are generic and dispatch on the parameter type:

- `eom_Nbody!` / `eom_Nbody`
- `eom_NbodySH!` / `eom_NbodySH`

The compatibility constructor `HighFidelityEphemerisModelParameters(...)` remains available only for backward compatibility and returns one of the concrete parameter types based on the supplied backend inputs.

Backend-specific names (`eom_Nbody_SPICE!`, `eom_Nbody_Ephemerides!`, `eom_NbodySH_SPICE!`, `eom_NbodySH_Ephemerides!`) remain available for compatibility and backend-specific tests.

The table below summarizes the models. Note:

- `Nbody`: central gravity term + third-body perturbations ($\boldsymbol{a}_{\mathrm{3bd},i}$), optional SRP and drag
- `NbodySH`: central gravity term + third-body perturbations + spherical harmonics perturbations up to `nmax` degree ($\boldsymbol{a}_{\mathrm{SH},n_{\max}}$), optional SRP and drag
- The STM is integrated with the Jacobian, which is computed via `ForwardDiff` (functions containing `_fd`)

| Parameters | Translational EOM | STM EOM | `EnsembleThreads` compatibility | Notes |
|------------|-------------------|---------|---------------------------------|-------|
| `SpiceParameters` | `eom_Nbody!`, `eom_NbodySH!` | `eom_stm_Nbody_SPICE_fd!`, `eom_stm_NbodySH_SPICE_fd!` | no | direct SPICE calls |
| `EphemeridesParameters` | `eom_Nbody!`, `eom_NbodySH!` | `eom_stm_Nbody_Ephemerides_fd!`, `eom_stm_NbodySH_Ephemerides_fd!` | yes | preferred non-SPICE backend |


!!! note

    All backends accept ForwardDiff dual numbers in the **state** (this is how the `_fd`
    STM equations of motion compute the Jacobian, also with `SpiceParameters`). Dual numbers
    in **time** (e.g. derivatives with respect to the epoch) are only supported by
    `EphemeridesParameters`, since `SpiceParameters` passes the epoch to SPICE.
    `EphemeridesParameters` also avoids SPICE calls inside the EOM altogether, so it is
    thread-safe and faster.

!!! note

    `EnsembleThreads` compatibility also requires a thread-safe `f_density` when drag is
    enabled: the default `harris_priester_f_density()` and
    `jacchia_roberts_f_density(sun_model = :analytic)` are; the SPICE-based
    `jacchia_roberts_f_density()` is not (CSPICE is not thread-safe).

!!! note

    The built-in Ephemerides.jl frame system supports the body-fixed frames `"MOON_PA"`
    (lunar binary PCK, e.g. `moon_pa_de440_200625.bpc`), `"ITRF93"` (binary Earth PCK,
    e.g. `earth_latest_high_prec.bpc`) and `"IAU_EARTH"` (IAU 2009 rotation model, no
    kernel needed). Unsupported body-fixed frames (e.g. `"IAU_MOON"`) raise an error at
    construction; a custom `FrameTransformations.FrameSystem` can be passed as
    `ephemerides_frame_system`, in which case `frame_PCPF` is not checked.

!!! compat "Interpolated backend removed after v0.3.0"

    The interpolated-ephemeris backend (`InterpParameters`, the `eom_*_Interp*` functions,
    `InterpolatedEphemeris` and `InterpolatedTransformation`) has been removed. Use
    `EphemeridesParameters` for SPICE-free, thread-safe and ForwardDiff-compatible
    propagation. Passing `interpolate_ephem_span` or `interpolation_time_step` to a
    parameter constructor now throws an `ArgumentError`.