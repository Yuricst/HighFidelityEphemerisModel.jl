# Overview

There are a number of equations of motion implemented in `HighFidelityEphemerisModel.jl`.

- functions starting with `eom_` integrates the translational state (`[x,y,z,vx,vy,vz]`)
- functions starting with `eom_stm_` integrates both the translational state and the flattened 6-by-6 STM.

!!! note

    The STM is flattened row-wise, so to extract the state & STM from the `ODESolution`, make sure to reshape then tranapose; for example,

    ```julia
    x_stm_tf = sol.u[end]                       # concatenated state & STM (flattened)
    x_tf     = x_stm_tf[1:6]                    # final state [x,y,z,vx,vy,vz]
    STM_tf   = reshape(sol.u[end][7:42],6,6)'   # final 6-by-6 STM
    ```

## Dynamics model

In `HighFidelityEphemerisModel.jl`, the dynamcis consists of the central gravitational term, together with the following perturbations:

- third-body perturbations
- spherical harmonics
- solar radiation pressure

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
In `HEFM.jl`, this term is implemented using Battin's $F(q)$ function:

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


### Solar Radiation Pressure 

#### Cannonball model

The cannonball model SRP acceleration is given by

```math
\boldsymbol{a}_{\mathrm{SRP}}(t) = 
P_{\odot} \left(\frac{\mathrm{AU}}{\| \boldsymbol{r}_{\odot} \|_2}\right)^2 C_r \frac{A}{m} \dfrac{\boldsymbol{r}_{\odot}}{\| \boldsymbol{r}_{\odot} \|_2}
```

where $\mathrm{AU}$ is the astronomical unit, $P_{\odot}$ is the ratiation pressure at $1\,\mathrm{AU}$, $C_r$ is the radiation pressure coefficient, $A/m$ is the area-to-mass ratio, and $\boldsymbol{r}_{\odot}$ is the Sun-to-spacecraft position vector.
In `HighFidelityEphemerisModel`, the above is computed in canonical scales as

```math
\boldsymbol{a}_{\mathrm{SRP}}(t) = k_{\mathrm{SRP}} \dfrac{\boldsymbol{r}_{\odot}}{\| \boldsymbol{r}_{\odot} \|_2^3}
```

where $k_{\mathrm{SRP}}$

```math
k_{\mathrm{SRP}} = 
P_{\odot} \left(\frac{\mathrm{AU}}{\mathrm{DU}}\right)^2 C_r \frac{A/m}{10^3}  \dfrac{\mathrm{TU}^2}{\mathrm{DU}}
```

is pre-computed and stored.


## List of equations of motion in `HighFidelityEphemerisModel.jl`

The table below summarizes the equations of motion. Note: 

- `Nbody`: central gravity term + third-body perturbations ($\boldsymbol{a}_{\mathrm{3bd},i}$)
- `NbodySH`: central gravity term + third-body perturbations + spherical harmonics perturbations up to `nmax` degree ($\boldsymbol{a}_{\mathrm{SH},n_{\max}}$)
- The STM is integrated with the Jacobian, which is computed either analytically (using symbolic derivative) or via `ForwardDiff` (functions containing `_fd`)

| eom                   | eom + STM (analytical)  | eom + STM (ForwardDiff)      | `EnsembleThreads` compatibility |
|-----------------------|-------------------------|------------------------------|---------------------------------|
| `eom_Nbody_SPICE!`    | `eom_stm_Nbody_SPICE!`  | `eom_stm_Nbody_SPICE_fd!`    | no                              |
| `eom_Nbody_Interp!`   | `eom_stm_Nbody_Interp!` | `eom_stm_Nbody_Interp_fd!`   | yes                             |
| `eom_NbodySH_SPICE!`  |                         | `eom_stm_NbodySH_SPICE_fd!`  | no                              |
| `eom_NbodySH_Interp!` |                         | `eom_stm_NbodySH_Interp_fd!` | yes                             |


!!! note

    In order to use Julia's dual numbers, make sure to use a function that does not contain SPICE calls (i.e. use ones with `_interp` in the name); this is enabled by interpolating ahead of time ephemerides/transformation matrices.

!!! warning

    The accuracy of interpolated equations of motion (with `_interp` in the name) depends on the `interpolation_time_step`; if high-accuracy integration is required, it is advised to directly use the equations of motion that internally call SPICE (i.e. with `_SPICE` in the name)

