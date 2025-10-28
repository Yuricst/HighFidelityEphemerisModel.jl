# Jacobians & Hessians

When designing certain algorithms (e.g. differential correction, nonlinear programming solvers, DDP, etc.), it may be necessary to evaluate the Jacobian or Hessian of the dynamics at a given state $\boldsymbol{x}$ and time $t$.

!!! note
    
    In anticipation of such algorithms also having controls $\boldsymbol{u}$ as inputs, the functions `eom_jacobian_fd` and `eom_hessian_fd` take as input `u`, a place-holder argument for control, as input. For dynamics that do not contain control (e.g. `HighFidelityEphemerisModel.eom_Nbody(or NbodySH)_SPICE(or Interp)`), you just need to pass anything (e.g. `0.0`, `nothing`, etc.).

We begin by defining the parameters object (same setup as when we solve an IVP): 

```julia
using SPICE
using HighFidelityEphemerisModel

# load SPICE kernels
spice_dir = ENV["SPICE"]
furnsh(joinpath(spice_dir, "lsk", "naif0012.tls"))
furnsh(joinpath(spice_dir, "spk", "de440.bsp"))
furnsh(joinpath(spice_dir, "pck", "gm_de440.tpc"))
furnsh(joinpath(spice_dir, "pck", "moon_pa_de440_200625.bpc"))
furnsh(joinpath(spice_dir, "fk", "moon_de440_250416.tf"))

# define parameters as usual
define parameters
naif_ids = ["301", "399", "10"]
GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
naif_frame = "J2000"
abcorr = "NONE"
DU = 3000.0
filepath_spherical_harmonics = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab")
nmax = 4

et0 = str2et("2020-01-01T00:00:00")
etf = et0 + 30 * 86400.0
interpolate_ephem_span = [et0, etf]
interpolation_time_step = 30.0
parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
    et0, DU, GMs, naif_ids, naif_frame, abcorr;
    interpolate_ephem_span=interpolate_ephem_span,
    filepath_spherical_harmonics = filepath_spherical_harmonics,
    nmax = nmax,
    frame_PCPF = "MOON_PA",
    interpolation_time_step = interpolation_time_step,
)
```

Now we can call `eom_jacobian_fd` to get the 6-by-6 Jacobian, or `eom_hessian_fd` to get the 6-by-6-by-6 Hessian.

```julia
# evaluate Jacobian & Hessian
x0 = [1.0, 0.0, 0.3, 0.5, 1.0, 0.0]     # state when we want to evaluate Jacobian/Hessian, in canonical scales
t = 5.1                                 # time when we want to evaluate Jacobian/Hessian, in canonical time

jac_fd = HighFidelityEphemerisModel.eom_jacobian_fd(
    HighFidelityEphemerisModel.eom_Nbody_Interp,      # this can be some other static equations of motion
    x0,
    0.0,                                              # place-holder for control argument
    parameters,
    t
)

hess_fd = HighFidelityEphemerisModel.eom_hessian_fd(
    HighFidelityEphemerisModel.eom_Nbody_Interp,      # this can be some other static equations of motion
    x0,
    0.0,                                              # place-holder for control argument
    parameters,
    t
)
```

!!! warning

    The first argument to either `eom_jacobian_fd` or `eom_hessian_fd` is expected to be in allocating form, i.e. of the form 

    ```julia
    function lorenz(u, p, t)
        dx = 10.0 * (u[2] - u[1])
        dy = u[1] * (28.0 - u[3]) - u[2]
        dz = u[1] * u[2] - (8 / 3) * u[3]
        return [dx, dy, dz]
    end
    ```

    instead of 

    ```julia
    function lorenz!(du, u, p, t)
        du[1] = 10.0 * (u[2] - u[1])
        du[2] = u[1] * (28.0 - u[3]) - u[2]
        du[3] = u[1] * u[2] - (8 / 3) * u[3]
        return nothing
    end
    ```

    (c.f. [docs from `DifferentialEquations.jl`](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/faster_ode_example/)).
    If using an eom function from `HighFidelityEphemerisModel`, as per convention, make sure to use equations of motion *without* the `!` at the end of the name.