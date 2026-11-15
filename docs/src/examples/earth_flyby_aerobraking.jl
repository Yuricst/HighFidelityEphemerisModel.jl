# # Aerobraking during an Earth flyby
#
# !!! tip "Running this example"
#
#     This page is generated with [Literate.jl](https://github.com/fredrikekre/Literate.jl) from the script [`docs/src/examples/earth_flyby_aerobraking.jl`](https://github.com/Yuricst/HighFidelityEphemerisModel.jl/blob/main/docs/src/examples/earth_flyby_aerobraking.jl), which is run when the documentation is built. To run it yourself, from the repository root:
#
#     ```bash
#     julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#     julia --project=docs --threads=auto docs/src/examples/earth_flyby_aerobraking.jl
#     ```
#
#     or `include` the script in a REPL started with `julia --project=docs --threads=auto` to display the figures. The ensemble propagations use all the threads Julia was started with.
#
#     SPICE kernels are loaded from the directory given by the `SPICE` environment variable, which should contain the `lsk/`, `spk/`, and `pck/` subfolders of the [NAIF generic kernels](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/) used below.
#
# A spacecraft that flies by the Earth low enough passes through the upper atmosphere, where drag removes part of its orbital energy. This example sweeps the periapsis altitude and the hyperbolic excess speed $v_\infty$ of an Earth flyby, propagating all flybys in parallel as `EnsembleProblem`s, and looks at
#
# - the drag deceleration during the atmospheric pass,
# - the energy removed by drag, expressed as an equivalent impulsive $\Delta v$ at periapsis, and
# - the outgoing characteristic energy $C_3 = v_\infty^2$, which tells whether the spacecraft is captured ($C_3 < 0$).

# ## Setup

using LinearAlgebra
using Printf
using OrdinaryDiffEq
using SPICE
using PlotlyBase
using HighFidelityEphemerisModel

## load SPICE kernels (used for the GMs, the epoch, and post-processing)
spice_dir = ENV["SPICE"]
furnsh(joinpath(spice_dir, "lsk", "naif0012.tls"))
furnsh(joinpath(spice_dir, "spk", "de440.bsp"))
furnsh(joinpath(spice_dir, "pck", "gm_de440.tpc"))
furnsh(joinpath(spice_dir, "pck", "earth_latest_high_prec.bpc"))   # ITRF93 frame

println("Julia is running with $(Threads.nthreads()) threads")

# ## Dynamics model
#
# The spacecraft moves about the Earth (the central body), perturbed by the Moon and the Sun and by an 8×8 Earth gravity field (GGM03S, evaluated in the `ITRF93` frame). Ephemerides and the `J2000` to `ITRF93` rotation are evaluated with the thread-safe Ephemerides.jl backend.
#
# We build two sets of parameters with the same gravitational model: one without drag, used to design the flybys and as a reference, and one with drag. The drag model uses the Harris–Priester density with its diurnal bulge and the geodetic altitude above the WGS-84 ellipsoid; `harris_priester_f_density` uses an analytic Sun position, so the callback is thread-safe.

# Ephemerides.jl warns that it cannot parse the comment area of some versions of #src
# earth_latest_high_prec.bpc; this is harmless, so the warning is hidden on the docs page. #src
#md Base.CoreLogging.disable_logging(Base.CoreLogging.Warn) #hide
naif_ids = ["399", "301", "10"]             # Earth (central body), Moon, Sun
GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
μ = GMs[1]                                  # km^3/s^2
R_eq, flattening = 6378.137, 1 / 298.257223563   # WGS-84 ellipsoid
DU = R_eq                                   # canonical distance unit, km
et0 = str2et("2026-09-15T12:00:00")         # epoch of the periapsis passage (UTC)

## dynamics shared by both models: Earth gravity field (8x8), lunar and solar third-body gravity
model = (
    filepath_spherical_harmonics = joinpath(pkgdir(HighFidelityEphemerisModel), "data", "earth", "GGM03S.tab"),
    nmax = 8,
    frame_PCPF = "ITRF93",
    ephemerides_files = [
        joinpath(spice_dir, "spk", "de440.bsp"),
        joinpath(spice_dir, "pck", "earth_latest_high_prec.bpc"),
    ],
)
params_nodrag = EphemeridesParameters(et0, DU, GMs, naif_ids, "J2000", "NONE"; model...)

## same model + drag, with the Harris-Priester density (n = 2 for a low-inclination pass)
drag_Cd, drag_Am = 2.2, 0.02                # drag coefficient, area-to-mass ratio in m^2/kg
params_drag = EphemeridesParameters(et0, DU, GMs, naif_ids, "J2000", "NONE"; model...,
    include_drag = true,
    drag_Cd = drag_Cd,
    drag_Am = drag_Am,
    f_density = harris_priester_f_density(R_eq; flattening = flattening, n = 2),
)
TU, VU = params_drag.TU, params_drag.VU;
#md Base.CoreLogging.disable_logging(Base.CoreLogging.BelowMinLevel); nothing #hide

# !!! warning
#
#     The Harris–Priester density is zero outside 100–1000 km (geodetic altitude), so the drag of a pass below 100 km would be silently underestimated. All the flybys below stay above 100 km, which we check after the propagation.

# !!! note
#
#     `EnsembleThreads()` evaluates the equations of motion from several threads at once, so the parameters must be thread-safe. `EphemeridesParameters` are, as long as `f_density` is (the default Harris–Priester callback is, see [Perturbations](../tutorials/perturbations.md)). `SpiceParameters` are not, because CSPICE is not thread-safe: with them, use `EnsembleSerial()`.

# ## Flyby geometry
#
# Each flyby is defined by its periapsis: periapsis altitude $h_p$ above the equatorial radius and excess speed $v_\infty$, with an inclination of 10° and the periapsis at the ascending node (over the equator, so that $h_p$ is also close to the geodetic altitude). The periapsis speed follows from the energy $v_\infty^2 / 2$ including the $J_2$ term of the gravitational potential: at 100–200 km altitude over the equator, a pure two-body design would give a $C_3$ about 0.06 km²/s² lower than $v_\infty^2$.

J2 = 1.0826e-3                              # Earth oblateness coefficient

"""
Periapsis state (canonical units) of a flyby with periapsis altitude `h_p` (km above the
equatorial radius) and hyperbolic excess speed `v_inf` (km/s). The periapsis is placed at
the ascending node, i.e. over the equator, where the J2 term deepens the gravitational
potential by μ J2 R_eq² / (2 r_p³); the periapsis speed accounts for it so that the
energy far from the Earth is v_inf² / 2.
"""
function periapsis_state(h_p, v_inf; inc = deg2rad(10), raan = deg2rad(30))
    r_p = R_eq + h_p
    v_p = sqrt(v_inf^2 + 2μ / r_p * (1 + J2 / 2 * (R_eq / r_p)^2))
    r = r_p * [cos(raan), sin(raan), 0]
    v = v_p * [-sin(raan) * cos(inc), cos(raan) * cos(inc), sin(inc)]
    return [r / DU; v / VU]
end

h_ps   = 102.5:2.5:180.0                    # periapsis altitudes, km
v_infs = [0.5,1.0,2.0]               # hyperbolic excess speeds, km/s
cases  = [(h_p, v_inf) for h_p in h_ps, v_inf in v_infs]   # every combination
println("number of flybys: ", length(cases))

T_in, T_out = 3 * 3600 / TU, 6 * 3600 / TU; # propagate from 3 h before to 6 h after periapsis

# We obtain the initial states 3 hours before periapsis by propagating the periapsis states *backward* in time with the drag-free model. This is a first ensemble, with a negative time span; `output_func` keeps only the final state of each trajectory. At these initial states, 60,000 km or more from the Earth, the two-body $C_3$ is close to the design value $v_\infty^2$.

x_peri = [periapsis_state(c...) for c in vec(cases)]

prob_back = ODEProblem(eom_NbodySH!, x_peri[1], (0.0, -T_in), params_nodrag)
ensemble_back = EnsembleProblem(prob_back;
    prob_func   = (prob, i, repeat) -> remake(prob; u0 = x_peri[i]),
    output_func = (sol, i) -> (sol.u[end], false),   # only keep the state at t = -T_in
    safetycopy  = false,
)
@time x_start = solve(ensemble_back, Vern9(), EnsembleThreads();
    trajectories = length(x_peri), reltol = 1e-12, abstol = 1e-12).u;

## check: two-body C3 at the initial states (60,000 km away or more) vs. the design v∞²
energy(x) = (VU * norm(x[4:6]))^2 / 2 - μ / (DU * norm(x[1:3]))   # two-body energy, km^2/s^2
C3_in = reshape([2 * energy(x) for x in x_start], size(cases))
@printf("largest |C3_in - v∞²|: %.4f km²/s²\n", maximum(abs.(C3_in .- [c[2]^2 for c in cases])))

# ## Propagating the flybys
#
# From these initial states, we propagate all flybys forward to 6 hours after periapsis, once with drag and once without. Since the drag-free dynamics are the same as in the backward propagation, the drag-free flybys pass through the designed periapsis states, and comparing each flyby with its drag-free twin isolates the effect of drag from that of the other perturbations.

# !!! warning
#
#     When a `prob_func` is given, `EnsembleProblem` by default deep-copies the problem, including its parameters, for every trajectory (`safetycopy = true`). For `EphemeridesParameters` this copies the loaded ephemeris data each time, about 130 MB per trajectory for DE440 in this example, which makes the ensemble many times slower. The equations of motion do not modify the parameters, so pass `safetycopy = false`.

"Propagate all flybys from `x_start` with the dynamics defined by `params`."
function propagate_flybys(params)
    prob = ODEProblem(eom_NbodySH!, x_start[1], (-T_in, T_out), params)
    ensemble = EnsembleProblem(prob;
        prob_func  = (prob, i, repeat) -> remake(prob; u0 = x_start[i]),
        safetycopy = false,
    )
    return solve(ensemble, Vern9(), EnsembleThreads();
        trajectories = length(x_start), reltol = 1e-12, abstol = 1e-12)
end

@time sols_drag = propagate_flybys(params_drag)
@time sols_nodrag = propagate_flybys(params_nodrag)
println("all successful: ", all(SciMLBase.successful_retcode, [sols_drag.u; sols_nodrag.u]))

# ## Post-processing
#
# For each flyby we compute
#
# - the geodetic altitude and the drag deceleration $\frac{1}{2} \rho C_D \frac{A}{m} v_{\mathrm{rel}}^2$ during the pass, from the state in `ITRF93` (whose velocity is the velocity relative to the co-rotating atmosphere) and the same density callback as the dynamics,
# - the two-body energy $\varepsilon = v^2/2 - \mu/r$ at the final time, with and without drag; the energy removed by drag is expressed as an equivalent impulsive $\Delta v \approx (\varepsilon_{\text{no drag}} - \varepsilon_{\text{drag}}) / v_p$ at periapsis, and the outgoing $C_3 = 2 \varepsilon_{\text{drag}}$.
#
# This step uses SPICE (`sxform`, `recgeo`), so it runs serially.

D = [fill(DU, 3); fill(VU, 3)]              # canonical -> km, km/s

"Geodetic altitude (km) and drag deceleration (m/s^2) along `sol` at times `ts`."
function pass_profile(sol, ts)
    h, a = similar(ts), similar(ts)
    for (k, t) in enumerate(ts)
        et = et0 + t * TU
        x_itrf = sxform("J2000", "ITRF93", et) * (D .* sol(t))
        h[k] = recgeo(x_itrf[1:3], R_eq, flattening)[3]
        ρ = params_drag.f_density(et, x_itrf[1:3])                  # kg/m^3
        v_rel = 1e3 * norm(x_itrf[4:6])     # m/s, w.r.t. the co-rotating atmosphere
        a[k] = 0.5 * ρ * drag_Cd * drag_Am * v_rel^2
    end
    return h, a
end

ts_pass = range(-0.25 * 3600 / TU, 0.25 * 3600 / TU, length = 1801)   # ± 15 min, 1 s steps
profiles = [pass_profile(sol, ts_pass) for sol in sols_drag.u]

h_min = reshape([minimum(p[1]) for p in profiles], size(cases))     # km
a_max = reshape([maximum(p[2]) for p in profiles], size(cases))     # m/s^2
E_drag   = reshape([energy(sol.u[end]) for sol in sols_drag.u], size(cases))
E_nodrag = reshape([energy(sol.u[end]) for sol in sols_nodrag.u], size(cases))

## energy removed by drag, expressed as an impulsive Δv at periapsis: ΔE ≈ v_p Δv
v_p = [VU * norm(periapsis_state(c...)[4:6]) for c in cases]
Δv_drag = 1e3 * (E_nodrag - E_drag) ./ v_p  # m/s
C3_out = 2 * E_drag                         # outgoing C3, km^2/s^2 (< 0: captured)
@printf("lowest geodetic altitude reached: %.2f km\n", minimum(h_min))

#-

g0 = 9.80665                                # standard gravity, m/s^2
j = findfirst(==(1.0), v_infs)              # v_inf = 1 km/s
println(" h_p [km]   h_min [km]   peak decel [g]   Δv_drag [m/s]   C3_out [km²/s²]")
for i in 1:4:length(h_ps)
    @printf("%8.1f %12.2f %16.4f %15.3f %17.3f%s\n", h_ps[i], h_min[i, j], a_max[i, j] / g0,
        Δv_drag[i, j], C3_out[i, j], C3_out[i, j] < 0 ? "   captured" : "")
end

println("\ncaptured flybys (C3 < 0):")
for idx in findall(<(0), C3_out)
    h_p, v_inf = cases[idx]
    r_a = 2μ / -C3_out[idx]                 # apoapsis radius ≈ 2a for a near-parabolic orbit
    @printf("  v∞ = %.1f km/s, h_p = %5.1f km: C3 = %6.3f km²/s², apoapsis ≈ %.2f million km\n",
        v_inf, h_p, C3_out[idx], r_a / 1e6)
end

# ## Results
#
# The figures below are interactive: hover for values, drag to zoom, and click on legend entries to hide traces.
#
# ### The atmospheric pass

palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical, fixed order
r4(x) = round.(x; sigdigits = 4)            # keep the data embedded in the figures small
t_min = ts_pass * TU / 60                   # minutes from periapsis

j = findfirst(==(1.0), v_infs)
fig = Plot(Layout(Subplots(rows = 2, cols = 1, shared_xaxes = true, vertical_spacing = 0.06);
    height = 600, template = templates.plotly_white, hovermode = "x unified",
    title = "Atmospheric pass for v∞ = 1 km/s",
    xaxis2_title = "time from periapsis, min",
    yaxis_title = "geodetic altitude, km",
    yaxis2_title = "drag deceleration, m/s²", yaxis2_type = "log"))
for (c, h_p) in enumerate((102.5, 110.0, 125.0, 150.0))
    i = findfirst(≈(h_p), h_ps)
    h, a = profiles[LinearIndices(cases)[i, j]]
    keep = a .> 0                           # Harris-Priester is zero above 1000 km
    for (row, y, x) in ((1, h, t_min), (2, a[keep], t_min[keep]))
        add_trace!(fig, scatter(x = r4(x[1:5:end]), y = r4(y[1:5:end]), mode = "lines",
            line = attr(color = palette[c], width = 2), name = "h_p = $(h_p) km",
            legendgroup = "$c", showlegend = row == 1); row)
    end
end
## (axis ranges are set per subplot after creating the figure; log axes take log10 values)
fig.layout["yaxis_range"] = [95,250]
fig.layout["yaxis2_range"] = [-6,0.5]
fig.layout["xaxis_range"] = fig.layout["xaxis2_range"] = [-6,6]
fig
#md Main.docs_figure(fig) #hide

# Drag acts only during a few minutes around periapsis, and its peak grows by more than two orders of magnitude when the periapsis is lowered from 150 km to 102.5 km.
#
# ### Drag losses

## the drag losses barely depend on v∞: the periapsis speed only varies by a few percent
spread = maximum(abs.(Δv_drag ./ Δv_drag[:, 1] .- 1))
@printf("largest relative difference of Δv_drag between the v∞ values: %.1f %%\n", 100 * spread)

fig = Plot(Layout(Subplots(rows = 1, cols = 2, horizontal_spacing = 0.12);
    height = 420, template = templates.plotly_white, hovermode = "x unified", showlegend = false,
    title = "Drag losses and peak deceleration (v∞ = 1 km/s)",
    xaxis_title = "periapsis altitude h_p, km", xaxis2_title = "periapsis altitude h_p, km",
    yaxis_title = "equivalent Δv at periapsis, m/s", yaxis_type = "log",
    yaxis2_title = "peak drag deceleration, m/s²", yaxis2_type = "log"))
j = findfirst(==(1.0), v_infs)
add_trace!(fig, scatter(x = collect(h_ps), y = r4(Δv_drag[:, j]), mode = "lines+markers",
    line = attr(color = palette[1], width = 2), marker = attr(size = 8), name = "Δv"); col = 1)
add_trace!(fig, scatter(x = collect(h_ps), y = r4(a_max[:, j]), mode = "lines+markers",
    line = attr(color = palette[1], width = 2), marker = attr(size = 8), name = "peak deceleration"); col = 2)
fig.layout["yaxis_range"] = [-1,2]       # log10 values
fig.layout["yaxis2_range"] = [-4,0.3]
fig
#md Main.docs_figure(fig) #hide

# The equivalent $\Delta v$ grows roughly exponentially as the periapsis is lowered, following the density profile. It barely depends on $v_\infty$: a faster flyby has a higher periapsis speed (by 1–2 % here), but its more open hyperbola also curves away from the Earth faster, so it spends a shorter arc in the atmosphere, and the two effects nearly cancel.
#
# ### Capture

fig = Plot(Layout(height = 450, template = templates.plotly_white, hovermode = "x unified",
    title = "Outgoing C3 after the atmospheric pass",
    xaxis_title = "periapsis altitude h_p, km",
    yaxis_title = "C3 = v∞², km²/s²",
    annotations = [attr(x = 180, y = 0, text = "captured below this line", showarrow = false,
        xanchor = "right", yanchor = "top", font = attr(color = "#52514e"))]))
add_hline!(fig, 0.0; line = attr(color = "#52514e", dash = "dash", width = 1))
for (j, v_inf) in enumerate(v_infs)
    add_trace!(fig, scatter(x = collect(h_ps), y = r4(C3_out[:, j]), mode = "lines+markers",
        line = attr(color = palette[j], width = 2), marker = attr(size = 8),
        name = "v∞ = $(v_inf) km/s (C3 before the pass: $(v_inf^2) km²/s²)"))
end
fig
#md Main.docs_figure(fig) #hide

# Since the drag losses are nearly the same for all $v_\infty$, the slowest flybys are the easiest to capture: in this sweep, capture occurs only for periapsis altitudes up to 112.5 km at $v_\infty = 0.5$ km/s and only for the lowest pass at $v_\infty = 1$ km/s, while the 2 km/s flybys all escape.
#
# !!! note
#
#     $C_3 < 0$ only means that the spacecraft is bound to the Earth in the two-body sense. The captured orbits printed above have apoapses of at least about 0.8 million km, about twice the lunar distance, and most of them lie beyond the Earth's Hill sphere (about 1.5 million km), so their fate is decided by solar and lunar perturbations. A capture into a lower orbit needs larger drag losses: a lower periapsis, which is outside the validity range of the Harris–Priester model, or a larger $C_D A/m$.
#
# ## Going further
#
# - Add a `ContinuousCallback` on the geodetic altitude to stop the propagation below 100 km, the lower limit of the Harris–Priester model.
# - Use `jacchia_roberts_f_density(sun_model = :analytic)` for a density model that depends on the solar and geomagnetic activity; it is also thread-safe.
# - Sweep the area-to-mass ratio as well: in `prob_func`, return `remake(prob; u0, p)` with `p = copy(params_drag)` (which shares the ephemeris data) and `p.k_drag` scaled in proportion to the area-to-mass ratio.
