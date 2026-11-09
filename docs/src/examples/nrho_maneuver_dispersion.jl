# # Maneuver dispersion analysis on an NRHO
#
# !!! tip "Running this example"
#
#     This page is generated with [Literate.jl](https://github.com/fredrikekre/Literate.jl) from the script [`docs/src/examples/nrho_maneuver_dispersion.jl`](https://github.com/Yuricst/HighFidelityEphemerisModel.jl/blob/main/docs/src/examples/nrho_maneuver_dispersion.jl), which is run when the documentation is built. To run it yourself, from the repository root:
#
#     ```bash
#     julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#     julia --project=docs --threads=auto docs/src/examples/nrho_maneuver_dispersion.jl
#     ```
#
#     or `include` the script in a REPL started with `julia --project=docs --threads=auto` to display the figures. The ensemble propagations use all the threads Julia was started with.
#
#     SPICE kernels are loaded from the directory given by the `SPICE` environment variable, which should contain the `lsk/`, `spk/`, and `pck/` subfolders of the [NAIF generic kernels](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/) used below.
#
# This example estimates how the execution errors of a maneuver performed at apolune of a near-rectilinear halo orbit (NRHO) grow over the following revolutions. We
#
# 1. set up a thread-safe, high-fidelity lunar dynamics model with `EphemeridesParameters`,
# 2. sample the executed maneuver from an execution-error model and propagate all samples in parallel as an `EnsembleProblem` solved with `EnsembleThreads()`,
# 3. compare the Monte Carlo dispersions with a linear covariance analysis that maps the maneuver covariance with the state-transition matrix (STM).
#
# The NRHO is the 9:2 synodic-resonant southern L2 NRHO baselined for NASA's Gateway, with a period of about 6.56 days and perilune and apolune radii of roughly 3,400 km and 71,000 km.

# ## Setup

using LinearAlgebra
using Printf
using Random
using Statistics
using OrdinaryDiffEq
using SPICE
using PlotlyBase
using HighFidelityEphemerisModel

## load SPICE kernels (used for the GMs, the epoch, and post-processing)
spice_dir = ENV["SPICE"]
furnsh(joinpath(spice_dir, "lsk", "naif0012.tls"))
furnsh(joinpath(spice_dir, "spk", "de440.bsp"))
furnsh(joinpath(spice_dir, "pck", "gm_de440.tpc"))

println("Julia is running with $(Threads.nthreads()) threads")

# ## Dynamics model
#
# The spacecraft moves about the Moon (the central body), perturbed by the Earth and the Sun, by an 8×8 lunar gravity field (GRAIL model `gggrx_1200l`, evaluated in the `MOON_PA` frame), and by cannonball solar radiation pressure. Ephemerides and the `J2000` to `MOON_PA` rotation are evaluated with the Julia-native Ephemerides.jl backend, which is thread-safe.

naif_ids = ["301", "399", "10"]             # Moon (central body), Earth, Sun
GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
DU = 1e5                                    # canonical distance unit, km
et0 = str2et("2027-01-07T15:30:59.539")     # epoch of the apolune passage (UTC)

parameters = EphemeridesParameters(
    et0, DU, GMs, naif_ids, "J2000", "NONE";
    filepath_spherical_harmonics = joinpath(
        pkgdir(HighFidelityEphemerisModel), "data", "luna", "gggrx_1200l_sha_20x20.tab",
    ),
    nmax = 8,                               # 8x8 lunar gravity field
    frame_PCPF = "MOON_PA",
    include_srp = true,
    srp_Cr = 1.2,
    srp_Am = 0.005,                         # m^2/kg
    ephemerides_files = [
        joinpath(spice_dir, "spk", "de440.bsp"),
        joinpath(spice_dir, "pck", "moon_pa_de440_200625.bpc"),
    ],
)
TU, VU = parameters.TU, parameters.VU
println("DU = $DU km, TU = $(round(TU, digits=1)) s, VU = $(round(VU, digits=6)) km/s")

# !!! note
#
#     `EnsembleThreads()` evaluates the equations of motion from several threads at once, so the parameters must be thread-safe. `EphemeridesParameters` and `InterpParameters` are, as long as `f_density` is (the default Harris–Priester callback is, see [Perturbations](../tutorials/perturbations.md)). `SpiceParameters` are not, because CSPICE is not thread-safe: with them, use `EnsembleSerial()`.

# ## Initial state at apolune
#
# The initial state is the apolune state on 2027-01-07 of the Gateway reference trajectory [`receding_horiz_3189_1burnApo_DiffCorr_15yr.bsp`](https://naif.jpl.nasa.gov/pub/naif/misc/MORE_PROJECTS/DSG/) (NAIF ID `-60000`). It is hard-coded below so that the example does not need this kernel.
#
# All trajectories are reported on a common time grid spanning four revolutions.

## Moon-centered J2000 state at apolune, km and km/s
x0_km = [7515.6206033898, 20208.1270104193, -68140.4636066553,
         -0.0736747576144, -0.0151307748977, -0.0126132987093]
x0 = [x0_km[1:3] / DU; x0_km[4:6] / VU]

R_moon = 1737.4                             # mean lunar radius, km
period = 6.56 * 86400 / TU                  # approximate NRHO period (6.56 days)
N_rev = 4
tspan = (0.0, N_rev * period)
t_grid = range(tspan..., length = 801)      # common output grid for all trajectories
t_days = t_grid * TU / 86400;

# ## Maneuver and execution-error model
#
# The nominal maneuver is a 0.5 m/s burn along the velocity at apolune. Each executed maneuver deviates from it by a magnitude error with a fixed part $\delta_{\mathrm{fix}}$ and a part $\delta_{\mathrm{prop}}$ proportional to the magnitude, and by a pointing error $(\epsilon_1, \epsilon_2)$ about two axes orthogonal to the nominal direction $\hat{\boldsymbol{u}}$:
#
# ```math
# \Delta \boldsymbol{v} = \left( \Delta v_{\mathrm{nom}} (1 + \delta_{\mathrm{prop}}) + \delta_{\mathrm{fix}} \right)
# \dfrac{\hat{\boldsymbol{u}} + \epsilon_1 \hat{\boldsymbol{e}}_1 + \epsilon_2 \hat{\boldsymbol{e}}_2}{\| \hat{\boldsymbol{u}} + \epsilon_1 \hat{\boldsymbol{e}}_1 + \epsilon_2 \hat{\boldsymbol{e}}_2 \|}
# ```
#
# where all four errors are independent, zero-mean Gaussian variables. The 1σ values below are illustrative.

Δv_nom = 0.5e-3                             # nominal maneuver magnitude, km/s (0.5 m/s)
u_nom = normalize(x0_km[4:6])               # nominal direction: along the velocity
e1 = normalize(cross(u_nom, x0_km[1:3]))    # two unit vectors orthogonal to u_nom
e2 = cross(u_nom, e1)

σ_fix   = 1e-6                              # fixed magnitude error, km/s (1 mm/s), 1σ
σ_prop  = 0.01                              # proportional magnitude error (1 %), 1σ
σ_point = deg2rad(1.0)                      # pointing error per axis, 1σ

"Sample one executed maneuver, in km/s."
function sample_maneuver(rng)
    magnitude = Δv_nom * (1 + σ_prop * randn(rng)) + σ_fix * randn(rng)
    direction = normalize(u_nom + σ_point * (randn(rng) * e1 + randn(rng) * e2))
    return magnitude * direction
end

N_mc = 1000
rng = Xoshiro(2027)
Δv_samples = [sample_maneuver(rng) for _ in 1:N_mc]

## post-maneuver state of the nominal trajectory
x0_nom = [x0[1:3]; x0[4:6] + Δv_nom * u_nom / VU];

# ## Nominal trajectory
#
# The perilune passages are found as the zeros of $\boldsymbol{r} \cdot \boldsymbol{v}$ (going from negative to positive) on the continuous solution. We first apply this to the nominal post-maneuver trajectory.

"""
Perilune passages of the trajectory `sol`, as a vector of `(time, radius)` (canonical
units). Perilunes are the zeros of r⋅v at which r⋅v changes from negative to positive.
"""
function perilune_passages(sol; n_scan = 2000)
    rdotv(t) = (x = sol(t); dot(x[1:3], x[4:6]))
    ts = range(sol.t[1], sol.t[end]; length = n_scan)
    passages = Tuple{Float64,Float64}[]
    for k in 1:n_scan-1
        if rdotv(ts[k]) < 0 <= rdotv(ts[k+1])
            a, b = ts[k], ts[k+1]
            for _ in 1:50                   # bisection
                m = (a + b) / 2
                rdotv(m) < 0 ? (a = m) : (b = m)
            end
            t_p = (a + b) / 2
            push!(passages, (t_p, norm(sol(t_p)[1:3])))
        end
    end
    return passages
end
nothing #hide

#-

prob = ODEProblem(eom_NbodySH!, x0_nom, tspan, parameters)
sol_nom = solve(prob, Vern9(); reltol = 1e-12, abstol = 1e-12)
X_nom = Array(sol_nom(t_grid))              # 6 × length(t_grid)

perilunes_nom = perilune_passages(sol_nom)
for (k, (t_p, r_p)) in enumerate(perilunes_nom)
    @printf("perilune %d: t = %5.2f days, altitude = %6.1f km\n", k, t_p * TU / 86400, r_p * DU - R_moon)
end

# ## Monte Carlo with an ensemble
#
# An `EnsembleProblem` wraps the nominal `ODEProblem` together with
#
# - `prob_func(prob, i, repeat)`, which returns the problem for the `i`-th trajectory, here with the `i`-th sampled maneuver added to the velocity, and
# - `output_func(sol, i)`, which reduces each solution to what we need (the states on `t_grid` and the perilune passages) before the next trajectory is solved, so that the full solutions do not have to be kept in memory. Its second return value tells the solver whether to rerun the trajectory.
#
# `solve` then distributes the trajectories over the threads with `EnsembleThreads()`.

# !!! warning
#
#     When a `prob_func` is given, `EnsembleProblem` by default deep-copies the problem, including its parameters, for every trajectory (`safetycopy = true`). For `EphemeridesParameters` this copies the loaded ephemeris data each time, about 130 MB per trajectory for DE440 in this example, which makes the ensemble many times slower. The equations of motion do not modify the parameters, so pass `safetycopy = false`.

## i-th trajectory: apply the i-th sampled maneuver to the pre-maneuver state
prob_func(prob, i, repeat) = remake(prob; u0 = [x0[1:3]; x0[4:6] + Δv_samples[i] / VU])

## reduce each solution to what we need: states on `t_grid` and perilune passages
output_func(sol, i) = ((u = Array(sol(t_grid)), perilunes = perilune_passages(sol), retcode = sol.retcode), false)

ensemble = EnsembleProblem(prob; prob_func, output_func, safetycopy = false)
@time results = solve(ensemble, Vern9(), EnsembleThreads();
    trajectories = N_mc, reltol = 1e-12, abstol = 1e-12)
println("all successful: ", all(r -> SciMLBase.successful_retcode(r.retcode), results.u))

# ## Linear covariance with the STM
#
# To first order, a maneuver error $\delta \Delta \boldsymbol{v}$ maps to the state deviation $\delta \boldsymbol{x}(t) = \boldsymbol{\Phi}_v(t) \, \delta \Delta \boldsymbol{v}$, where $\boldsymbol{\Phi}_v$ holds the velocity columns of the STM $\boldsymbol{\Phi}(t, t_0)$ of the nominal post-maneuver trajectory. The covariance of the executed maneuver,
#
# ```math
# \boldsymbol{P}_{\Delta v} = \left( \sigma_{\mathrm{fix}}^2 + \sigma_{\mathrm{prop}}^2 \Delta v_{\mathrm{nom}}^2 \right) \hat{\boldsymbol{u}} \hat{\boldsymbol{u}}^{\top}
# + \Delta v_{\mathrm{nom}}^2 \sigma_{\mathrm{point}}^2 \left( \hat{\boldsymbol{e}}_1 \hat{\boldsymbol{e}}_1^{\top} + \hat{\boldsymbol{e}}_2 \hat{\boldsymbol{e}}_2^{\top} \right),
# ```
#
# then maps to $\boldsymbol{P}(t) = \boldsymbol{\Phi}_v(t) \boldsymbol{P}_{\Delta v} \boldsymbol{\Phi}_v(t)^{\top}$. The STM is propagated once along the nominal trajectory with `eom_stm_NbodySH_Ephemerides_fd!`, the Ephemerides.jl counterpart of the SPICE-backed STM propagation shown in [Basics](../tutorials/basics.md).

x0_stm = [x0_nom; vec(Matrix(1.0I, 6, 6))]
prob_stm = ODEProblem(eom_stm_NbodySH_Ephemerides_fd!, x0_stm, tspan, parameters)
sol_stm = solve(prob_stm, Vern9(); reltol = 1e-12, abstol = 1e-12, saveat = t_grid)

## covariance of the executed maneuver, (km/s)^2
P_Δv = (σ_fix^2 + (σ_prop * Δv_nom)^2) * u_nom * u_nom' +
       (Δv_nom * σ_point)^2 * (e1 * e1' + e2 * e2')

## map it with the velocity columns of the STM: P(t) = Φ_v(t) P_Δv Φ_v(t)ᵀ, in km and km/s
D = Diagonal([fill(DU, 3); fill(VU, 3)])    # canonical -> km, km/s
P_lin = map(sol_stm.u) do x
    Φ = reshape(x[7:42], 6, 6)
    Φ_v = D * Φ[:, 4:6] / VU                # sensitivity of [km, km/s] to Δv in km/s
    Symmetric(Φ_v * P_Δv * Φ_v')
end;

# ## Results
#
# We compare the root-mean-square (RMS) position and velocity deviations from the nominal trajectory over the Monte Carlo samples with their linear prediction $\sqrt{\operatorname{tr} \boldsymbol{P}_{rr}(t)}$ and $\sqrt{\operatorname{tr} \boldsymbol{P}_{vv}(t)}$:

δX = [D * (r.u - X_nom) for r in results.u]          # deviations from the nominal, km and km/s
δr = [norm(δ[1:3, k]) for δ in δX, k in eachindex(t_grid)]          # km, N_mc × N_t
δv = [1e3 * norm(δ[4:6, k]) for δ in δX, k in eachindex(t_grid)]    # m/s

rms(x) = sqrt(mean(abs2, x))
δr_rms = rms.(eachcol(δr))
δv_rms = rms.(eachcol(δv))
δr_lin = [sqrt(tr(P[1:3, 1:3])) for P in P_lin]      # E[|δr|²]^(1/2) of the linear model
δv_lin = [1e3 * sqrt(tr(P[4:6, 4:6])) for P in P_lin]

for days in (1, 6.56, 13.12, 19.68, 26.24)
    k = argmin(abs.(t_days .- days))
    @printf("t = %5.2f d: RMS |δr| MC = %8.2f km, linear = %8.2f km | RMS |δv| MC = %7.3f m/s, linear = %7.3f m/s\n",
        t_days[k], δr_rms[k], δr_lin[k], δv_rms[k], δv_lin[k])
end

## sampling error: covariance of the 1000 drawn maneuver errors vs. the model covariance P_Δv
δΔv = reduce(hcat, Δv_samples) .- Δv_nom * u_nom
@printf("\ntr(sample covariance of the maneuver errors) / tr(P_Δv) = %.3f\n", tr(δΔv * δΔv' / N_mc) / tr(P_Δv))

# ### Trajectories
#
# For plotting, the positions are rotated into the Moon-centered Earth–Moon rotating frame, with $x$ along the Earth–Moon line (pointing away from the Earth) and $z$ along the orbital angular momentum of the Moon. The figures below are interactive: drag to rotate or zoom, hover for values, and click on legend entries to hide traces.

"Rotation from J2000 to the Earth–Moon rotating frame (x: Earth → Moon, z: orbit normal of the Moon)."
function C_J2000_to_EM(et)
    s, _ = spkezr("301", et, "J2000", "NONE", "399")   # Moon w.r.t. Earth
    x̂ = normalize(s[1:3])
    ẑ = normalize(cross(s[1:3], s[4:6]))
    return [x̂ cross(ẑ, x̂) ẑ]'
end
C_grid = [C_J2000_to_EM(et0 + t * TU) for t in t_grid]

## Moon-centered positions in the rotating frame, km (3 × length(t_grid))
to_rotating(X) = reduce(hcat, [DU * (C * X[1:3, k]) for (k, C) in enumerate(C_grid)]);

#-

colors = (nominal = "#0b0b0b", samples = "rgba(42, 120, 214, 0.35)", linear = "#eb6834")
r1(x) = round.(x; digits = 1)               # keep the data embedded in the figure small

traces = GenericTrace[]
for (i, res) in enumerate(results.u[1:15])  # a subset of the samples
    R = to_rotating(res.u)[:, 1:2:end]
    push!(traces, scatter3d(x = r1(R[1, :]), y = r1(R[2, :]), z = r1(R[3, :]), mode = "lines",
        line = attr(color = colors.samples, width = 2), name = "Monte Carlo samples",
        legendgroup = "mc", showlegend = i == 1, hoverinfo = "skip"))
end
R_nom = to_rotating(X_nom)
push!(traces, scatter3d(x = r1(R_nom[1, :]), y = r1(R_nom[2, :]), z = r1(R_nom[3, :]),
    mode = "lines", line = attr(color = colors.nominal, width = 3), name = "nominal",
    hovertemplate = "x = %{x:.0f} km<br>y = %{y:.0f} km<br>z = %{z:.0f} km<extra>nominal</extra>"))
push!(traces, scatter3d(x = [R_nom[1,1]], y = [R_nom[2,1]], z = [R_nom[3,1]], mode = "markers",
    marker = attr(color = colors.linear, size = 6), name = "maneuver at apolune"))

## the Moon (surface coordinates given as vectors of rows, which IJulia can serialize)
θs, φs = range(0, π, length = 16), range(0, 2π, length = 32)
push!(traces, surface(
    x = [r1([R_moon * sin(θ) * cos(φ) for φ in φs]) for θ in θs],
    y = [r1([R_moon * sin(θ) * sin(φ) for φ in φs]) for θ in θs],
    z = [r1([R_moon * cos(θ) for φ in φs]) for θ in θs],
    colorscale = [[0, "#b9b8b3"], [1, "#b9b8b3"]], showscale = false, name = "Moon", hoverinfo = "skip"))

fig = Plot(traces, Layout(
    title = "NRHO in the Moon-centered Earth–Moon rotating frame",
    scene = attr(aspectmode = "data",
        xaxis_title = "x, km", yaxis_title = "y, km", zaxis_title = "z, km"),
    height = 600, margin = attr(l = 0, r = 0, b = 0, t = 40), template = templates.plotly_white,
))
fig
#md Main.docs_figure(fig) #hide

# At this scale the dispersed trajectories lie on top of the nominal one; zoom in on the last revolution to see them separate.
#
# ### Growth of the dispersions

r3(x) = round.(x; sigdigits = 3)
fig = Plot(Layout(Subplots(rows = 2, cols = 1, shared_xaxes = true, vertical_spacing = 0.06);
    height = 650, template = templates.plotly_white, hovermode = "x unified",
    title = "Growth of the dispersions after the maneuver",
    xaxis2_title = "time since maneuver, days",
    yaxis_title = "|δr|, km", yaxis_type = "log",
    yaxis2_title = "|δv|, m/s", yaxis2_type = "log"))

for (row, samples, rms_mc, rms_lin) in ((1, δr, δr_rms, δr_lin), (2, δv, δv_rms, δv_lin))
    for i in 1:15                           # a subset of the samples, every other grid point
        add_trace!(fig, scatter(x = r3(t_days[1:2:end]), y = r3(samples[i, 1:2:end]), mode = "lines",
            line = attr(color = colors.samples, width = 1), name = "samples",
            legendgroup = "samples", showlegend = row == 1 && i == 1, hoverinfo = "skip"); row)
    end
    add_trace!(fig, scatter(x = r3(t_days), y = r3(rms_mc), mode = "lines",
        line = attr(color = colors.nominal, width = 2), name = "RMS, Monte Carlo",
        legendgroup = "mc", showlegend = row == 1); row)
    add_trace!(fig, scatter(x = r3(t_days), y = r3(rms_lin), mode = "lines",
        line = attr(color = colors.linear, width = 2, dash = "dash"), name = "RMS, linear covariance",
        legendgroup = "lin", showlegend = row == 1); row)
end
for (t_p, _) in perilunes_nom                 # mark the perilune passages
    add_vline!(fig, t_p * TU / 86400; line = attr(color = "#b9b8b3", dash = "dot", width = 1))
end
fig
#md Main.docs_figure(fig) #hide

# The dotted lines mark the nominal perilune passages. The dispersions grow with each revolution, and spike at every perilune: the spacecraft moves fastest there, so a small timing difference between a sample and the nominal trajectory becomes a large position and velocity difference at the same epoch. The RMS of the linear covariance follows the Monte Carlo RMS to within a few percent over the four revolutions, so for maneuver errors of this size the linear model is adequate. The remaining difference is not caused by nonlinearity: it is already present after one day, and it matches the sampling error of the 1000 drawn maneuvers, whose sample covariance is a few percent larger than $\boldsymbol{P}_{\Delta v}$ (printed above).
#
# ### Perilune passages
#
# Comparing each sample with the nominal trajectory at its *own* perilune passage instead shows the timing and altitude dispersions:

n_peri = length(perilunes_nom)
fig = Plot(Layout(Subplots(rows = 1, cols = n_peri, shared_yaxes = false, horizontal_spacing = 0.06,
        subplot_titles = reshape(["perilune $k" for k in 1:n_peri], 1, :));
    height = 400, template = templates.plotly_white, showlegend = false, margin = attr(t = 90),
    title = "Perilune passages: timing and altitude"))
for k in 1:n_peri
    t_p_nom, r_p_nom = perilunes_nom[k]
    ## deviation of the k-th perilune passage of each sample, in minutes and km
    Δt = [60 * 24 * (r.perilunes[k][1] - t_p_nom) * TU / 86400 for r in results.u]
    Δh = [(r.perilunes[k][2] - r_p_nom) * DU for r in results.u]
    add_trace!(fig, scatter(x = r3(Δt), y = r3(Δh), mode = "markers",
        marker = attr(color = colors.samples, size = 5),
        hovertemplate = "Δt = %{x:.2f} min<br>Δh = %{y:.2f} km<extra></extra>"); col = k)
    @printf("perilune %d: 1σ timing = %6.2f min, 1σ altitude = %6.2f km\n", k, std(Δt), std(Δh))
end
for k in 1:n_peri
    fig.layout["xaxis$(k == 1 ? "" : k)_title"] = "Δt, min"
end
fig.layout["yaxis_title"] = "Δ altitude, km"
fig
#md Main.docs_figure(fig) #hide

# The timing dispersion grows with each passage. The altitude dispersion grows from the second passage on, and at the third and fourth passages the timing and altitude deviations are strongly correlated: samples that reach perilune late also pass higher.
#
# ### Dispersion after four revolutions
#
# Finally, we compare the position dispersion at the final time with the linear covariance, in the rotating frame:

## final position deviations, in the rotating frame (km)
C_f = C_grid[end]
δr_f = reduce(hcat, [C_f * δ[1:3, end] for δ in δX])   # 3 × N_mc
P_f = Symmetric(C_f * P_lin[end][1:3, 1:3] * C_f')      # linear covariance, rotating frame

## fraction of samples inside the 3σ ellipsoid (97.1 % for a 3-D Gaussian)
m2 = [δ' * (P_f \ δ) for δ in eachcol(δr_f)]
@printf("inside the linear 3σ ellipsoid: %.1f %%\n", 100 * count(<(9), m2) / N_mc)
for (j, ax) in enumerate(("x", "y", "z"))
    @printf("σ_%s: Monte Carlo = %6.2f km, linear = %6.2f km\n", ax, std(δr_f[j, :]), sqrt(P_f[j, j]))
end

#-

"Points on the n-σ ellipse of the 2×2 covariance `P`."
function ellipse(P; n = 3, npts = 100)
    λ, V = eigen(Symmetric(P))
    return reduce(hcat, [n * V * (sqrt.(λ) .* [cos(a), sin(a)]) for a in range(0, 2π, length = npts)])
end

pairs = ((1, 2), (1, 3), (2, 3))
labels = ("δx, km", "δy, km", "δz, km")
fig = Plot(Layout(Subplots(rows = 1, cols = 3, horizontal_spacing = 0.08);
    height = 420, template = templates.plotly_white,
    title = "Position dispersion after $(N_rev) revolutions (rotating frame)"))
for (k, (a, b)) in enumerate(pairs)
    add_trace!(fig, scatter(x = r3(δr_f[a, :]), y = r3(δr_f[b, :]), mode = "markers",
        marker = attr(color = colors.samples, size = 5), name = "Monte Carlo samples",
        legendgroup = "mc", showlegend = k == 1); col = k)
    E = ellipse(P_f[[a, b], [a, b]])
    add_trace!(fig, scatter(x = r3(E[1, :]), y = r3(E[2, :]), mode = "lines",
        line = attr(color = colors.linear, width = 2), name = "3σ, linear covariance",
        legendgroup = "lin", showlegend = k == 1, hoverinfo = "skip"); col = k)
    fig.layout["xaxis$(k == 1 ? "" : k)_title"] = labels[a]
    fig.layout["yaxis$(k == 1 ? "" : k)_title"] = labels[b]
end
fig
#md Main.docs_figure(fig) #hide

# The Monte Carlo cloud is an elongated ellipsoid, mostly along the Earth–Moon line and the $y$ axis, that matches the shape of the linear 3σ ellipses. The fraction of samples inside the linear 3σ ellipsoid is close to the 97.1 % expected for a three-dimensional Gaussian distribution; as above, the small differences between the Monte Carlo and linear standard deviations are mostly sampling error.
#
# ## Going further
#
# - Add navigation errors by also perturbing the position and velocity before the maneuver in `prob_func`.
# - Replace `output_func` with one that also stores e.g. the state at a later maneuver epoch, to size the next correction maneuver.
# - With `InterpParameters` instead of `EphemeridesParameters`, the same ensemble runs on spline-interpolated ephemerides; with `SpiceParameters`, use `EnsembleSerial()`.
