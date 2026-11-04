"""
Hero figure summarizing the capabilities of HighFidelityEphemerisModel.jl.

Panels
  (a) Monte Carlo ensemble about an Earth-Moon 9:2 NRHO (`EnsembleThreads` + Ephemerides.jl
      backend), with perilune events detected via `get_trueanomaly_event`
  (b) acceleration budget along the NRHO (point mass, third bodies, spherical harmonics, SRP)
  (c) linear covariance propagated with the ForwardDiff-based STM vs. Monte Carlo dispersions
  (d) lunar 50x50 spherical-harmonics gravity field (GRGM1200L), with the NRHO perilune
      locations from (a)
  (e) atmospheric density models (Harris-Priester, Jacchia-Roberts)
  (f) LEO orbit decay under atmospheric drag, terminated by a callback at 120 km
  (g) agreement between ephemeris backends (SPICE / Ephemerides.jl / interpolation) and
      the SPK type-13 export round trip

Run from the repository root with threads enabled, e.g.

    julia --project=demos -t auto demos/demo_hero_figure.jl

(add `--startup-file=no` if your startup file activates a different project). Takes ~2 min
on 24 threads.

Requires SPICE kernels under `ENV["SPICE"]` (as in the other demos), including the NRHO
reference trajectory `misc/dsg_naif/receding_horiz_3189_1burnApo_DiffCorr_15yr.bsp` and the
Earth-Moon rotating frame kernel `fk/earth_moon_rotating_mc.tf`.

Outputs `demos/plots/hero_figure.pdf` (vector, 7 in wide, for the paper) and
`demos/plots/hero_figure.png`.
"""

using CairoMakie
using LinearAlgebra
using OrdinaryDiffEq
using Printf
using Random
using SPICE

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end
using .HighFidelityEphemerisModel
const HFEM = HighFidelityEphemerisModel

include(joinpath(@__DIR__, "..", "test", "utils.jl"))
furnsh_kernels()
spice_dir = ENV["SPICE"]
furnsh(joinpath(spice_dir, "fk", "earth_moon_rotating_mc.tf"))

@printf("Running with %d thread(s)\n", Threads.nthreads())

# ========================================================================================== #
# Common NRHO setup (Moon-centered, Earth & Sun third bodies, lunar SH, SRP)
# ========================================================================================== #
const NAIF_FRAME = "J2000"
const ABCORR = "NONE"
const ROT_FRAME = "EARTHMOONROTATINGMC"     # Moon-centered Earth-Moon rotating frame
const NMAX = 9                              # lunar SH degree & order used for propagation
const SH_MOON = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab")
const NMAX_MAP = 50                         # lunar SH degree & order of the gravity map (d)
const SH_MOON_MAP = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_50x50.tab")
const R_MOON = 1737.4                       # km, for plotting only

naif_ids = ["301", "399", "10"]
GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
DU = 1e5
et0 = str2et("2028-01-03T08:00:00")
period_days = 6.55
N_rev = 10
horizon_sec = N_rev * period_days * 86400.0

nrho_kwargs = (
    filepath_spherical_harmonics = SH_MOON,
    nmax = NMAX,
    frame_PCPF = "MOON_PA",
    include_srp = true,
    srp_Cr = 1.15,
    srp_Am = 0.002,
)

p_spice = SpiceParameters(et0, DU, GMs, naif_ids, NAIF_FRAME, ABCORR; nrho_kwargs...)
p_ephem = EphemeridesParameters(et0, DU, GMs, naif_ids, NAIF_FRAME, ABCORR; nrho_kwargs...,
    ephemerides_files = [joinpath(spice_dir, "spk", "de440.bsp"),
                         joinpath(spice_dir, "pck", "moon_pa_de440_200625.bpc")])
interp_span = [et0 - 86400.0, et0 + horizon_sec + 86400.0]
p_interp_fine = InterpParameters(et0, DU, GMs, naif_ids, NAIF_FRAME, ABCORR; nrho_kwargs...,
    interpolate_ephem_span = interp_span, interpolation_time_step = 600.0)
p_interp_coarse = InterpParameters(et0, DU, GMs, naif_ids, NAIF_FRAME, ABCORR; nrho_kwargs...,
    interpolate_ephem_span = interp_span, interpolation_time_step = 3600.0)

TU, VU = p_spice.TU, p_spice.VU
x0_km = spkezr("-60000", et0, NAIF_FRAME, ABCORR, "301")[1]
x0 = [x0_km[1:3] / DU; x0_km[4:6] / VU]
tspan = (0.0, horizon_sec / TU)
tol = (reltol = 1e-13, abstol = 1e-13)

# ========================================================================================== #
# Nominal NRHO with perilune / apolune event detection
# ========================================================================================== #
t_peri = Float64[]
t_apo = Float64[]
cb_peri = ContinuousCallback(get_trueanomaly_event(0.0), integ -> push!(t_peri, integ.t);
    save_positions = (false, false))
cb_apo = ContinuousCallback(get_trueanomaly_event(π), integ -> push!(t_apo, integ.t);
    save_positions = (false, false))
sol_nom = solve(ODEProblem(eom_NbodySH!, x0, tspan, p_ephem), Vern8(); tol...,
    callback = CallbackSet(cb_peri, cb_apo))
@printf("Nominal: %d perilune / %d apolune events\n", length(t_peri), length(t_apo))

# common plotting grid & rotation into the Earth-Moon rotating frame
ts = range(tspan..., length = 4000)
R_rot = [pxform(NAIF_FRAME, ROT_FRAME, et0 + t * TU) for t in ts]
to_rot(sol) = hcat([R * (sol(t)[1:3] * DU) for (R, t) in zip(R_rot, ts)]...)
r_nom_rot = to_rot(sol_nom)

# epoch at which Monte Carlo dispersions are compared against the STM (third apolune)
t_cmp = t_apo[3]
R_cmp = pxform(NAIF_FRAME, ROT_FRAME, et0 + t_cmp * TU)

# ========================================================================================== #
# (a, c) Monte Carlo ensemble (EnsembleThreads) & STM-based linear covariance
# ========================================================================================== #
N_mc = 200
σr_km, σv_kms = 1.0, 1e-4                  # 1 km, 10 cm/s (1σ, per axis)
Random.seed!(2026)
δx0s = [[σr_km / DU * randn(3); σv_kms / VU * randn(3)] for _ in 1:N_mc]

ens = EnsembleProblem(
    ODEProblem(eom_NbodySH!, x0, tspan, p_ephem);
    prob_func = (prob, i, repeat) -> remake(prob; u0 = x0 + δx0s[i]),
    output_func = (sol, i) -> ((to_rot(sol), R_cmp * (sol(t_cmp)[1:3] - sol_nom(t_cmp)[1:3]) * DU), false),
)
t_ens = @elapsed sols_mc = solve(ens, Vern8(), EnsembleThreads(); trajectories = N_mc,
    reltol = 1e-12, abstol = 1e-12)
@printf("Ensemble of %d trajectories: %.1f s\n", N_mc, t_ens)
δr_mc = hcat([s[2] for s in sols_mc.u]...)                       # 3 x N_mc, km (rotating axes)

x0_stm = [x0; reshape(Matrix(1.0I, 6, 6), 36)]
sol_stm = solve(ODEProblem(eom_stm_NbodySH_Ephemerides_fd!, x0_stm, (0.0, t_cmp), p_ephem),
    Vern8(); reltol = 1e-12, abstol = 1e-12)
Φ = reshape(sol_stm.u[end][7:42], 6, 6)
P0 = Diagonal([fill((σr_km / DU)^2, 3); fill((σv_kms / VU)^2, 3)])
P_rr = R_cmp * (Φ * P0 * Φ')[1:3, 1:3] * R_cmp' * DU^2             # km^2, rotating axes

# ========================================================================================== #
# (b) acceleration budget along the nominal NRHO (first two revolutions)
# ========================================================================================== #
ACC = DU / TU^2 * 1e3                      # canonical acceleration -> m/s^2
ts_acc = range(0.0, 2 * period_days * 86400.0 / TU, length = 1500)
acc = Dict(k => Float64[] for k in (:moon, :earth, :sun, :sh, :srp))
shd = p_spice.spherical_harmonics_data
for t in ts_acc
    et = et0 + t * TU
    r = sol_nom(t)[1:3]
    r_earth = spkpos("399", et, NAIF_FRAME, ABCORR, "301")[1] / DU
    r_sun = spkpos("10", et, NAIF_FRAME, ABCORR, "301")[1] / DU
    T_pa = pxform(NAIF_FRAME, "MOON_PA", et)
    push!(acc[:moon], p_spice.mus[1] / norm(r)^2 * ACC)
    push!(acc[:earth], norm(HFEM.third_body_accel(r, r_earth, p_spice.mus[2])) * ACC)
    push!(acc[:sun], norm(HFEM.third_body_accel(r, r_sun, p_spice.mus[3])) * ACC)
    push!(acc[:srp], norm(HFEM.srp_cannonball(r, r_sun, p_spice.k_srp_cannonball)) * ACC)
    a_sh = HFEM.spherical_harmonics_accel(T_pa, r * DU, shd["Cnm"], shd["Snm"], shd["GM"],
        shd["REFERENCE RADIUS"], NMAX, p_spice.factorial_alias)        # km/s^2
    push!(acc[:sh], norm(a_sh) * 1e3)
end

# ========================================================================================== #
# (d) lunar gravity field: gravity disturbance δg = -a_r of the SH field at the reference radius
#     (positive where gravity is stronger than for a point mass, e.g. over mascons)
# ========================================================================================== #
# evaluated with a higher-degree field than the one used for propagation
shd_map = HFEM.load_spherical_harmonics(SH_MOON_MAP, NMAX_MAP, true, HFEM.factorial_safe)
lons = range(-180, 180, length = 361)      # 1 deg grid resolves the ~7 deg degree-50 wavelength
lats = range(-90, 90, length = 181)
R_ref = shd_map["REFERENCE RADIUS"]
g_rad = zeros(length(lons), length(lats))
t_map = @elapsed Threads.@threads for idx in CartesianIndices(g_rad)
    λ, φ = lons[idx[1]], lats[idx[2]]
    r̂ = [cosd(φ) * cosd(λ), cosd(φ) * sind(λ), sind(φ)]
    a = HFEM.spherical_harmonics_accel_PCPF(R_ref * r̂, shd_map["Cnm"], shd_map["Snm"],
        shd_map["GM"], R_ref, NMAX_MAP)
    g_rad[idx] = -dot(a, r̂) * 1e8          # km/s^2 -> mGal; sign: positive = stronger downward pull
end
@printf("Gravity map (%d×%d field, %d points): %.1f s\n", NMAX_MAP, NMAX_MAP, length(g_rad), t_map)
peri_lonlat = map(t_peri) do t
    r_pa = pxform(NAIF_FRAME, "MOON_PA", et0 + t * TU) * sol_nom(t)[1:3]
    λ, φ, _ = HFEM.cart2sph(r_pa)
    (rad2deg(λ), rad2deg(φ))
end

# ========================================================================================== #
# (e) atmospheric density models & (f) LEO decay under drag
# ========================================================================================== #
R_EARTH = 6378.137
h_stop = 120.0
et0_leo = str2et("2026-01-05T00:00:00")
density_models = [
    ("Harris–Priester (min)", HFEM.harris_priester_f_density(R_EARTH; use_min = true)),
    ("Harris–Priester (max)", HFEM.harris_priester_f_density(R_EARTH; use_min = false)),
    ("Jacchia–Roberts", HFEM.jacchia_roberts_f_density(frame_PCPF = "IAU_EARTH")),
]

# density vs altitude above the equator at 0 deg longitude (body-fixed position, km)
alts = 120.0:2.0:1000.0
profiles = map(density_models) do (name, f_density)
    [f_density(et0_leo, [R_EARTH + h, 0.0, 0.0]) for h in alts]
end

earth_ids = ["399", "301", "10"]
GMs_earth = [bodvrd(ID, "GM", 1)[1] for ID in earth_ids]
decay = map(density_models) do (name, f_density)
    p = SpiceParameters(et0_leo, R_EARTH, GMs_earth, earth_ids, NAIF_FRAME, ABCORR;
        filepath_spherical_harmonics = joinpath(@__DIR__, "../data/earth/GGM03S.tab"),
        nmax = 4, frame_PCPF = "IAU_EARTH",
        include_drag = true, drag_Cd = 2.2, drag_Am = 0.02, f_density = f_density)
    a = (R_EARTH + 250.0) / p.DU
    inc = deg2rad(51.6)
    v = sqrt(p.mus[1] / a)
    x0_leo = [a, 0.0, 0.0, 0.0, v * cos(inc), v * sin(inc)]
    cb = ContinuousCallback((u, t, integ) -> norm(u[1:3]) * p.DU - (R_EARTH + h_stop), terminate!)
    t_run = @elapsed sol = solve(ODEProblem(eom_NbodySH!, x0_leo, (0.0, 20 * 86400 / p.TU), p),
        Vern7(); reltol = 1e-10, abstol = 1e-10, callback = cb)
    days = sol.t * p.TU / 86400
    @printf("Decay (%s): reaches %.0f km after %.2f days (%.1f s)\n", name, h_stop, days[end], t_run)
    (days, [norm(u[1:3]) * p.DU - R_EARTH for u in sol.u])
end

# ========================================================================================== #
# (g) backend agreement & SPK export round trip (reference: SPICE backend)
# ========================================================================================== #
backend_sols = Dict{String,Any}()
for (name, p) in [("SPICE", p_spice), ("Ephemerides.jl", p_ephem),
                  ("Interp. 10 min", p_interp_fine), ("Interp. 1 h", p_interp_coarse)]
    t_run = @elapsed backend_sols[name] = solve(ODEProblem(eom_NbodySH!, x0, tspan, p), Vern8(); tol...)
    @printf("Backend %-16s: %6.2f s\n", name, t_run)
end
sol_ref = backend_sols["SPICE"]
ts_cmp = range(tspan[1], tspan[2], length = 3000)[2:end-1]
Δr(sol) = [norm(sol(t)[1:3] - sol_ref(t)[1:3]) * DU * 1e3 for t in ts_cmp]   # m

spk_dir = mktempdir()
spk_id = -999001
res_spk = ode_sol_to_spk([sol_ref], et0, p_spice;
    output_spk = joinpath(spk_dir, "nrho.bsp"), spice_id = spk_id, center_id = 301,
    ref_frame_name = NAIF_FRAME, dt_sec = 300.0, write_metadata = false)
furnsh(res_spk.output_spk)
Δr_spk = [norm(spkpos(string(spk_id), et0 + t * TU, NAIF_FRAME, ABCORR, "301")[1] -
               sol_ref(t)[1:3] * DU) * 1e3 for t in ts_cmp]
unload(res_spk.output_spk)

errors_g = [
    ("Ephemerides.jl", accumulate(max, Δr(backend_sols["Ephemerides.jl"]))),
    ("Interp., 10 min nodes", accumulate(max, Δr(backend_sols["Interp. 10 min"]))),
    ("Interp., 1 h nodes", accumulate(max, Δr(backend_sols["Interp. 1 h"]))),
    ("SPK type 13 export", accumulate(max, Δr_spk)),
]
for (name, e) in errors_g
    @printf("  max |Δr| %-22s: %.3e m\n", name, e[end])
end

# ========================================================================================== #
# Plot
# ========================================================================================== #
# categorical slots (fixed order) + ink
C1, C2, C3, C4 = colorant"#2a78d6", colorant"#eb6834", colorant"#1baf7a", colorant"#eda100"
INK, INK2, GRID = colorant"#0b0b0b", colorant"#52514e", colorant"#e4e3df"
LBL = 6                                    # direct-label / legend font size

set_theme!(merge(theme_latexfonts(), Theme(
    fontsize = 7,
    Axis = (xgridcolor = GRID, ygridcolor = GRID, xgridwidth = 0.4, ygridwidth = 0.4,
            spinewidth = 0.5, xtickwidth = 0.5, ytickwidth = 0.5,
            xticksize = 2, yticksize = 2, titlesize = 7, titlefont = :bold,
            titlealign = :left, titlegap = 3, xlabelpadding = 0, ylabelpadding = 1,
            xticklabelpad = 1, yticklabelpad = 1),
    Axis3 = (xgridcolor = GRID, ygridcolor = GRID, zgridcolor = GRID,
             xgridwidth = 0.4, ygridwidth = 0.4, zgridwidth = 0.4,
             xspinewidth = 0.5, yspinewidth = 0.5, zspinewidth = 0.5,
             xticksize = 2, yticksize = 2, zticksize = 2, titlesize = 7,
             titlefont = :bold, titlealign = :left, titlegap = 3),
    Legend = (framevisible = false, labelsize = LBL, rowgap = 0, patchsize = (8, 5),
              padding = (2, 2, 2, 2), patchlabelgap = 2),
    Lines = (linewidth = 0.8,),
)))

fig = Figure(size = (504, 380), figure_padding = (1, 4, 1, 1))   # 7 in wide (1 unit = 1 pt)
label_txt!(ax, x, y; kw...) = text!(ax, x, y; fontsize = LBL, color = INK2, kw...)

# (a) NRHO ensemble ------------------------------------------------------------------------- #
S = 1e-3                                   # km -> 10^3 km
box_a = ((-10.0, 25.0), (-25.0, 25.0), (-80.0, 10.0))   # plotted region, 10^3 km
# blank out the parts of escaping trajectories that leave the plotted region
clip_a(r) = [all(box_a[k][1] <= r[k, j] * S <= box_a[k][2] for k in 1:3) ? r[i, j] * S : NaN
             for i in 1:3, j in axes(r, 2)]
ax_a = Axis3(fig[1:2, 1]; aspect = :data, azimuth = deg2rad(145), elevation = deg2rad(12),
    xlabel = "x, 10³ km", ylabel = "y, 10³ km", zlabel = "z, 10³ km",
    xlabeloffset = 14, ylabeloffset = 14, zlabeloffset = 24,
    xticks = [0, 20], yticks = [-20, 0, 20], zticks = -80:20:0,
    limits = (box_a...,), title = "(a) NRHO ensemble", protrusions = (10, 32, 12, 6))
# all dispersed trajectories as a single NaN-separated polyline, rasterized as one image
# to keep the PDF light (everything else stays vector)
r_mc = reduce(hcat, [hcat(clip_a(s[1]), fill(NaN, 3)) for s in sols_mc.u])
lines!(ax_a, r_mc[1, :], r_mc[2, :], r_mc[3, :]; color = (C2, 0.15), linewidth = 0.2, rasterize = 8)
lines!(ax_a, r_nom_rot[1, :] * S, r_nom_rot[2, :] * S, r_nom_rot[3, :] * S; color = INK, linewidth = 0.6)
mesh!(ax_a, Sphere(Point3f(0), Float32(R_MOON * S)); color = colorant"#b9b8b2")
r_peri = hcat([pxform(NAIF_FRAME, ROT_FRAME, et0 + t * TU) * sol_nom(t)[1:3] * DU for t in t_peri]...)
scatter!(ax_a, r_peri[1, :] * S, r_peri[2, :] * S, r_peri[3, :] * S; color = C1, markersize = 4,
    strokecolor = :white, strokewidth = 0.4)
r_cmp_nom = R_cmp * sol_nom(t_cmp)[1:3] * DU
scatter!(ax_a, [r_cmp_nom[1] * S], [r_cmp_nom[2] * S], [r_cmp_nom[3] * S]; color = C3,
    marker = :diamond, markersize = 6, strokecolor = :white, strokewidth = 0.4)
Legend(fig[3, 1],
    [LineElement(color = INK, linewidth = 1), LineElement(color = C2, linewidth = 1),
     MarkerElement(color = C1, marker = :circle, markersize = 4),
     MarkerElement(color = C3, marker = :diamond, markersize = 6)],
    ["nominal 9:2 NRHO, $N_rev revs", "$N_mc dispersed (EnsembleThreads)",
     "perilune events (callback)", "epoch of panel (c)"], "Earth–Moon rotating frame";
    titlefont = :regular, titlesize = LBL, titlehalign = :left, titlegap = 1,
    tellwidth = false, tellheight = false, halign = :left, valign = :top)

# (b) acceleration budget ------------------------------------------------------------------- #
days_acc = ts_acc * TU / 86400
ax_b = Axis(fig[1, 2]; yscale = log10, xlabel = "Time, days", ylabel = "|a|, m/s²",
    title = "(b) Acceleration budget", yticks = LogTicks(-9:3:0), yminorticksvisible = false,
    xticks = 0:4:12)
for (k, c) in [(:moon, INK), (:earth, C1), (:sun, C2), (:sh, C3), (:srp, C4)]
    lines!(ax_b, days_acc, acc[k]; color = c)
end
ylims!(ax_b, 1e-11, 30.0)
xlims!(ax_b, 0, days_acc[end])
n_half = length(ts_acc) ÷ 2                     # window holds two perilune passes
i_apo = (argmax(acc[:moon][1:n_half]) + n_half + argmax(acc[:moon][n_half+1:end])) ÷ 2
x_lab = days_acc[i_apo]
label_txt!(ax_b, x_lab, acc[:moon][i_apo] * 3; text = "Moon (point mass)", align = (:center, :bottom))
label_txt!(ax_b, x_lab, acc[:earth][i_apo] / 3; text = "Earth", align = (:center, :top))
label_txt!(ax_b, x_lab, acc[:sun][i_apo] / 1.5; text = "Sun", align = (:center, :top))
label_txt!(ax_b, x_lab, acc[:srp][i_apo] * 1.5; text = "SRP", align = (:center, :bottom))
label_txt!(ax_b, x_lab, acc[:sh][i_apo] / 1.5; text = "Moon SH, n ≤ $NMAX", align = (:center, :top))

# (c) STM vs Monte Carlo -------------------------------------------------------------------- #
# project onto the plane of the two largest principal directions of the STM covariance
F = eigen(Symmetric(P_rr))
V = F.vectors[:, end:-1:end-1]
ξ = V' * δr_mc
L2 = cholesky(Symmetric(V' * P_rr * V)).L
ax_c = Axis(fig[1, 3]; xlabel = "δξ₁, km", ylabel = "δξ₂, km", title = "(c) STM vs. Monte Carlo")
scatter!(ax_c, ξ[1, :], ξ[2, :]; color = (C2, 0.75), markersize = 2.2)
θs = range(0, 2π, length = 200)
for k in 1:3
    e = k * L2 * [cos.(θs)'; sin.(θs)']
    lines!(ax_c, e[1, :], e[2, :]; color = C1, linewidth = 0.7)
    label_txt!(ax_c, e[1, 25], e[2, 25]; text = "$(k)σ", align = (:left, :bottom), offset = (0.5, 0.5))
end
lim1 = 3.6 * sqrt((V' * P_rr * V)[1, 1]); lim2 = 3.6 * sqrt((V' * P_rr * V)[2, 2])
xlims!(ax_c, -lim1, lim1); ylims!(ax_c, -lim2, 1.9 * lim2)
axislegend(ax_c, [LineElement(color = C1, linewidth = 1), MarkerElement(color = C2, marker = :circle, markersize = 3)],
    ["STM (ForwardDiff Jacobian)", "Monte Carlo, $N_mc samples"]; position = :lt)

# (d) lunar gravity field ------------------------------------------------------------------- #
gl_d = GridLayout(fig[2, 2])
ax_d = Axis(gl_d[1, 1]; xlabel = "Longitude, deg", ylabel = "Latitude, deg",
    title = "(d) Lunar gravity, $(NMAX_MAP)×$(NMAX_MAP) field", xticks = -180:90:180, yticks = -90:45:90,
    xgridvisible = false, ygridvisible = false)
gmax = maximum(abs, g_rad)
hm = heatmap!(ax_d, lons, lats, g_rad; colormap = :RdBu, colorrange = (-gmax, gmax))
scatter!(ax_d, first.(peri_lonlat), last.(peri_lonlat); color = INK, marker = :xcross,
    markersize = 4, strokewidth = 0)
xlims!(ax_d, -180, 180); ylims!(ax_d, -90, 90)
Colorbar(gl_d[1, 2], hm; label = "gravity disturbance, mGal", width = 4, ticksize = 2, tickwidth = 0.5,
    spinewidth = 0.5, labelpadding = 1, ticklabelpad = 1, ticklabelsize = LBL, labelsize = LBL)
colgap!(gl_d, 3)

# (e) density models ------------------------------------------------------------------------ #
ax_e = Axis(fig[2, 3]; xscale = log10, xlabel = "Density, kg/m³", ylabel = "Altitude, km",
    title = "(e) Atmosphere models", xticks = LogTicks(-15:3:-6), xminorticksvisible = false,
    yticks = 200:200:1000)
for (ρ, c) in zip(profiles, (C1, C2, C3))
    lines!(ax_e, ρ, collect(alts); color = c)
end
ylims!(ax_e, 100, 1000)
axislegend(ax_e, [LineElement(color = c, linewidth = 1) for c in (C1, C2, C3)],
    first.(density_models); position = :rt)

# (f) drag decay ---------------------------------------------------------------------------- #
ax_f = Axis(fig[3, 2]; xlabel = "Time, days", ylabel = "Altitude, km", title = "(f) LEO decay with drag",
    yticks = 120:40:240)
hlines!(ax_f, [h_stop]; color = INK2, linestyle = :dash, linewidth = 0.5)
for ((days, h), c) in zip(decay, (C1, C2, C3))
    lines!(ax_f, days, h; color = c, linewidth = 0.4)
end
label_txt!(ax_f, 0.0, h_stop; text = "terminal event", align = (:left, :bottom), offset = (1, 1))
label_txt!(ax_f, 0.05, 160; text = "colors as in (e)", align = (:left, :bottom))
ylims!(ax_f, 110, 260)
xlims!(ax_f, 0, nothing)

# (g) backend agreement --------------------------------------------------------------------- #
days_cmp = ts_cmp * TU / 86400
ax_g = Axis(fig[3, 3]; yscale = log10, xlabel = "Time, days", ylabel = "max |Δr| vs. SPICE, m",
    title = "(g) Backends & SPK export", yticks = LogTicks(-6:2:2), yminorticksvisible = false)
for ((name, e), c) in zip(errors_g, (C1, C2, C3, C4))
    lines!(ax_g, days_cmp, max.(e, 1e-7); color = c, label = name)
end
xlims!(ax_g, 0, days_cmp[end])
ylims!(ax_g, 1e-6, 1e4)
axislegend(ax_g; position = :lt, nbanks = 1)

colsize!(fig.layout, 1, Relative(0.30))
colgap!(fig.layout, 1, 6)
colgap!(fig.layout, 2, 8)
rowgap!(fig.layout, 4)

plots_dir = joinpath(@__DIR__, "plots")
mkpath(plots_dir)
save(joinpath(plots_dir, "hero_figure.pdf"), fig; pt_per_unit = 1)
save(joinpath(plots_dir, "hero_figure.png"), fig; px_per_unit = 600 / 72)
println("Saved hero figure to ", plots_dir)
isinteractive() && display(fig)
