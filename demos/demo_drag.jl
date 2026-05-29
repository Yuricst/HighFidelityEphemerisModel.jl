"""Demo using atmospheric drag term"""

using GLMakie
using LinearAlgebra
using OrdinaryDiffEq
using SPICE

include(joinpath(@__DIR__, "../../AstrodynamicsCore.jl/src/AstrodynamicsCore.jl"))
include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))

# load SPICE kernels
spice_dir = ENV["SPICE"]
furnsh(joinpath(spice_dir, "lsk", "naif0012.tls"))
furnsh(joinpath(spice_dir, "spk", "de440.bsp"))
furnsh(joinpath(spice_dir, "pck", "gm_de440.tpc"))

furnsh(joinpath(spice_dir, "pck", "earth_latest_high_prec.bpc"))
furnsh(joinpath(spice_dir, "pck", "pck00011.tpc"))
furnsh(joinpath(spice_dir, "fk", "earth_assoc_itrf93.tf"))

naif_frame = "J2000"
abcorr = "NONE"
et0 = str2et("2026-01-05T00:00:00")

naif_ids = ["399",]# "301", "10"]   # Earth-centered; Sun for optional third-body / SRP
naif_ids = ["399", "301", "10"]   # Earth-centered; Sun for optional third-body / SRP
GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
DU = 6378.0
R_earth_km = 6378.0        # Earth radius in km (for Harris–Priester altitude; not DU)

filepath_spherical_harmonics = joinpath(@__DIR__, "../data/earth/GGM03S.tab")
nmax = 4

f_density = HighFidelityEphemerisModel.harris_priester_f_density(R_earth_km; use_min=true)

parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
    et0, DU, GMs, naif_ids, naif_frame, abcorr;
    filepath_spherical_harmonics = filepath_spherical_harmonics,
    nmax = nmax,
    frame_PCPF = "IAU_EARTH",
    include_drag = true,
    drag_Cd = 2.2,
    drag_Am = 0.03,
    f_density = f_density,
)

kep0 = [(DU+250)/DU, 1e-3, deg2rad(50), deg2rad(100), deg2rad(200), deg2rad(300)]
x0_earth = AstrodynamicsCore.kep2rv(kep0, parameters.mus[1])
tspan = (0.0, 3 * 86400 / parameters.TU)
ode = ODEProblem(HighFidelityEphemerisModel.eom_Nbody_SPICE!, x0_earth, tspan, parameters)

# detect hitting min altitude
function condition(u, t, integrator) # Event when condition(u,t,integrator) == 0
    return norm(u[1:3]) - (R_earth_km + 100.0)/parameters.DU
end
affect!(integrator) = terminate!(integrator)
cb = ContinuousCallback(condition, affect!)
sol = solve(ode, Vern7(), reltol=1e-12, abstol=1e-12, callback=cb)

# elements history
kep_hist = hcat([AstrodynamicsCore.rv2kep(rv, parameters.mus[1]) for rv in sol.u]...)

# plot
fontsize = 18
labels = ["h, km", "e", "i, deg", "Ω, deg", "ω, deg", "θ, deg"]
multipliers = [DU, 1.0, 180/π, 180/π, 180/π, 180/π]
offset = [R_earth_km,0,0,0,0,0,0]
fig_elements = Figure(size=(1500,600))
ax_h = Axis(fig_elements[1,1]; xlabel="Time, hours", ylabel="h, km",
    xlabelsize = fontsize, ylabelsize = fontsize, xticklabelsize = fontsize, yticklabelsize = fontsize)
lines!(ax_h, (sol.t .- sol.t[1])*parameters.TU/3600, norm.(eachcol(Array(sol)[1:3,:])) * parameters.DU .- R_earth_km, color=:blue)
hlines!(ax_h, [100.0], color=:red, linestyle=:dash)

for i in 2:6
    nrow, ncol = fld1(i, 3), mod1(i, 3)
    ax_W = Axis(fig_elements[nrow,ncol]; xlabel="Time, hours", ylabel=labels[i],
        xlabelsize = fontsize, ylabelsize = fontsize, xticklabelsize = fontsize, yticklabelsize = fontsize)
    lines!(ax_W, (sol.t .- sol.t[1])*parameters.TU/3600, multipliers[i] * kep_hist[i,:] .- offset[i], color=:blue)
end

ax_traj = Axis3(fig_elements[1:2,4:7]; xlabel="x, km", ylabel="y, km", zlabel="z, km", aspect=:data,
    xlabelsize = fontsize, ylabelsize = fontsize, zlabelsize = fontsize,
    xticklabelsize = fontsize, yticklabelsize = fontsize, zticklabelsize = fontsize)
lines!(ax_traj, Array(sol)[1,:], Array(sol)[2,:], Array(sol)[3,:], color=:blue)

save(joinpath(@__DIR__, "plots", "demo_drag.png"), fig_elements; px_per_unit=2)
display(fig_elements)