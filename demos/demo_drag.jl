"""Demo using atmospheric drag term"""

using SPICE
using OrdinaryDiffEq
using GLMakie

include(joinpath(@__DIR__, "../../AstrodynamicsCore.jl/src/AstrodynamicsCore.jl"))
include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))

# load SPICE kernels
spice_dir = ENV["SPICE"]
furnsh(joinpath(spice_dir, "lsk", "naif0012.tls"))
furnsh(joinpath(spice_dir, "spk", "de440.bsp"))
furnsh(joinpath(spice_dir, "pck", "gm_de440.tpc"))
furnsh(joinpath(spice_dir, "pck", "moon_pa_de440_200625.bpc"))
furnsh(joinpath(spice_dir, "fk", "moon_de440_250416.tf"))

naif_frame = "J2000"
abcorr = "NONE"
et0 = str2et("2026-01-05T00:00:00")

naif_ids = ["399",]# "301", "10"]   # Earth-centered; Sun for optional third-body / SRP
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
    drag_Am = 0.01,
    f_density = f_density,
)

kep0 = [6600/DU, 1e-3, deg2rad(50), deg2rad(100), deg2rad(200), deg2rad(300)]
x0_earth = AstrodynamicsCore.kep2rv(kep0, parameters.mus[1])
tspan = (0.0, 3 * 86400 / parameters.TU)
ode = ODEProblem(HighFidelityEphemerisModel.eom_NbodySH_SPICE!, x0_earth, tspan, parameters)
sol = solve(ode, Vern7(), reltol=1e-12, abstol=1e-12)

# elements history
kep_hist = hcat([AstrodynamicsCore.rv2kep(rv, parameters.mus[1]) for rv in sol.u]...)

# plot
labels = ["a, km", "e", "i, deg", "Ω, deg", "ω, deg", "θ, deg"]
multipliers = [DU, 1.0, 180/π, 180/π, 180/π, 180/π]
fig_elements = Figure(size=(1600,800))
for i in 1:6
    nrow, ncol = fld1(i, 3), mod1(i, 3)
    ax_W = Axis(fig_elements[nrow,ncol]; xlabel="Time, hours", ylabel=labels[i])
    lines!(ax_W, (sol.t .- sol.t[1])/3600, multipliers[i] * kep_hist[i,:], color=:blue)
end

ax_traj = Axis3(fig_elements[1:2,4:8]; xlabel="x, km", ylabel="y, km", zlabel="z, km", aspect=:data)
lines!(ax_traj, Array(sol)[1,:], Array(sol)[2,:], Array(sol)[3,:], color=:blue)

display(fig_elements)