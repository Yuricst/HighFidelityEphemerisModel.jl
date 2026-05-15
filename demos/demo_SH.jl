"""Demo for spherical harmonics"""

using GLMakie
using LinearAlgebra
using OrdinaryDiffEq
using SPICE
using Test

include(joinpath(@__DIR__, "../../AstrodynamicsCore.jl/src/AstrodynamicsCore.jl"))
include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))

# furnish SPICE
spice_dir = ENV["SPICE"]
furnsh(joinpath(spice_dir, "lsk", "naif0012.tls"))
furnsh(joinpath(spice_dir, "spk", "de440.bsp"))
furnsh(joinpath(spice_dir, "pck", "gm_de440.tpc"))

furnsh(joinpath(spice_dir, "pck", "earth_latest_high_prec.bpc"))
furnsh(joinpath(spice_dir, "pck", "pck00011.tpc"))
furnsh(joinpath(spice_dir, "fk", "earth_assoc_itrf93.tf"))


# define parameters
naif_ids = ["399",]
GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
naif_frame = "J2000"
abcorr = "NONE"
DU = 1.0
et0 = str2et("2026-01-05T00:00:00")
filepath_spherical_harmonics = joinpath(@__DIR__, "../data/earth/GGM03S.tab")
nmax = 4

parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
    et0, DU, GMs, naif_ids, naif_frame, abcorr;
    filepath_spherical_harmonics = filepath_spherical_harmonics,
    nmax = nmax,
    frame_PCPF = "IAU_EARTH",
    use_canonical_scales = false,
)

# initial state
rv0 = AstrodynamicsCore.kep2rv([8000.0, 0.05, deg2rad(47.0), deg2rad(30), deg2rad(-40), deg2rad(60.0)], GMs[1])
period = 2π * sqrt(8000^3/GMs[1])

# propagate 
tspan = (et0, et0 + 10 * period)
ode = ODEProblem(HighFidelityEphemerisModel.eom_NbodySH_SPICE!, rv0, tspan, parameters)
sol = solve(ode, Tsit5(), reltol=1e-11, abstol=1e-12)

# elements history
kep_hist = hcat([AstrodynamicsCore.rv2kep(rv, GMs[1]) for rv in sol.u]...)

# plot
labels = ["a, km", "e", "i, deg", "Ω, deg", "ω, deg", "M, deg"]
multipliers = [1.0, 1.0, 180/π, 180/π, 180/π, 180/π]
fig_elements = Figure(size=(1200,800))
for i in 1:6
    nrow, ncol = fld1(i, 3), mod1(i, 3)
    ax_W = Axis(fig_elements[nrow,ncol]; xlabel="Time, hours", ylabel=labels[i])
    lines!(ax_W, (sol.t .- sol.t[1])/3600, multipliers[i] * kep_hist[i,:], color=:blue)
end

# fig = Figure(size=(1200,800))
# ax_inr = Axis3(fig[1,1]; aspect=:data, xlabel="x, km", ylabel="y, km", zlabel="z, km")
# lines!(ax_inr, sol.u[1, :], sol.u[2, :], sol.u[3, :], color=:blue)

display(fig_elements)