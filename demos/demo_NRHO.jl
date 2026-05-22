"""Demo propagating NRHO state"""


using LinearAlgebra
using OrdinaryDiffEq
using Printf
using Random
using SPICE
using Test
using GLMakie

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end

include(joinpath(@__DIR__, "..", "test", "utils.jl"))
furnsh_kernels()
furnsh(joinpath(ENV["SPICE"], "fk", "earth_moon_rotating_mc.tf"))

# define parameters
naif_ids = ["301", "399", "10"]
GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
naif_frame = "J2000"
abcorr = "NONE"
DU = 100_000.0

et0 = str2et("2028-01-03T08:00:00")
parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(et0, DU, GMs, naif_ids, naif_frame, abcorr)

x0 = spkezr("-60000", et0, naif_frame, abcorr, "399")[1] - spkezr("301", et0, naif_frame, abcorr, "399")[1]
x0[1:3] ./= DU
x0[4:6] ./= parameters.VU

# propagate nominal
N_rev = 10
period = 6.55 * 86400.0 / parameters.TU
tspan = (0.0, N_rev * period)
prob = ODEProblem(HighFidelityEphemerisModel.eom_Nbody_SPICE!, x0, tspan, parameters)
sol = solve(prob, Vern8(), reltol=1e-12, abstol=1e-12)

# small perturbations
N_mc = 30
δr_sigma = 1.0 / parameters.DU
δv_sigma = 1e-3 * 0.1 / parameters.VU
sols = ODESolution[]
for i in 1:N_mc
    @printf("   Propagating perturbed trajectory %d of %d\n", i, N_mc)
    Random.seed!(i)
    δr = δr_sigma * randn(3)
    δv = δv_sigma * randn(3)
    x0_ptrb = x0 + [δr; δv]
    _prob = ODEProblem(HighFidelityEphemerisModel.eom_Nbody_SPICE!, x0_ptrb, tspan, parameters)
    _sol = solve(_prob, Vern8(), reltol=1e-12, abstol=1e-12)
    push!(sols, _sol)
end

# rotate result into Earth-Moon rotating frame
DU2SI = [parameters.DU, parameters.DU, parameters.DU, parameters.VU, parameters.VU, parameters.VU]
Ts_inr2em = sxform.(naif_frame, "EARTHMOONROTATINGMC", et0 .+ sol.t * parameters.TU)
rs_em = hcat([T * u .* DU2SI for (T, u) in zip(Ts_inr2em, sol.u)]...)

# plot trajectory in Earth-Moon rotating frame 
labels = ["x, 1e4 km", "y, 1e4 km", "z, 1e4 km", "vx, km/s", "vy, km/s", "vz, km/s"]
SCALE_PLOT = 1 / 1e4
fontsize = 18

fig = Figure(size=(1600,500))
ax = Axis3(fig[1:2,1]; aspect=:data, xlabel="x, 1e4 km", ylabel="y, 1e4 km", zlabel="z, 1e4 km",
    xlabelsize = fontsize, ylabelsize = fontsize, zlabelsize = fontsize,
    xticklabelsize = fontsize, yticklabelsize = fontsize, zticklabelsize = fontsize,
    azimuth=deg2rad(190), elevation=deg2rad(10))
axrs = [Axis(fig[1,1+i]; xlabel="Time, day", ylabel=labels[i], xlabelsize = fontsize, ylabelsize = fontsize, xticklabelsize = fontsize, yticklabelsize = fontsize) for i in 1:3]
axvs = [Axis(fig[2,1+i]; xlabel="Time, day", ylabel=labels[i+3], xlabelsize = fontsize, ylabelsize = fontsize, xticklabelsize = fontsize, yticklabelsize = fontsize) for i in 1:3]

scatter!(ax, rs_em[1,1] * SCALE_PLOT, rs_em[2,1] * SCALE_PLOT, rs_em[3,1] * SCALE_PLOT, color=:black, markersize=10)
lines!(ax, rs_em[1, :] * SCALE_PLOT, rs_em[2, :] * SCALE_PLOT, rs_em[3, :] * SCALE_PLOT, color=:black)
for i in 1:3
    lines!(axrs[i], sol.t * parameters.TU / 86400, rs_em[i, :] * SCALE_PLOT, color=:black)
    lines!(axvs[i], sol.t * parameters.TU / 86400, rs_em[i+3, :], color=:black)
end

for _sol in sols
    _Ts_inr2em = sxform.(naif_frame, "EARTHMOONROTATINGMC", et0 .+ _sol.t * parameters.TU)
    _rs_em = hcat([T * u .* DU2SI for (T, u) in zip(_Ts_inr2em, _sol.u)]...)
    scatter!(ax, _rs_em[1,1] * SCALE_PLOT, _rs_em[2,1] * SCALE_PLOT, _rs_em[3,1] * SCALE_PLOT, color=:crimson, markersize=10)
    lines!(ax, _rs_em[1, :] * SCALE_PLOT, _rs_em[2, :] * SCALE_PLOT, _rs_em[3, :] * SCALE_PLOT, color=:crimson, linewidth=0.25)

    for i in 1:3
        lines!(axrs[i], _sol.t * parameters.TU / 86400, _rs_em[i, :] * SCALE_PLOT, color=:crimson, linewidth=0.15)
        lines!(axvs[i], _sol.t * parameters.TU / 86400, _rs_em[i+3, :], color=:crimson, linewidth=0.15)
    end
end
ylims!(ax, -3, 3)
zlims!(ax, -9, 1.5)
save(joinpath(@__DIR__, "plots", "demo_NRHO_deviations.png"), fig; px_per_unit=2)
fig