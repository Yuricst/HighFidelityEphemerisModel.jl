"""Compare Jacchia-Roberts vs Harris-Priester density vs altitude"""

using GLMakie
using SPICE

include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))

if haskey(ENV, "SPICE")
    spice_dir = ENV["SPICE"]
    furnsh(joinpath(spice_dir, "lsk", "naif0012.tls"))
    furnsh(joinpath(spice_dir, "spk", "de440.bsp"))
    furnsh(joinpath(spice_dir, "pck", "gm_de440.tpc"))
    furnsh(joinpath(spice_dir, "pck", "earth_latest_high_prec.bpc"))
    furnsh(joinpath(spice_dir, "pck", "pck00011.tpc"))
else
    include(joinpath(@__DIR__, "../test/utils.jl"))
    furnsh_kernels()
end

et = str2et("2020-01-01T00:00:00")
naif_frame = "J2000"
frame_PCPF = "IAU_EARTH"
Re = bodvrd("399", "RADII", 3)[1]
R_earth_km = 6378.0

f_jr = HighFidelityEphemerisModel.jacchia_roberts_f_density(frame_PCPF=frame_PCPF)
f_hp_min = HighFidelityEphemerisModel.harris_priester_f_density(R_earth_km; use_min=true)
f_hp_max = HighFidelityEphemerisModel.harris_priester_f_density(R_earth_km; use_min=false)

alts = 120.0:2.0:1000.0
rho_jr = Float64[]
rho_hp_min = Float64[]
rho_hp_max = Float64[]
T = SPICE.pxform(naif_frame, frame_PCPF, et)

for alt in alts
    r_pcpf = T * [Re + alt, 0.0, 0.0]
    push!(rho_jr, f_jr(et, r_pcpf))
    push!(rho_hp_min, f_hp_min(et, r_pcpf))
    push!(rho_hp_max, f_hp_max(et, r_pcpf))
end

fig = Figure(size=(900, 500))
ax = Axis(
    fig[1, 1];
    xlabel="Altitude, km",
    ylabel="Density, kg/m³",
    yscale=log10,
    title="Atmosphere models (2020-01-01, equatorial)",
)
lines!(ax, collect(alts), rho_jr; label="Jacchia-Roberts")
lines!(ax, collect(alts), rho_hp_min; label="Harris-Priester (min)")
lines!(ax, collect(alts), rho_hp_max; label="Harris-Priester (max)")
axislegend(ax; position=:rt)

plots_dir = joinpath(@__DIR__, "plots")
mkpath(plots_dir)
save(joinpath(plots_dir, "demo_atmosphere_jr_vs_hp.png"), fig; px_per_unit=2)
isinteractive() && display(fig)