"""Demo for spherical harmonics"""

using GLMakie
using LinearAlgebra
using OrdinaryDiffEq
using SPICE
using Test

include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))

function _kepler_E(M, e; atol=1e-13)
    E = e < 0.8 ? mod(M, 2π) : π
    for _ in 1:30
        ΔE = (E - e * sin(E) - M) / (1 - e * cos(E))
        E -= ΔE
        abs(ΔE) < atol && break
    end
    return E
end

function _pqw_to_inertial(i, Ω, ω)
    cΩ, sΩ = cos(Ω), sin(Ω)
    ci, si = cos(i), sin(i)
    cω, sω = cos(ω), sin(ω)
    return [
        cΩ*cω - sΩ*sω*ci  -cΩ*sω - sΩ*cω*ci   sΩ*si;
        sΩ*cω + cΩ*sω*ci  -sΩ*sω + cΩ*cω*ci  -cΩ*si;
        sω*si              cω*si               ci
    ]
end

function _kep2rv(kep, μ)
    a, e, i, Ω, ω, M = kep
    E = _kepler_E(M, e)
    ν = 2 * atan(sqrt(1 + e) * sin(E/2), sqrt(1 - e) * cos(E/2))
    p = a * (1 - e^2)
    r_pqw = p / (1 + e * cos(ν)) * [cos(ν), sin(ν), 0.0]
    v_pqw = sqrt(μ / p) * [-sin(ν), e + cos(ν), 0.0]
    T = _pqw_to_inertial(i, Ω, ω)
    return [T * r_pqw; T * v_pqw]
end

function _rv2kep(rv, μ)
    rvec, vvec = rv[1:3], rv[4:6]
    r = norm(rvec)
    v = norm(vvec)
    hvec = cross(rvec, vvec)
    h = norm(hvec)
    nvec = cross([0.0, 0.0, 1.0], hvec)
    n = norm(nvec)
    evec = cross(vvec, hvec) / μ - rvec / r
    e = norm(evec)
    a = -μ / (v^2 - 2μ / r)
    i = acos(clamp(hvec[3] / h, -1.0, 1.0))
    Ω = n > 1e-12 ? mod(atan(nvec[2], nvec[1]), 2π) : 0.0
    ω = (n > 1e-12 && e > 1e-12) ? mod(atan(dot(cross(nvec, evec), hvec) / (n * e * h), dot(nvec, evec) / (n * e)), 2π) : 0.0
    ν = e > 1e-12 ? mod(atan(dot(cross(evec, rvec), hvec) / (e * r * h), dot(evec, rvec) / (e * r)), 2π) : 0.0
    E = 2 * atan(sqrt(1 - e) * sin(ν/2), sqrt(1 + e) * cos(ν/2))
    M = mod(E - e * sin(E), 2π)
    return [a, e, i, Ω, ω, M]
end

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
rv0 = _kep2rv([8000.0, 0.05, deg2rad(47.0), deg2rad(30), deg2rad(-40), deg2rad(60.0)], GMs[1])
period = 2π * sqrt(8000^3/GMs[1])

# propagate; EOM time is elapsed seconds from et0
tspan = (0.0, 10 * period)
ode = ODEProblem(HighFidelityEphemerisModel.eom_NbodySH_SPICE!, rv0, tspan, parameters)
sol = solve(ode, Tsit5(), reltol=1e-11, abstol=1e-12)

# elements history
kep_hist = hcat([_rv2kep(rv, GMs[1]) for rv in sol.u]...)

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