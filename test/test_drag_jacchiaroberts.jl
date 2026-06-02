"""Tests for Jacchia-Roberts atmospheric density model"""

using LinearAlgebra
using OrdinaryDiffEq
using SPICE
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "utils.jl"))
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
    furnsh_kernels()
end


function _pcpf_position(et, alt_km, naif_frame="J2000", frame_PCPF="IAU_EARTH")
    Re = bodvrd("399", "RADII", 3)[1]
    r_inertial = [Re + alt_km, 0.0, 0.0]
    T = SPICE.pxform(naif_frame, frame_PCPF, et)
    return T * r_inertial
end


function test_jacchia_roberts_f_density_api()
    et = str2et("2020-01-01T00:00:00")
    f_density = HighFidelityEphemerisModel.jacchia_roberts_f_density()
    r_km = _pcpf_position(et, 400.0)
    rho = f_density(et, r_km)
    @test isfinite(rho)
    @test rho > 0.0
end


function test_jacchia_roberts_altitude_guard()
    et = str2et("2020-01-01T00:00:00")
    f_density = HighFidelityEphemerisModel.jacchia_roberts_f_density()
    r_km = _pcpf_position(et, 80.0)
    @test_throws ErrorException f_density(et, r_km)
end


function test_jacchia_roberts_regression()
    et = str2et("2020-01-01T00:00:00")
    f_density = HighFidelityEphemerisModel.jacchia_roberts_f_density()
    @test f_density(et, _pcpf_position(et, 200.0)) ≈ 3.0708677711266465e-10 rtol=1e-6
    @test f_density(et, _pcpf_position(et, 400.0)) ≈ 5.591753480974712e-12 rtol=1e-6
    @test f_density(et, _pcpf_position(et, 800.0)) ≈ 2.5775215118405146e-14 rtol=1e-6
end


function test_jacchia_roberts_vs_harris_priester()
    et = str2et("2020-01-01T00:00:00")
    r_km = _pcpf_position(et, 400.0)
    rho_jr = HighFidelityEphemerisModel.jacchia_roberts_f_density()(et, r_km)
    rho_hp = HighFidelityEphemerisModel.harris_priester_f_density(6378.0; use_min=true)(et, r_km)
    @test rho_jr > 0.0
    @test rho_hp > 0.0
    @test rho_jr != rho_hp
    @test abs(rho_jr - rho_hp) < 1e-11
end


function test_jacchia_roberts_eom_smoke()
    naif_ids = ["399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    et0 = str2et("2020-01-01T00:00:00")
    DU = 6378.0
    f_density = HighFidelityEphemerisModel.jacchia_roberts_f_density(frame_PCPF="IAU_EARTH")
    parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, DU, GMs, naif_ids, "J2000", "NONE";
        frame_PCPF = "IAU_EARTH",
        include_drag = true,
        drag_Cd = 2.2,
        drag_Am = 0.01,
        f_density = f_density,
    )
    u0 = [1.05, 0.0, 0.01, 0.0, 1.0, 0.0]
    tspan = (0.0, 3600.0 / parameters.TU)
    prob = ODEProblem(HighFidelityEphemerisModel.eom_Nbody_SPICE!, u0, tspan, parameters)
    sol = solve(prob, Vern7(), reltol=1e-12, abstol=1e-12)
    @test all(isfinite, sol.u[end])
end


# test_jacchia_roberts_f_density_api()
# test_jacchia_roberts_altitude_guard()
# test_jacchia_roberts_regression()
test_jacchia_roberts_vs_harris_priester()
# test_jacchia_roberts_eom_smoke()
