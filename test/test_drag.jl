"""Tests for atmospheric drag perturbation"""

using LinearAlgebra
using OrdinaryDiffEq
using SPICE
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end


function test_get_drag_coefficient()
    DU = 6378.0
    GM = 398600.4415
    VU = sqrt(GM / DU)
    TU = DU / VU
    drag_Cd = 2.2
    drag_Am = 0.01
    rho = 1e-12
    v_can = [1.0, 0.2, -0.1]

    k_drag = HighFidelityEphemerisModel.get_drag_coefficient(DU, TU, VU, drag_Cd, drag_Am)
    v_atm = zeros(3)
    a_can = HighFidelityEphemerisModel.drag(zeros(3), v_can, v_atm, rho, k_drag)

    v_km = v_can * VU
    v_m = v_km * 1000.0
    a_m = -0.5 * rho * drag_Cd * drag_Am * norm(v_m) * v_m
    a_km = a_m / 1000.0
    a_can_expected = a_km * TU^2 / DU

    @test a_can ≈ a_can_expected atol=1e-12
end


function test_atmospheric_velocity()
    r_can = [1.0, 0.3, -0.2]
    TU = 806.0
    omega_atm = [0.0, 0.0, 7.2921159e-5]

    v_atm = HighFidelityEphemerisModel.atmospheric_velocity(r_can, TU, omega_atm)
    @test v_atm ≈ TU * cross(omega_atm, r_can) atol=1e-14
end


function test_drag_opposes_relative_velocity()
    k_drag = 1.0
    rho = 1e-10
    v = [0.5, 0.1, -0.2]
    v_atm = [0.1, 0.0, 0.05]
    a = HighFidelityEphemerisModel.drag(zeros(3), v, v_atm, rho, k_drag)
    v_rel = v - v_atm
    @test dot(a, v_rel) < 0.0
end


function test_eom_Nbody_SPICE_drag()
    naif_ids = ["399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 6378.0

    et0 = str2et("2020-01-01T00:00:00")
    # f_density = (et, r) -> 1e-12
    f_density = HighFidelityEphemerisModel.harris_priester_f_density()
    parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        include_drag = true,
        drag_Cd = 2.2,
        drag_Am = 0.01,
        f_density = f_density,
    )

    u0 = [1.05, 0.0, 0.01, 0.0, 1.0, 0.0]
    tspan = (0.0, 2 * 86400 / parameters.TU)

    prob = ODEProblem(HighFidelityEphemerisModel.eom_Nbody_SPICE!, u0, tspan, parameters)
    sol = solve(prob, Vern7(), reltol=1e-12, abstol=1e-12)
    u_check = [-0.6474833041677026, 0.932094603810321, -0.0061648570310309965,
               -0.7821138078881879, -0.4957243892672692, -0.0074455370225118885]
    @test norm(sol.u[end] - u_check) < 1e-11

    et = parameters.et0
    r_km = u0[1:3] * parameters.DU
    rho = f_density(et, r_km)
    v_atm = HighFidelityEphemerisModel.atmospheric_velocity(u0[1:3], parameters.TU, parameters.omega_atm)
    a = HighFidelityEphemerisModel.drag(u0[1:3], u0[4:6], v_atm, rho, parameters.k_drag)
    v_rel = u0[4:6] - v_atm
    @test dot(a, v_rel) < 0.0
end


function test_harris_priester_model()
    rho_min = HighFidelityEphemerisModel.HarrisPriesterModel(400.0; use_min=true)
    rho_max = HighFidelityEphemerisModel.HarrisPriesterModel(400.0; use_min=false)
    @test rho_min ≈ 2.249e-12 atol=1e-15
    @test rho_max ≈ 7.492e-12 atol=1e-15

    rho_clamped = HighFidelityEphemerisModel.HarrisPriesterModel(50.0; use_min=true)
    @test rho_clamped ≈ 497400.0 * 1e-12 atol=1e-15

    f_density = HighFidelityEphemerisModel.harris_priester_f_density(6378.0; use_min=true)
    r_km = [6378.0 + 400.0, 0.0, 0.0]
    @test f_density(0.0, r_km) ≈ rho_min atol=1e-15
end


test_get_drag_coefficient()
test_atmospheric_velocity()
test_drag_opposes_relative_velocity()
test_eom_Nbody_SPICE_drag()
test_harris_priester_model()
