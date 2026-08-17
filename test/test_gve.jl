"""Test Gauss variational equations and RTN transformations."""

using AstrodynamicsCore
using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq
using SPICE
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end


@testset "RTN transformations" begin
    r = [2.0, -1.0, 3.0]
    v = [0.4, 1.2, -0.1]
    T = HighFidelityEphemerisModel.pxform_inr2rtn(r, v)
    h = cross(r, v)

    @test T * T' ≈ I(3) atol = 1e-14
    @test T' * T ≈ I(3) atol = 1e-14
    @test det(T) ≈ 1.0 atol = 1e-14
    @test T[1, :] ≈ r / norm(r)
    @test T[3, :] ≈ h / norm(h)
    @test cross(T[1, :], T[2, :]) ≈ T[3, :]

    a_inr = [-0.2, 0.4, 0.7]
    a_rtn = zeros(3)
    HighFidelityEphemerisModel.project_inr_to_rtn!(a_rtn, a_inr, r, v)
    @test a_rtn ≈ T * a_inr
    @test HighFidelityEphemerisModel.pxform_rtn2inr(r, v) * a_rtn ≈ a_inr

    T_simple = HighFidelityEphemerisModel.pxform_inr2rtn(
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    @test T_simple ≈ I(3)
    @test_throws DomainError HighFidelityEphemerisModel.pxform_inr2rtn(zeros(3), ones(3))
    @test_throws DomainError HighFidelityEphemerisModel.pxform_inr2rtn(
        [1.0, 0.0, 0.0], [2.0, 0.0, 0.0])
end


@testset "AstrodynamicsCore conversions" begin
    mu = 1.0
    keplerian_cases = [
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.3],
        [2.0, 0.0, 0.6, 0.4, 0.0, 1.2],
        [2.5, 0.2, 0.0, 0.0, 0.7, 2.1],
        [3.0, 0.4, 0.8, 1.0, 0.5, 2.4],
        [8.0, 0.85, 0.3, 2.0, 1.1, 0.2],
    ]

    for kep in keplerian_cases
        rv = AstrodynamicsCore.kep2rv(kep, mu)
        mee = AstrodynamicsCore.kep2mee(kep)
        @test AstrodynamicsCore.mee2rv(mee, mu) ≈ rv rtol = 1e-12 atol = 1e-12
        @test AstrodynamicsCore.kep2rv(AstrodynamicsCore.rv2kep(rv, mu), mu) ≈ rv rtol = 1e-12 atol = 1e-12
        @test AstrodynamicsCore.mee2rv(AstrodynamicsCore.rv2mee(rv, mu), mu) ≈ rv rtol = 1e-12 atol = 1e-12
    end

    for (kep, mu_dimensional) in (
        ([8000.0, 0.12, deg2rad(85.0), deg2rad(240.0), deg2rad(200.0), deg2rad(130.0)],
            398600.435507),
        ([2000.0, 0.05, deg2rad(30.0), deg2rad(80.0), deg2rad(20.0), deg2rad(45.0)],
            4902.800066),
    )
        rv = AstrodynamicsCore.kep2rv(kep, mu_dimensional)
        @test AstrodynamicsCore.kep2rv(AstrodynamicsCore.rv2kep(rv, mu_dimensional),
            mu_dimensional) ≈ rv rtol = 1e-12 atol = 1e-12
        @test AstrodynamicsCore.mee2rv(AstrodynamicsCore.rv2mee(rv, mu_dimensional),
            mu_dimensional) ≈ rv rtol = 1e-12 atol = 1e-12
    end
end


@testset "MEE derivatives" begin
    mee = [2.0, 0.1, -0.05, 0.2, -0.1, 0.7]
    dmee = zeros(6)

    HighFidelityEphemerisModel.gve_mee_derivs!(dmee, mee, zeros(3), 1.0)
    @test dmee ≈ [0.0, 0.0, 0.0, 0.0, 0.0, 0.38555237549806753]

    HighFidelityEphemerisModel.gve_mee_derivs!(dmee, mee, [1.0, 0.0, 0.0], 1.0)
    @test dmee ≈ [0.0, 0.9110613904121715, -1.0816501943328265,
        0.0, 0.0, 0.38555237549806753]

    HighFidelityEphemerisModel.gve_mee_derivs!(dmee, mee, [0.0, 1.0, 0.0], 1.0)
    @test dmee ≈ [5.417024512000698, 2.2528680262159195, 1.715784334719905,
        0.0, 0.0, 0.38555237549806753]

    HighFidelityEphemerisModel.gve_mee_derivs!(dmee, mee, [0.0, 0.0, 1.0], 1.0)
    @test dmee ≈ [0.0, 0.01390331860249281, 0.02780663720498562,
        0.5437909150186148, 0.45802876912156504, 0.6636187475479238]

    f_mee = function(x)
        dx = similar(x)
        HighFidelityEphemerisModel.gve_mee_derivs!(dx, x, [0.01, -0.02, 0.03], 1.0)
        return dx
    end
    @test all(isfinite, ForwardDiff.jacobian(f_mee, mee))
    @test_throws DomainError HighFidelityEphemerisModel.gve_mee_derivs!(
        dmee, [0.0, mee[2:end]...], zeros(3), 1.0)
end


@testset "Two-body element propagation" begin
    params = HighFidelityEphemerisModel.SpiceParameters(
        0.0, 1.0, [1.0], ["399"], "J2000", "NONE")
    keplerian_cases = [
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.3],
        [2.0, 0.0, 0.6, 0.4, 0.0, 1.2],
        [2.5, 0.2, 1e-4, 0.2, 0.7, 2.1],
        [3.0, 0.4, 0.8, 1.0, 0.5, 2.4],
    ]

    for kep0 in keplerian_cases
        rv0 = AstrodynamicsCore.kep2rv(kep0, 1.0)
        mee0 = AstrodynamicsCore.kep2mee(kep0)
        ts = range(0.0, 4.0, length = 21)
        sol_rv = solve(ODEProblem(HighFidelityEphemerisModel.eom_Nbody!, rv0,
            (ts[1], ts[end]), params), Vern9(), reltol = 1e-13, abstol = 1e-13,
            saveat = ts)
        sol_mee = solve(ODEProblem(HighFidelityEphemerisModel.gve_mee_Nbody!, mee0,
            (ts[1], ts[end]), params), Vern9(), reltol = 1e-13, abstol = 1e-13,
            saveat = ts)
        errors = [norm(AstrodynamicsCore.mee2rv(sol_mee(t), 1.0) - sol_rv(t)) for t in ts]
        @test maximum(errors) < 2e-10
    end

    kep0 = [2.5, 0.2, 0.5, 0.3, 0.7, 1.1]
    rv0 = AstrodynamicsCore.kep2rv(kep0, 1.0)
    sol_rv = solve(ODEProblem(HighFidelityEphemerisModel.eom_Nbody!, rv0,
        (0.0, 4.0), params), Vern9(), reltol = 1e-13, abstol = 1e-13)
    sol_kep = solve(ODEProblem(HighFidelityEphemerisModel.eom_kep_twobody!, kep0,
        (0.0, 4.0), 1.0), Vern9(), reltol = 1e-13, abstol = 1e-13)
    @test AstrodynamicsCore.kep2rv(sol_kep.u[end], 1.0) ≈ sol_rv.u[end] atol = 2e-10
end


@testset "GVE backend dispatch" begin
    mee = [2.0, 0.1, -0.05, 0.2, -0.1, 0.7]
    params_SPICE = HighFidelityEphemerisModel.SpiceParameters(
        0.0, 1.0, [1.0], ["399"], "J2000", "NONE")
    params_Interp = HighFidelityEphemerisModel.InterpParameters(
        0.0, 1.0, [1.0], ["399"], "J2000", "NONE";
        interpolate_ephem_span = [0.0, 7200.0])
    params_Ephemerides = HighFidelityEphemerisModel.EphemeridesParameters(
        0.0, 1.0, [1.0], ["399"], "J2000", "NONE";
        ephemerides_frame_system = :test_frame_system)
    expected = zeros(6)
    HighFidelityEphemerisModel.gve_mee_derivs!(expected, mee, zeros(3), 1.0)

    for params in (params_SPICE, params_Interp, params_Ephemerides)
        @test HighFidelityEphemerisModel.gve_mee_Nbody(mee, params, 0.0) ≈ expected
    end
    for params in (params_Interp, params_Ephemerides)
        @test HighFidelityEphemerisModel.gve_mee_NbodySH(mee, params, 0.0) ≈ expected
    end
end


@testset "GVE and Cartesian force models" begin
    naif_ids = ["301", "399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    et0 = str2et("2020-01-01T00:00:00")
    filepath_SH = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab")
    params = HighFidelityEphemerisModel.SpiceParameters(
        et0, 3000.0, GMs, naif_ids, "J2000", "NONE")
    params_SH = HighFidelityEphemerisModel.SpiceParameters(
        et0, 3000.0, GMs, naif_ids, "J2000", "NONE";
        filepath_spherical_harmonics = filepath_SH, nmax = 4, frame_PCPF = "MOON_PA")
    rv0 = [1.0, 0.0, 0.3, 0.5, 1.0, 0.0]
    kep0 = AstrodynamicsCore.rv2kep(rv0, params.mus[1])
    rv_cases = [
        rv0,
        AstrodynamicsCore.kep2rv([1.5, 0.01, 1e-4, 0.2, 0.4, 0.6], params.mus[1]),
        AstrodynamicsCore.kep2rv([2.0, 0.3, 0.6, 1.0, 0.5, 2.0], params.mus[1]),
    ]

    for (eom, gve, parameters) in (
        (HighFidelityEphemerisModel.eom_Nbody!, HighFidelityEphemerisModel.gve_mee_Nbody!, params),
        (HighFidelityEphemerisModel.eom_NbodySH!, HighFidelityEphemerisModel.gve_mee_NbodySH!, params_SH),
    ), initial_rv in rv_cases
        initial_mee = AstrodynamicsCore.rv2mee(initial_rv, parameters.mus[1])
        ts = range(0.0, 0.25, length = 6)
        sol_rv = solve(ODEProblem(eom, initial_rv, (ts[1], ts[end]), parameters),
            Vern9(), reltol = 1e-13, abstol = 1e-13, saveat = ts)
        sol_mee = solve(ODEProblem(gve, initial_mee, (ts[1], ts[end]), parameters),
            Vern9(), reltol = 1e-13, abstol = 1e-13, saveat = ts)
        position_errors = Float64[]
        velocity_errors = Float64[]
        for t in ts
            rv_mee = AstrodynamicsCore.mee2rv(sol_mee(t), parameters.mus[1])
            push!(position_errors, norm(rv_mee[1:3] - sol_rv(t)[1:3]))
            push!(velocity_errors, norm(rv_mee[4:6] - sol_rv(t)[4:6]))
        end
        @test maximum(position_errors) < 2e-10
        @test maximum(velocity_errors) < 2e-10
    end

    sol_rv = solve(ODEProblem(HighFidelityEphemerisModel.eom_Nbody!, rv0,
        (0.0, 0.25), params), Vern9(), reltol = 1e-13, abstol = 1e-13)
    sol_kep = solve(ODEProblem(HighFidelityEphemerisModel.gve_kep_Nbody!, kep0,
        (0.0, 0.25), params), Vern9(), reltol = 1e-13, abstol = 1e-13)
    @test AstrodynamicsCore.kep2rv(sol_kep.u[end], params.mus[1]) ≈ sol_rv.u[end] atol = 2e-10

    dkep = zeros(6)
    @test_throws DomainError HighFidelityEphemerisModel.gve_kep_derivs!(
        dkep, [2.0, 0.0, 0.5, 0.0, 0.0, 0.0], zeros(3), 1.0)
    @test_throws DomainError HighFidelityEphemerisModel.gve_kep_derivs!(
        dkep, [2.0, 0.1, 0.0, 0.0, 0.0, 0.0], zeros(3), 1.0)
end