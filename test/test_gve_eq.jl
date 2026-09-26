"""Test ordinary equinoctial GVE and direct MEE RTN projection."""

using AstrodynamicsCore
using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq
using Random
using SPICE
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end


@testset "Direct MEE RTN projection" begin
    rng = MersenneTwister(42)
    acceleration = [0.3, -0.2, 0.7]
    keplerian_cases = [
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.3],
        [2.0, 0.0, 0.7, 0.4, 0.0, 1.2],
        [3.0, 0.4, 0.8, 1.0, 0.5, 2.4],
        [8.0, 0.85, 0.3, 2.0, 1.1, 0.2],
        [4.0, 0.1, deg2rad(170.0), 0.4, 0.2, 0.5],
    ]

    for _ in 1:50
        push!(keplerian_cases, [
            1.0 + 4.0 * rand(rng),
            0.9 * rand(rng),
            deg2rad(175.0) * rand(rng),
            2pi * rand(rng),
            2pi * rand(rng),
            2pi * rand(rng),
        ])
    end

    for kep in keplerian_cases
        mee = AstrodynamicsCore.kep2mee(kep)
        rv = AstrodynamicsCore.mee2rv(mee, 1.0)
        projected_cartesian = zeros(3)
        projected_mee = zeros(3)
        HighFidelityEphemerisModel.project_inr_to_rtn!(
            projected_cartesian, acceleration, view(rv, 1:3), view(rv, 4:6))
        HighFidelityEphemerisModel._project_inr_to_rtn_mee!(
            projected_mee, acceleration, mee)
        @test isapprox(projected_mee, projected_cartesian; rtol = 2e-14, atol = 2e-14)
    end

    mee = AstrodynamicsCore.kep2mee([3.0, 0.2, 0.7, 0.4, 0.5, 1.2])
    project_mee = function(x)
        projected = similar(x, 3)
        HighFidelityEphemerisModel._project_inr_to_rtn_mee!(projected, acceleration, x)
        return projected
    end
    @test all(isfinite, ForwardDiff.jacobian(project_mee, mee))
end


@testset "Ordinary equinoctial derivatives" begin
    eq = [2.0, 0.1, -0.05, 0.2, -0.1, 0.7]
    deq = zeros(6)
    HighFidelityEphemerisModel.gve_eq_derivs!(deq, eq, zeros(3), 1.0)
    @test isapprox(deq, [0.0, 0.0, 0.0, 0.0, 0.0, sqrt(1.0 / eq[1]^3)];
        rtol = 2e-14, atol = 2e-14)

    for acceleration in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        mee = AstrodynamicsCore.eq2mee(eq)
        dmee = zeros(6)
        HighFidelityEphemerisModel.gve_mee_derivs!(dmee, mee, acceleration, 1.0)
        reference = ForwardDiff.jacobian(AstrodynamicsCore.mee2eq, mee) * dmee
        HighFidelityEphemerisModel.gve_eq_derivs!(deq, eq, acceleration, 1.0)
        @test isapprox(deq, reference; rtol = 2e-12, atol = 2e-12)
    end

    circular_eq = [2.0, 0.0, 0.0, 0.0, 0.0, 0.3]
    HighFidelityEphemerisModel.gve_eq_derivs!(
        deq, circular_eq, [0.01, -0.02, 0.03], 1.0)
    @test all(isfinite, deq)

    f_eq = function(x)
        dx = similar(x)
        HighFidelityEphemerisModel.gve_eq_derivs!(dx, x, [0.01, -0.02, 0.03], 1.0)
        return dx
    end
    @test all(isfinite, ForwardDiff.jacobian(f_eq, eq))
    @test_throws DomainError HighFidelityEphemerisModel.gve_eq_derivs!(
        deq, [-2.0, eq[2:end]...], zeros(3), 1.0)
    @test_throws DomainError HighFidelityEphemerisModel.gve_eq_derivs!(
        deq, [2.0, 1.0, 0.0, 0.0, 0.0, 0.0], zeros(3), 1.0)
end


@testset "Ordinary equinoctial two-body propagation" begin
    params = HighFidelityEphemerisModel.SpiceParameters(
        0.0, 1.0, [1.0], ["399"], "J2000", "NONE")
    keplerian_cases = [
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.3],
        [2.0, 0.0, 0.6, 0.4, 0.0, 1.2],
        [2.5, 0.2, 1e-6, 0.2, 0.7, 2.1],
        [3.0, 0.4, 0.8, 1.0, 0.5, 2.4],
        [8.0, 0.85, 0.3, 2.0, 1.1, 0.2],
        [4.0, 0.1, deg2rad(170.0), 0.4, 0.2, 0.5],
    ]

    for kep0 in keplerian_cases
        rv0 = AstrodynamicsCore.kep2rv(kep0, 1.0)
        eq0 = AstrodynamicsCore.kep2eq(kep0)
        ts = range(0.0, 4.0, length = 21)
        sol_rv = solve(ODEProblem(HighFidelityEphemerisModel.eom_Nbody!, rv0,
            (ts[1], ts[end]), params), Vern9(), reltol = 1e-13, abstol = 1e-13,
            saveat = ts)
        sol_eq = solve(ODEProblem(HighFidelityEphemerisModel.gve_eq_Nbody!, eq0,
            (ts[1], ts[end]), params), Vern9(), reltol = 1e-13, abstol = 1e-13,
            saveat = ts)
        errors = [norm(AstrodynamicsCore.eq2rv(sol_eq(t), 1.0) - sol_rv(t)) for t in ts]
        @test maximum(errors) < 2e-10
    end

    eq0 = AstrodynamicsCore.kep2eq([2.0, 0.2, 0.5, 0.3, 0.7, 1.1])
    rv0 = AstrodynamicsCore.eq2rv(eq0, 1.0)
    period = 2pi * sqrt(eq0[1]^3)
    ts = range(0.0, 3period, length = 61)
    sol_rv = solve(ODEProblem(HighFidelityEphemerisModel.eom_Nbody!, rv0,
        (ts[1], ts[end]), params), Vern9(), reltol = 1e-12, abstol = 1e-12,
        saveat = ts)
    sol_eq = solve(ODEProblem(HighFidelityEphemerisModel.gve_eq_Nbody!, eq0,
        (ts[1], ts[end]), params), Vern9(), reltol = 1e-12, abstol = 1e-12,
        saveat = ts)
    errors = [norm(AstrodynamicsCore.eq2rv(sol_eq(t), 1.0) - sol_rv(t)) for t in ts]
    @test maximum(errors) < 2e-9
end


@testset "Ordinary equinoctial backend dispatch" begin
    eq = [2.0, 0.1, -0.05, 0.2, -0.1, 0.7]
    params_SPICE = HighFidelityEphemerisModel.SpiceParameters(
        0.0, 1.0, [1.0], ["399"], "J2000", "NONE")
    params_Interp = HighFidelityEphemerisModel.InterpParameters(
        0.0, 1.0, [1.0], ["399"], "J2000", "NONE";
        interpolate_ephem_span = [0.0, 7200.0])
    params_Ephemerides = HighFidelityEphemerisModel.EphemeridesParameters(
        0.0, 1.0, [1.0], ["399"], "J2000", "NONE";
        ephemerides_frame_system = :test_frame_system)
    expected = zeros(6)
    HighFidelityEphemerisModel.gve_eq_derivs!(expected, eq, zeros(3), 1.0)

    for params in (params_SPICE, params_Interp, params_Ephemerides)
        @test isapprox(HighFidelityEphemerisModel.gve_eq_Nbody(eq, params, 0.0), expected)
    end
    for params in (params_Interp, params_Ephemerides)
        @test isapprox(HighFidelityEphemerisModel.gve_eq_NbodySH(eq, params, 0.0), expected)
    end
end


@testset "Ordinary equinoctial force models" begin
    naif_ids = ["301", "399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    et0 = str2et("2020-01-01T00:00:00")
    filepath_SH = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab")
    params = HighFidelityEphemerisModel.SpiceParameters(
        et0, 3000.0, GMs, naif_ids, "J2000", "NONE")
    params_SH = HighFidelityEphemerisModel.SpiceParameters(
        et0, 3000.0, GMs, naif_ids, "J2000", "NONE";
        filepath_spherical_harmonics = filepath_SH, nmax = 4, frame_PCPF = "MOON_PA")
    rv_cases = [
        [1.0, 0.0, 0.3, 0.5, 1.0, 0.0],
        AstrodynamicsCore.kep2rv([1.5, 0.01, 1e-6, 0.2, 0.4, 0.6], 1.0),
        AstrodynamicsCore.kep2rv([2.0, 0.3, 0.6, 1.0, 0.5, 2.0], 1.0),
        AstrodynamicsCore.kep2rv([3.0, 0.8, deg2rad(170.0), 0.4, 0.2, 0.5], 1.0),
    ]

    for (eom, gve, parameters) in (
        (HighFidelityEphemerisModel.eom_Nbody!, HighFidelityEphemerisModel.gve_eq_Nbody!, params),
        (HighFidelityEphemerisModel.eom_NbodySH!, HighFidelityEphemerisModel.gve_eq_NbodySH!, params_SH),
    ), initial_rv in rv_cases
        initial_eq = AstrodynamicsCore.rv2eq(initial_rv, parameters.mus[1])
        ts = range(0.0, 0.1, length = 6)
        sol_rv = solve(ODEProblem(eom, initial_rv, (ts[1], ts[end]), parameters),
            Vern9(), reltol = 1e-13, abstol = 1e-13, saveat = ts)
        sol_eq = solve(ODEProblem(gve, initial_eq, (ts[1], ts[end]), parameters),
            Vern9(), reltol = 1e-13, abstol = 1e-13, saveat = ts)
        position_errors = Float64[]
        velocity_errors = Float64[]
        for t in ts
            rv_eq = AstrodynamicsCore.eq2rv(sol_eq(t), parameters.mus[1])
            push!(position_errors, norm(rv_eq[1:3] - sol_rv(t)[1:3]))
            push!(velocity_errors, norm(rv_eq[4:6] - sol_rv(t)[4:6]))
        end
        @test maximum(position_errors) < 2e-10
        @test maximum(velocity_errors) < 2e-10
    end
end