"""Test shared-domain and formulation-domain GVE propagation."""

using AstrodynamicsCore
using LinearAlgebra
using OrdinaryDiffEq
using SPICE
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end


function _gve_validation_ephemerides_files()
    if haskey(ENV, "SPICE")
        spice_dir = ENV["SPICE"]
        return [
            joinpath(spice_dir, "spk", "de440.bsp"),
            joinpath(spice_dir, "pck", "moon_pa_de440_200625.bpc"),
        ]
    end

    spice_dir = joinpath(@__DIR__, "../spice/test")
    return [
        joinpath(spice_dir, "de440.bsp"),
        joinpath(spice_dir, "moon_pa_de440_200625.bpc"),
    ]
end


function _gve_cartesian_residuals(sol_rv, sol_elements, elements2rv, mu, ts)
    position_errors = Float64[]
    velocity_errors = Float64[]
    for t in ts
        rv = elements2rv(sol_elements(t), mu)
        push!(position_errors, norm(rv[1:3] - sol_rv(t)[1:3]))
        push!(velocity_errors, norm(rv[4:6] - sol_rv(t)[4:6]))
    end
    return position_errors, velocity_errors
end


@testset "Shared elliptic lunar N-body+SH validation" begin
    naif_ids = ["301", "399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    et0 = str2et("2020-01-01T00:00:00")
    DU = 3000.0
    filepath_SH = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab")
    params = HighFidelityEphemerisModel.SpiceParameters(
        et0, DU, GMs, naif_ids, "J2000", "NONE";
        filepath_spherical_harmonics = filepath_SH, nmax = 4, frame_PCPF = "MOON_PA")

    kep0 = [
        1900.0 / DU,
        0.03,
        deg2rad(45.0),
        deg2rad(35.0),
        deg2rad(55.0),
        deg2rad(25.0),
    ]
    mu = params.mus[1]
    rv0 = AstrodynamicsCore.kep2rv(kep0, mu)
    mee0 = AstrodynamicsCore.rv2mee(rv0, mu)
    eq0 = AstrodynamicsCore.rv2eq(rv0, mu)
    period = 2pi * sqrt(kep0[1]^3 / mu)
    ts = range(0.0, period, length = 61)
    tspan = (ts[1], ts[end])

    params_point_mass = HighFidelityEphemerisModel.SpiceParameters(
        et0, DU, GMs, naif_ids, "J2000", "NONE")
    acceleration_SH = HighFidelityEphemerisModel.eom_NbodySH(rv0, params, 0.0)[4:6]
    acceleration_point_mass =
        HighFidelityEphemerisModel.eom_Nbody(rv0, params_point_mass, 0.0)[4:6]
    @test norm(acceleration_SH - acceleration_point_mass) > 1e-6

    sol_rv = solve(ODEProblem(HighFidelityEphemerisModel.eom_NbodySH!, rv0,
        tspan, params), Vern9(), reltol = 1e-12, abstol = 1e-12, saveat = ts)
    @test sol_rv.retcode == SciMLBase.ReturnCode.Success
    @test all(isfinite, Array(sol_rv))

    for (gve!, gve, elements0, elements2rv) in (
        (HighFidelityEphemerisModel.gve_mee_NbodySH!,
            HighFidelityEphemerisModel.gve_mee_NbodySH, mee0, AstrodynamicsCore.mee2rv),
        (HighFidelityEphemerisModel.gve_kep_NbodySH!,
            HighFidelityEphemerisModel.gve_kep_NbodySH, kep0, AstrodynamicsCore.kep2rv),
        (HighFidelityEphemerisModel.gve_eq_NbodySH!,
            HighFidelityEphemerisModel.gve_eq_NbodySH, eq0, AstrodynamicsCore.eq2rv),
    )
        derivatives = similar(elements0)
        gve!(derivatives, elements0, params, 0.0)
        @test derivatives ≈ gve(elements0, params, 0.0) rtol = 2e-14 atol = 2e-14

        sol = solve(ODEProblem(gve!, elements0, tspan, params), Vern9(),
            reltol = 1e-12, abstol = 1e-12, saveat = ts)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        @test all(isfinite, Array(sol))
        position_errors, velocity_errors =
            _gve_cartesian_residuals(sol_rv, sol, elements2rv, mu, ts)
        @test position_errors[end] < 2e-10
        @test velocity_errors[end] < 2e-10
        @test maximum(position_errors) < 2e-10
        @test maximum(velocity_errors) < 2e-10
    end
end


@testset "L2 1072 formulation domains" begin
    ephemerides_files = _gve_validation_ephemerides_files()
    all(isfile, ephemerides_files) || error("Required GVE validation kernels are missing")
    naif_ids = ["301", "399", "10", "199", "299", "4", "5"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    et0 = utc2et("2026-01-21T00:00:00")
    params = HighFidelityEphemerisModel.EphemeridesParameters(
        et0, 1e4, GMs, naif_ids, "J2000", "NONE";
        filepath_spherical_harmonics =
            joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab"),
        nmax = 4,
        frame_PCPF = "MOON_PA",
        ephemerides_files,
    )
    rv0 = [
        3.854431517450734,
        -2.351585726041779,
        -1.1229515075875247,
        -0.1242026636081911,
        -0.4676313543570961,
        0.552957587746057,
    ]
    mu = params.mus[1]
    kep0 = AstrodynamicsCore.rv2kep(rv0, mu)
    mee0 = AstrodynamicsCore.rv2mee(rv0, mu)
    @test kep0[2] ≈ 1.5118414516694487 rtol = 1e-13
    @test all(isfinite, mee0)
    @test_throws DomainError AstrodynamicsCore.rv2eq(rv0, mu)

    period_SI = 2.2195106026355695e1 * 86400
    tspan = (0.0, 3period_SI / params.TU)
    ts = range(tspan[1], tspan[2], length = 401)
    sol_rv = solve(ODEProblem(HighFidelityEphemerisModel.eom_NbodySH!, rv0,
        tspan, params), Vern8(), reltol = 1e-12, abstol = fill(1e-14, 6),
        saveat = ts, maxiters = 100_000)
    sol_mee = solve(ODEProblem(HighFidelityEphemerisModel.gve_mee_NbodySH!, mee0,
        tspan, params), Vern8(), reltol = 1e-12,
        abstol = [1e-13, 2e-14, 2e-14, 2e-14, 2e-14, 1e-13],
        saveat = ts, maxiters = 100_000)
    @test sol_rv.retcode == SciMLBase.ReturnCode.Success
    @test sol_mee.retcode == SciMLBase.ReturnCode.Success
    @test all(isfinite, Array(sol_rv))
    @test all(isfinite, Array(sol_mee))
    position_errors, velocity_errors =
        _gve_cartesian_residuals(sol_rv, sol_mee, AstrodynamicsCore.mee2rv, mu, ts)
    @test position_errors[end] * params.DU < 1e-5
    @test velocity_errors[end] * params.VU < 2e-11
    # Allow small cross-platform/adaptive-step variation over the long L2 propagation.
    @test maximum(position_errors) * params.DU < 5e-5
    @test maximum(velocity_errors) * params.VU < 5e-11

    callback = ContinuousCallback(
        (u, t, integrator) -> min(u[2], abs(u[2] - 1), abs(sin(u[3]))) - 1e-3,
        terminate!; save_positions = (true, true))
    sol_kep = solve(ODEProblem(HighFidelityEphemerisModel.gve_kep_NbodySH!, kep0,
        tspan, params), Vern8(), reltol = 1e-12,
        abstol = [1e-12, 2e-14, 2e-13, 2e-12, 2e-12, 2e-12],
        callback = callback, maxiters = 100_000)
    @test sol_kep.retcode == SciMLBase.ReturnCode.Terminated
    @test all(isfinite, Array(sol_kep))
    @test 2.0 < sol_kep.t[end] * params.TU / 86400 < 2.3
    @test sol_kep.u[end][2] ≈ 1.001 atol = 1e-10
end