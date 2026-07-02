"""
Test Ephemerides.jl/FrameTransformations-based N-body dynamics.
"""

using LinearAlgebra
using OrdinaryDiffEq
using Random
using SPICE
using Test

if !@isdefined(furnsh_kernels)
    include(joinpath(@__DIR__, "utils.jl"))
end

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
    furnsh_kernels()
end

if !@isdefined(verbose)
    verbose = false
end


function _ephemerides_test_spk()
    if haskey(ENV, "SPICE")
        return joinpath(ENV["SPICE"], "spk", "de440.bsp")
    else
        return joinpath(@__DIR__, "../spice/test/de440.bsp")
    end
end


function _nbody_spice_ephemerides_parameters()
    naif_ids = ["301", "399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 1e5
    et0 = str2et("2026-01-05T00:00:00")

    parameters_spice = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        include_srp = true,
    )
    parameters_ephem = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        include_srp = true,
        ephemerides_files = _ephemerides_test_spk(),
    )

    return parameters_spice, parameters_ephem
end


test_get_pos_ephemerides = function()
    naif_ids = ["399", "301"]
    GMs = [1.0, 1.0e-6]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 1.0
    et0 = 0.0

    parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        ephemerides_files = _ephemerides_test_spk(),
    )

    r_ephem = HighFidelityEphemerisModel.get_pos_ephemerides(
        parameters.ephemerides_frame_system,
        naif_ids[2],
        naif_ids[1],
        et0;
        axes = naif_frame,
    )
    r_spice, _ = spkpos(naif_ids[2], et0, naif_frame, abcorr, naif_ids[1])

    @test length(collect(r_ephem)) == 3
    @test isapprox(collect(r_ephem), r_spice; atol = 1e-10, rtol = 5e-16)
end


test_eom_Nbody_Ephemerides = function()
    parameters_spice, parameters_ephem = _nbody_spice_ephemerides_parameters()

    x0 = [1.0, 0.1, 0.2, 0.0, 0.3, -0.1]

    dx_ephem = HighFidelityEphemerisModel.eom_Nbody_Ephemerides(x0, parameters_ephem, 0.0)
    dx_ephem_inplace = zeros(6)
    HighFidelityEphemerisModel.eom_Nbody_Ephemerides!(dx_ephem_inplace, x0, parameters_ephem, 0.0)

    @test dx_ephem ≈ dx_ephem_inplace atol=1e-15

    dx_spice = HighFidelityEphemerisModel.eom_Nbody_SPICE(x0, parameters_spice, 0.0)
    @test dx_ephem ≈ dx_spice atol=1e-14
end


test_eom_stm_Nbody_Ephemerides = function(;verbose::Bool = false)
    parameters_spice, parameters_ephem = _nbody_spice_ephemerides_parameters()

    x0 = [1.0, 0.1, 0.2, 0.0, 0.3, -0.1]
    x0_stm = [x0; reshape(I(6),36)]

    jac_ephem = HighFidelityEphemerisModel.eom_jacobian_fd(
        HighFidelityEphemerisModel.eom_Nbody_Ephemerides, x0, 0.0, parameters_ephem, 0.0
    )
    jac_spice = HighFidelityEphemerisModel.eom_jacobian_fd(
        HighFidelityEphemerisModel.eom_Nbody_SPICE, x0, 0.0, parameters_spice, 0.0
    )

    if verbose
        println("AD-SPICE Jacobian:")
        print_matrix(jac_spice)
        println()
        println("AD-Ephemerides Jacobian:")
        print_matrix(jac_ephem)
        println()
    end

    @test jac_spice ≈ jac_ephem atol=1e-14

    tspan = (0.0, 0.1)
    saveat = range(tspan[1], tspan[2]; length = 5)

    sol_ephem = solve(
        ODEProblem(HighFidelityEphemerisModel.eom_stm_Nbody_Ephemerides_fd!, x0_stm, tspan, parameters_ephem),
        Vern7(); reltol=1e-12, abstol=1e-12, saveat=saveat
    )
    sol_spice = solve(
        ODEProblem(HighFidelityEphemerisModel.eom_stm_Nbody_SPICE_fd!, x0_stm, tspan, parameters_spice),
        Vern7(); reltol=1e-12, abstol=1e-12, saveat=saveat
    )

    @test maximum(abs.(Array(sol_ephem)[1:6,:] .- Array(sol_spice)[1:6,:])) < 1e-12
    @test maximum(abs.(Array(sol_ephem)[7:42,:] .- Array(sol_spice)[7:42,:])) < 1e-12
end


test_Nbody_Ephemerides_ensemble = function()
    _, parameters = _nbody_spice_ephemerides_parameters()

    x0 = [1.0, 0.1, 0.2, 0.0, 0.3, -0.1]
    tspan = (0.0, 0.1)

    N_traj = 4
    x0_conditions = [
        x0 .+ 1e-4 .* [i, -i, 0.5*i, 0.0, 0.0, 0.0]
        for i in 1:N_traj
    ]

    function prob_func_Nbody(ode_problem, i, repeat)
        remake(ode_problem, u0=x0_conditions[i])
    end

    prob_base = ODEProblem(HighFidelityEphemerisModel.eom_Nbody_Ephemerides!, x0, tspan, parameters)
    ensemble_prob = EnsembleProblem(prob_base; prob_func = prob_func_Nbody)

    sols_ensemble = solve(ensemble_prob, Vern7(), EnsembleThreads();
        trajectories=N_traj, reltol=1e-12, abstol=1e-12)

    sols_serial = []
    for i = 1:N_traj
        prob = ODEProblem(HighFidelityEphemerisModel.eom_Nbody_Ephemerides!, x0_conditions[i], tspan, parameters)
        sol = solve(prob, Vern7(); reltol=1e-12, abstol=1e-12)
        push!(sols_serial, sol)
    end

    for (sol_ensemble, sol_serial) in zip(sols_ensemble, sols_serial)
        @test sol_ensemble.u[end] ≈ sol_serial.u[end] atol=1e-12
    end
end


function test_random_Nbody_SPICE_vs_Ephemerides_integrations(; N::Int = parse(Int, get(ENV, "HFEM_EPHEMERIDES_RANDOM_TEST_N", "10")))
    params_spice, params_ephem = _nbody_spice_ephemerides_parameters()

    rng = MersenneTwister(20260628)
    x_nom = [1.0, 0.05, 0.2, 0.0, 0.25, -0.08]

    for _ in 1:N
        x0 = x_nom .+ [1e-3 .* randn(rng, 3); 1e-4 .* randn(rng, 3)]
        tf = 0.05 + 0.20 * rand(rng)  # short, random tspan in TU
        tspan = (0.0, tf)
        saveat = range(tspan[1], tspan[2]; length = 6)

        sol_spice = solve(
            ODEProblem(HighFidelityEphemerisModel.eom_Nbody_SPICE!, x0, tspan, params_spice),
            Vern7(); reltol=1e-12, abstol=1e-12, saveat=saveat
        )
        sol_ephem = solve(
            ODEProblem(HighFidelityEphemerisModel.eom_Nbody_Ephemerides!, x0, tspan, params_ephem),
            Vern7(); reltol=1e-12, abstol=1e-12, saveat=saveat
        )

        @test maximum(abs.(Array(sol_ephem) .- Array(sol_spice))) < 1e-12

        x0_stm = [x0; reshape(I(6),36)]
        sol_spice_stm = solve(
            ODEProblem(HighFidelityEphemerisModel.eom_stm_Nbody_SPICE_fd!, x0_stm, tspan, params_spice),
            Vern7(); reltol=1e-11, abstol=1e-11, saveat=saveat
        )
        sol_ephem_stm = solve(
            ODEProblem(HighFidelityEphemerisModel.eom_stm_Nbody_Ephemerides_fd!, x0_stm, tspan, params_ephem),
            Vern7(); reltol=1e-11, abstol=1e-11, saveat=saveat
        )

        @test maximum(abs.(Array(sol_ephem_stm)[1:6,:] .- Array(sol_spice_stm)[1:6,:])) < 1e-12
        @test maximum(abs.(Array(sol_ephem_stm)[7:42,:] .- Array(sol_spice_stm)[7:42,:])) < 1e-12
    end
end


test_get_pos_ephemerides()
test_eom_Nbody_Ephemerides()
test_eom_stm_Nbody_Ephemerides(verbose=verbose)
test_Nbody_Ephemerides_ensemble()
test_random_Nbody_SPICE_vs_Ephemerides_integrations()


function ephemerides_test_moon_pa_bpc()
    if haskey(ENV, "SPICE")
        return joinpath(ENV["SPICE"], "pck", "moon_pa_de440_200625.bpc")
    else
        return joinpath(@__DIR__, "../spice/test/moon_pa_de440_200625.bpc")
    end
end


function nbody_ephemerides_drag_constant_density(et, r_km)
    return 1e-12
end


function nbody_drag_spice_ephemerides_parameters()
    naif_ids = ["301", "399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 1e5
    et0 = str2et("2026-01-05T00:00:00")

    kwargs = (
        frame_PCPF = "MOON_PA",
        include_srp = true,
        include_drag = true,
        f_density = nbody_ephemerides_drag_constant_density,
    )

    parameters_spice = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        kwargs...,
    )
    parameters_ephem = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        kwargs...,
        ephemerides_files = [_ephemerides_test_spk(), ephemerides_test_moon_pa_bpc()],
    )

    return parameters_spice, parameters_ephem
end


test_eom_Nbody_Ephemerides_drag = function()
    parameters_spice, parameters_ephem = nbody_drag_spice_ephemerides_parameters()

    x0 = [1.0, 0.0, 0.3, 0.5, 1.0, 0.0]
    t = 0.1

    dx_ephem = HighFidelityEphemerisModel.eom_Nbody_Ephemerides(x0, parameters_ephem, t)
    dx_ephem_inplace = zeros(6)
    HighFidelityEphemerisModel.eom_Nbody_Ephemerides!(dx_ephem_inplace, x0, parameters_ephem, t)

    @test dx_ephem ≈ dx_ephem_inplace atol=1e-14

    dx_spice = HighFidelityEphemerisModel.eom_Nbody_SPICE(x0, parameters_spice, t)
    @test dx_ephem ≈ dx_spice atol=1e-14
end


function test_random_Nbody_drag_SPICE_vs_Ephemerides_integrations(; N::Int = parse(Int, get(ENV, "HFEM_EPHEMERIDES_RANDOM_TEST_N", "10")))
    params_spice, params_ephem = nbody_drag_spice_ephemerides_parameters()

    rng = MersenneTwister(20260628)
    x_nom = [1.0, 0.05, 0.2, 0.0, 0.25, -0.08]

    for _ in 1:N
        x0 = x_nom .+ [1e-3 .* randn(rng, 3); 1e-4 .* randn(rng, 3)]
        tf = 0.02 + 0.08 * rand(rng)  # short, random tspan in TU
        tspan = (0.0, tf)
        saveat = range(tspan[1], tspan[2]; length = 6)

        sol_spice = solve(
            ODEProblem(HighFidelityEphemerisModel.eom_Nbody_SPICE!, x0, tspan, params_spice),
            Vern7(); reltol=1e-12, abstol=1e-12, saveat=saveat
        )
        sol_ephem = solve(
            ODEProblem(HighFidelityEphemerisModel.eom_Nbody_Ephemerides!, x0, tspan, params_ephem),
            Vern7(); reltol=1e-12, abstol=1e-12, saveat=saveat
        )

        @test maximum(abs.(Array(sol_ephem) .- Array(sol_spice))) < 1e-12
    end
end


test_eom_Nbody_Ephemerides_drag()
test_random_Nbody_drag_SPICE_vs_Ephemerides_integrations()