"""
Test integrating N-body+SH dynamics with SPICE call within eom.
"""

using LinearAlgebra
using OrdinaryDiffEq
using SPICE
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end


test_NbodySH_Interp_ensemble = function(;verbose = false)
    # define parameters
    naif_ids = ["301", "399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 1e5

    filepath_spherical_harmonics = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab")
    nmax = 4

    et0 = str2et("2026-01-05T00:00:00")
    etf = et0 + 30 * 86400.0
    interpolate_ephem_span = [et0, etf]
    interpolation_time_step = 1000.0
    parameters = HighFidelityEphemerisModel.InterpParameters(et0, DU, GMs, naif_ids, naif_frame, abcorr;
        interpolate_ephem_span=interpolate_ephem_span,
        filepath_spherical_harmonics = filepath_spherical_harmonics,
        nmax = nmax,
        frame_PCPF = "MOON_PA",
        interpolation_time_step = interpolation_time_step,
    )

    # initial state (in canonical scale)
    x0_dim, _ = spkezr("-60000", et0, naif_frame, abcorr, naif_ids[1])
    x0 = [x0_dim[1:3]/parameters.DU; x0_dim[4:6]/parameters.VU]
    x0_stm = [x0; reshape(I(6),36)]

    # time span (in canonical scale)
    tspan = (0.0, 7*86400/parameters.TU)

    # ------------------------------------------------------------------------------------------------ #
    # first we compare ensemble vs. serial without STMs
    N_traj = 4
    x0_conditions = [
        [x0[1:3] + 10/parameters.DU * randn(3); x0[4:6] + 1e-3/parameters.VU * randn(3)]
        for _ in 1:N_traj
    ]
    function prob_func_Nbody(ode_problem, i, repeat)
        _x0 = x0_conditions[i]
        remake(ode_problem, u0=_x0)
    end

    # create ensemble problem
    prob_base = ODEProblem(HighFidelityEphemerisModel.eom_NbodySH!, x0, tspan, parameters)
    ensemble_prob = EnsembleProblem(
        prob_base;
        prob_func = prob_func_Nbody
    )

    # solve ensemble problem
    sols_ensemble = solve(ensemble_prob, Vern9(), EnsembleThreads();
        trajectories=N_traj, reltol=1e-14, abstol=1e-14)

    # solve in serial
    sols_serial = []
    for i = 1:N_traj
        prob = ODEProblem(HighFidelityEphemerisModel.eom_NbodySH!, x0_conditions[i], tspan, parameters)
        sol = solve(prob, Vern9(), reltol=1e-14, abstol=1e-14)
        push!(sols_serial, sol)
    end

    # check consistency between serial & parallel runs
    for (sol_ensemble, sol_serial) in zip(sols_ensemble, sols_serial)
        @test sol_ensemble.u[end] == sol_serial.u[end]
        #@test norm(sols_ensemble.u[i][end] - sols_serial[i].u[end]) < 1e-16
    end

    # ------------------------------------------------------------------------------------------------ #
    # now we do the same but with STMs
    function prob_func_Nbody_stm(ode_problem, i, repeat)
        _x0_stm = [x0_conditions[i]; reshape(I(6),36)]
        remake(ode_problem, u0=_x0_stm)
    end

    # create ensemble problem
    prob_base = ODEProblem(HighFidelityEphemerisModel.eom_stm_NbodySH_Interp_fd!, x0_stm, tspan, parameters)
    ensemble_prob = EnsembleProblem(
        prob_base;
        prob_func = prob_func_Nbody_stm
    )

    # solve ensemble problem
    sols_stm_ensemble = solve(ensemble_prob, Vern9(), EnsembleThreads();
        trajectories=N_traj, reltol=1e-14, abstol=1e-14)

    # solve in serial
    sols_stm_serial = []
    for i = 1:N_traj
        prob = ODEProblem(HighFidelityEphemerisModel.eom_stm_NbodySH_Interp_fd!, [x0_conditions[i]; reshape(I(6),36)], tspan, parameters)
        sol = solve(prob, Vern9(), reltol=1e-14, abstol=1e-14)
        push!(sols_stm_serial, sol)
    end

    # check consistency between serial & parallel runs
    for (sol_stm_ensemble, sol_stm_serial) in zip(sols_stm_ensemble, sols_stm_serial)
        @test sol_stm_ensemble.u[end] == sol_stm_serial.u[end]
        if verbose
            @show sol_stm_ensemble.u[end][1:6]
            @show sol_stm_serial.u[end][1:6]
            print_matrix(reshape(sol_stm_ensemble.u[end][7:end], (6,6))')
            println()
            print_matrix(reshape(sol_stm_serial.u[end][7:end], (6,6))')
            println()
        end
    end
end


test_NbodySH_Interp_ensemble(;verbose = false)