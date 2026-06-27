"""
Test Ephemerides.jl-based N-body dynamics
"""

using LinearAlgebra
using OrdinaryDiffEq
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


test_get_pos_ephemerides = function()
    # define parameters
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

    # Compare the preferred FrameTransformations-backed Ephemerides query
    # against SPICE. Ephemerides.jl reads stored SPK records; the frame system
    # performs point-chain concatenation for arbitrary target/center pairs.
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
    # Use Earth relative to the Earth-Moon barycenter because this pair is
    # directly available in de440.bsp through Ephemerides.jl.
    naif_ids = ["399", "301"]
    GMs = [1.0, 1.0e-6]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 1.0

    et0 = 0.0
    parameters_spice = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
    )
    parameters_ephem = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        ephemerides_files = _ephemerides_test_spk(),
    )

    # initial state (in canonical scale)
    x0 = [1.0, 0.1, 0.2, 0.0, 0.3, -0.1]

    # compare static and in-place APIs
    dx_ephem = HighFidelityEphemerisModel.eom_Nbody_Ephemerides(x0, parameters_ephem, 0.0)
    dx_ephem_inplace = zeros(6)
    HighFidelityEphemerisModel.eom_Nbody_Ephemerides!(dx_ephem_inplace, x0, parameters_ephem, 0.0)

    @test dx_ephem ≈ dx_ephem_inplace atol=1e-15

    # compare against SPICE backend
    dx_spice = HighFidelityEphemerisModel.eom_Nbody_SPICE(x0, parameters_spice, 0.0)
    @test dx_ephem ≈ dx_spice atol=1e-15
end


test_eom_stm_Nbody_Ephemerides = function(;verbose::Bool = false)
    # define parameters
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

    # initial state (in canonical scale)
    x0 = [1.0, 0.1, 0.2, 0.0, 0.3, -0.1]
    x0_stm = [x0; reshape(I(6),36)]

    # evaluate Jacobian
    # Rs_initial = copy(parameters.Rs)
    # R_sun_initial = copy(parameters.R_sun)
    # jac_analytical = HighFidelityEphemerisModel.dfdx_Nbody_Ephemerides(x0, 0.0, parameters, 0.0)
    # @test parameters.Rs == Rs_initial
    # @test parameters.R_sun == R_sun_initial

    jac_ephem = HighFidelityEphemerisModel.eom_jacobian_fd(
        HighFidelityEphemerisModel.eom_Nbody_Ephemerides, x0, 0.0, parameters, 0.0
    )

    jac_spice = HighFidelityEphemerisModel.eom_jacobian_fd(
        HighFidelityEphemerisModel.eom_Nbody_SPICE, x0, 0.0, parameters, 0.0
    )

    if verbose
        println("AD-SPICE Jacobian:")
        print_matrix(jac_spice)
        println()
        println("AD-Ephemerides Jacobian:")
        print_matrix(jac_ephem)
        println()
    end

    @test jac_spice ≈ jac_ephem atol=1e-15

    # STM propagation
    sol_ephem = solve(
        ODEProblem(HighFidelityEphemerisModel.eom_Nbody_Ephemerides!, x0_stm, (0.0, 1.0), parameters),
        Vern7(), reltol=1e-12, abstol=1e-12
    )
    sol_spice = solve(
        ODEProblem(HighFidelityEphemerisModel.eom_Nbody_SPICE!, x0_stm, (0.0, 1.0), parameters),
        Vern7(), reltol=1e-12, abstol=1e-12
    )
    @test sol_ephem.u[end] ≈ sol_spice.u[end] atol=1e-15
end


test_Nbody_Ephemerides_ensemble = function()
    # define parameters
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

    # initial state (in canonical scale)
    x0 = [1.0, 0.1, 0.2, 0.0, 0.3, -0.1]

    # time span (in canonical scale)
    tspan = (0.0, 0.1)

    # compare threaded ensemble propagation against serial propagation
    N_traj = 4
    x0_conditions = [
        x0 .+ 1e-4 .* [i, -i, 0.5*i, 0.0, 0.0, 0.0]
        for i in 1:N_traj
    ]

    function prob_func_Nbody(ode_problem, i, repeat)
        remake(ode_problem, u0=x0_conditions[i])
    end

    prob_base = ODEProblem(HighFidelityEphemerisModel.eom_Nbody_Ephemerides!, x0, tspan, parameters)
    ensemble_prob = EnsembleProblem(
        prob_base;
        prob_func = prob_func_Nbody
    )

    sols_ensemble = solve(ensemble_prob, Vern7(), EnsembleThreads();
        trajectories=N_traj, reltol=1e-12, abstol=1e-12)

    sols_serial = []
    for i = 1:N_traj
        prob = ODEProblem(HighFidelityEphemerisModel.eom_Nbody_Ephemerides!, x0_conditions[i], tspan, parameters)
        sol = solve(prob, Vern7(), reltol=1e-12, abstol=1e-12)
        push!(sols_serial, sol)
    end

    for (sol_ensemble, sol_serial) in zip(sols_ensemble, sols_serial)
        @test sol_ensemble.u[end] ≈ sol_serial.u[end] atol=1e-12
    end
end


test_get_pos_ephemerides()
test_eom_Nbody_Ephemerides()
test_eom_stm_Nbody_Ephemerides(verbose=verbose)
test_Nbody_Ephemerides_ensemble()