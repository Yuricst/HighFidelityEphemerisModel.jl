"""
Test integrating N-body dynamics with SPICE call within eom.
Uses low-level API
"""

using LinearAlgebra
using OrdinaryDiffEq
using SPICE
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end


test_eom_Nbody_SPICE = function()
    # define parameters
    naif_ids = ["301", "399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 3000.0

    et0 = str2et("2020-01-01T00:00:00")
    parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(et0, DU, GMs, naif_ids, naif_frame, abcorr)
    # @show parameters.DU, parameters.TU, parameters.VU
    # @show parameters.mus

    # initial state (in canonical scale)
    u0 = [1.0, 0.0, 0.3, 0.5, 1.0, 0.0]

    # time span (in canonical scale)
    tspan = (0.0, 7*86400/parameters.TU)

    # solve
    prob = ODEProblem(HighFidelityEphemerisModel.eom_Nbody_SPICE!, u0, tspan, parameters)
    sol = solve(prob, Vern7(), reltol=1e-12, abstol=1e-12)
    u_check = [0.5223145338552279, 2.0961454012986276, -0.16366028913053066,
              -0.4093613718782754, 0.2538623882288259, -0.1600564978501581]
    @test norm(sol.u[end] - u_check) < 1e-11

    # now also include SRP
    parameters.include_srp = true
    prob = ODEProblem(HighFidelityEphemerisModel.eom_Nbody_SPICE!, u0, tspan, parameters)
    sol = solve(prob, Vern7(), reltol=1e-12, abstol=1e-12)
    u_check = [0.5223136131171607, 2.09613604449625, -0.16366138825154172,
              -0.40936218196868585, 0.25386900598607015, -0.160056371691089]
    # @show sol.u[end]
    # @show norm(sol.u[end] - u_check)
    @test norm(sol.u[end] - u_check) < 1e-11
end


test_eom_stm_Nbody_SPICE = function(;verbose::Bool = false)
    # define parameters
    naif_ids = ["301", "399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 3000.0

    et0 = str2et("2020-01-01T00:00:00")
    parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        include_srp = true,
    )
    # @show parameters.DU, parameters.TU, parameters.VU
    # @show parameters

    # initial state (in canonical scale)
    x0 = [1.0, 0.0, 0.3, 0.5, 1.0, 0.0]
    x0_stm = [x0; reshape(I(6),36)]

    # evaluate Jacobian
    jac_analytical = HighFidelityEphemerisModel.dfdx_Nbody_SPICE(x0, 0.0, parameters, 0.0)
    # @show jac_analytical

    f_eval = zeros(6)
    HighFidelityEphemerisModel.eom_Nbody_SPICE!(f_eval, x0, parameters, 0.0)
    jac_numerical = zeros(6,6)
    h = 1e-8
    for i = 1:6
        x0_copy = copy(x0)
        x0_copy[i] += h
        _f_eval = zeros(6)
        HighFidelityEphemerisModel.eom_Nbody_SPICE!(_f_eval, x0_copy, parameters, 0.0)
        jac_numerical[:,i] = (_f_eval - f_eval) / h
    end
    jac_numerical_fd = HighFidelityEphemerisModel.eom_jacobian_fd(
        HighFidelityEphemerisModel.eom_Nbody_SPICE, x0, 0.0, parameters, 0.0
    )
    if verbose
        println("Analytical Jacobian:")
        print_matrix(jac_analytical)
        println()
        println("Numerical Jacobian:")
        print_matrix(jac_numerical)
        println()
        println("ForwardDiff Jacobian:")
        print_matrix(jac_numerical_fd)
        println()
        println("Diff analytical - numerical:")
        print_matrix(jac_analytical - jac_numerical)
        println()
        println("Diff analytical - ForwardDiff:")
        print_matrix(jac_analytical - jac_numerical_fd)
    end
    @test maximum(abs.(jac_analytical - jac_numerical)) < 1e-6
    @test maximum(abs.(jac_analytical - jac_numerical_fd)) < 1e-12

    # time span (in canonical scale)
    tspan = (0.0, 1.0)

    # solve
    prob = ODEProblem(HighFidelityEphemerisModel.eom_stm_Nbody_SPICE!, x0_stm, tspan, parameters)
    sol = solve(prob, Vern7(), reltol=1e-12, abstol=1e-12)
    
    # construct STM
    STM_analytical = reshape(sol.u[end][7:42],6,6)'
    STM_numerical = zeros(6,6)
    h = 1e-8
    for i = 1:6
        x0_copy = copy(x0)
        x0_copy[i] += h
        sol_ptrb = solve(ODEProblem(HighFidelityEphemerisModel.eom_Nbody_SPICE!, x0_copy, tspan, parameters), Vern7(), reltol=1e-12, abstol=1e-12)
        STM_numerical[:,i] = (sol_ptrb.u[end][1:6] - sol.u[end][1:6]) / h
    end
    if verbose
        println("Analytical STM:")
        print_matrix(STM_analytical)
        println()
        println("Numerical STM:")
        print_matrix(STM_numerical)
        println()
        println("Diff:")
        print_matrix(STM_analytical - STM_numerical)
    end
    @test maximum(abs.(STM_analytical - STM_numerical)) < 1e-6
end


test_eom_Nbody_SPICE()
test_eom_stm_Nbody_SPICE(verbose = false)