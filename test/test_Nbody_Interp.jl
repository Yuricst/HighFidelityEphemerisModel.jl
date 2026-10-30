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


test_eom_Nbody_Interp = function()
    # define parameters 
    naif_ids = ["301", "399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 3000.0

    et0 = str2et("2020-01-01T00:00:00")
    etf = et0 + 30 * 86400.0
    interpolate_ephem_span = [et0, etf]
    parameters = HighFidelityEphemerisModel.InterpParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        interpolate_ephem_span=interpolate_ephem_span,
        interpolation_time_step=100.0)
    # @show parameters.DU, parameters.TU, parameters.VU
    # @show parameters.mus
    # @show parameters.interpolated_ephems
    # @show parameters.interpolated_transformation

    # initial state (in canonical scale)
    u0 = [1.0, 0.0, 0.3, 0.5, 1.0, 0.0]

    # time span (in canonical scale)
    tspan = (0.0, 7*86400/parameters.TU)

    # solve
    prob = ODEProblem(HighFidelityEphemerisModel.eom_Nbody_Interp!, u0, tspan, parameters)
    sol = solve(prob, Vern7(), reltol=1e-12, abstol=1e-12)
    u_check = [0.5223145338552279, 2.0961454012986276, -0.16366028913053066,
              -0.4093613718782754, 0.2538623882288259, -0.1600564978501581]
    # @show sol.u[end]
    # @show norm(sol.u[end] - u_check)
    @test norm(sol.u[end] - u_check) < 1e-11

    # now also include SRP
    parameters.include_srp = true
    prob = ODEProblem(HighFidelityEphemerisModel.eom_Nbody_Interp!, u0, tspan, parameters)
    sol = solve(prob, Vern7(), reltol=1e-12, abstol=1e-12)
    u_check = [0.5223136131171607, 2.09613604449625, -0.16366138825154172,
              -0.40936218196868585, 0.25386900598607015, -0.160056371691089]
    # @show sol.u[end]
    # @show norm(sol.u[end] - u_check)
    @test norm(sol.u[end] - u_check) < 1e-11
end


test_eom_stm_Nbody_Interp = function(;verbose::Bool = false)
    # define parameters
    naif_ids = ["301", "399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 1e5

    et0 = str2et("2026-01-05T00:00:00")
    etf = et0 + 30 * 86400.0
    interpolate_ephem_span = [et0, etf]
    parameters = HighFidelityEphemerisModel.InterpParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        interpolate_ephem_span=interpolate_ephem_span,
        interpolation_time_step=100.0,
        include_srp = true,
    )

    # initial state (in canonical scale)
    x0_dim, _ = spkezr("-60000", et0, naif_frame, abcorr, naif_ids[1])
    x0 = [x0_dim[1:3]/parameters.DU; x0_dim[4:6]/parameters.VU]
    x0_stm = [x0; reshape(I(6),36)]

    # time span (in canonical scale)
    tspan = (0.0, 7*86400/parameters.TU)

    # solve just the state
    prob = ODEProblem(HighFidelityEphemerisModel.eom_Nbody_Interp!, x0, tspan, parameters)
    sol = solve(prob, Vern8(), reltol=1e-14, abstol=1e-14)
    @test sol.retcode == SciMLBase.ReturnCode.Success

    # solve with ForwardDiff Jacobian STM
    dx_stm = similar(x0_stm)
    HighFidelityEphemerisModel.eom_stm_Nbody_Interp_fd!(dx_stm, x0_stm, parameters, 0.0)

    prob_fd = ODEProblem(HighFidelityEphemerisModel.eom_stm_Nbody_Interp_fd!, x0_stm, tspan, parameters)
    sol_fd = solve(prob_fd, Vern8(), reltol=1e-14, abstol=1e-14)
    @test sol_fd.retcode == SciMLBase.ReturnCode.Success

    # plot trajectories
    if verbose
        fig = Figure(size=(800,800))
        ax3d = Axis3(fig[1,1]; aspect=:data)
        lines!(ax3d, Array(sol)[1,:], Array(sol)[2,:], Array(sol)[3,:], color=:blue)
        lines!(ax3d, Array(sol_fd)[1,:], Array(sol_fd)[2,:], Array(sol_fd)[3,:], color=:green)
        display(fig)
    end

    # compare integrated STM against finite-difference STM
    STM_fd = reshape(sol_fd.u[end][7:42], 6, 6)
    STM_numerical = zeros(6,6)
    h = 1e-7
    for i = 1:6
        x0_plus = copy(x0)
        x0_plus[i] += h
        sol_ptrb = solve(ODEProblem(HighFidelityEphemerisModel.eom_Nbody_Interp!, x0_plus, tspan, parameters), Vern7(), reltol=1e-12, abstol=1e-12)

        x0_min = copy(x0)
        x0_min[i] -= h
        sol_ptrb_min = solve(ODEProblem(HighFidelityEphemerisModel.eom_Nbody_Interp!, x0_min, tspan, parameters), Vern7(), reltol=1e-12, abstol=1e-12)

        STM_numerical[:,i] = (sol_ptrb.u[end][1:6] - sol_ptrb_min.u[end][1:6]) / (2*h)
    end
    if verbose
        println("ForwardDiff STM:")
        print_matrix(STM_fd)
        println()
        println("Numerical STM:")
        print_matrix(STM_numerical)
        println()
        println("Diff:")
        print_matrix(STM_fd - STM_numerical)
    end
    @test maximum(abs.(STM_fd - STM_numerical)) < 1e-5
end


test_eom_Nbody_Interp()
test_eom_stm_Nbody_Interp(verbose = false)