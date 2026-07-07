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


test_callback_trueanomaly = function()
    # define parameters 
    GMs = [
        4.9028000661637961E+03,
        3.9860043543609598E+05,
        1.3271244004193938E+11,
    ]
    naif_ids = ["301", "399", "10"]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 3000.0

    et0 = str2et("2026-01-05T00:00:00")
    etf = et0 + 30 * 86400.0
    interpolate_ephem_span = [et0, etf]
    parameters = HighFidelityEphemerisModel.InterpParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        interpolate_ephem_span=interpolate_ephem_span)

    # initial state (in canonical scale)
    x0_dim, _ = spkezr("-60000", et0, naif_frame, abcorr, naif_ids[1])
    x0 = [x0_dim[1:3]/parameters.DU; x0_dim[4:6]/parameters.VU]

    # time span (in canonical scale)
    tspan = (0.0, 12*86400/parameters.TU)

    # solve without callback
    prob = ODEProblem(HighFidelityEphemerisModel.eom_Nbody!, x0, tspan, parameters)
    sol = solve(prob, Vern7(), reltol=1e-12, abstol=1e-12)

    # # plot
    # fig = Figure(size=(600, 600))
    # ax3d = Axis3(fig[1,1]; aspect=:data)
    # scatter!(ax3d, [sol.u[1][1]], [sol.u[1][2]], [sol.u[1][3]], color=:black)
    # lines!(ax3d, Array(sol)[1,:], Array(sol)[2,:], Array(sol)[3,:], color=:black, linewidth=1.5)

    θ_target_list = deg2rad.([-60, -30, 0, 30, 80, 120, 175, 180, 200, 250, 320, 350, 358])
    for θ_target in θ_target_list
        # solve with callback
        time_bounds = (3*86400/parameters.TU, tspan[end])
        radius_bounds = (0.0, 1e6/parameters.DU)
        detect_trueanomaly = HighFidelityEphemerisModel.get_trueanomaly_event(θ_target; t_bounds=time_bounds, radius_bounds=radius_bounds)#, parameters.mus[1])
        affect!(integrator) = terminate!(integrator)
        cb = ContinuousCallback(detect_trueanomaly, affect!)
        sol_cb = solve(prob, Tsit5(), callback = cb, reltol=1e-12, abstol=1e-12)
        @test HighFidelityEphemerisModel.angle_difference(HighFidelityEphemerisModel.cart2trueanomaly(sol_cb.u[end][1:6], parameters.mus[1]), θ_target) < 1e-12
        #@show HighFidelityEphemerisModel.angle_difference(HighFidelityEphemerisModel.cart2trueanomaly(sol_cb.u[end][1:6], parameters.mus[1]), θ_target)

        # # append to plot
        # scatter!(ax3d, [sol_cb.u[end][1]], [sol_cb.u[end][2]], [sol_cb.u[end][3]], color=:red, marker=:cross, markersize=10)
    end
    #display(fig)
end


test_callback_trueanomaly()