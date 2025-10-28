"""
Test eom_NbodySH_SPICE
"""

using LinearAlgebra
using OrdinaryDiffEq
using SPICE
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end


test_eom_NbodySH_SPICE = function()
    # load gggrd_20x20.tab file
    nmax = 8
    denormalize = true

    filepath_spherical_harmonics = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab")
    #spherical_harmonics_data = HighFidelityEphemerisModel.load_spherical_harmonics(filepath, nmax, denormalize)

    # define parameters
    GMs = [
        4.9028000661637961E+03,
        0.0, #3.9860043543609598E+05,
        0.0, #1.3271244004193938E+11,
    ]
    naif_ids = ["301", "399", "10"]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 1737.4

    et0 = str2et("2020-01-01T00:00:00")
    parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        filepath_spherical_harmonics = filepath_spherical_harmonics,
        nmax = nmax,
        frame_PCPF = "MOON_PA",
        include_srp = true,
        srp_Cr = 1.15,
        srp_Am = 0.002,
        srp_P0 = 4.56e-6,
    )
    # @show parameters.DU, parameters.TU, parameters.VU
    # @show parameters.mus

    # initial state (in canonical scale)
    u0 = [1.0, 0.0, 0.3, 0.5, 1.0, 0.0]

    # time span (in canonical scale)
    tspan = (0.0, 3*86400/parameters.TU)

    # solve
    prob = ODEProblem(HighFidelityEphemerisModel.eom_NbodySH_SPICE!, u0, tspan, parameters)
    sol = solve(prob, Vern7(), reltol=1e-14, abstol=1e-14)
    u_check = [-1.3008031550828112, 1.082155017821313, -0.5688825611441191,
               -0.13115469895637183, -0.6582761913248995, 0.06125335489936464]
    @test norm(sol.u[end] - u_check) < 1e-11

    # also solve the two-body problem for plotting
    # prob_twobody = ODEProblem(HighFidelityEphemerisModel.eom_Nbody_SPICE!, u0, tspan, parameters)
    # sol_twobody = solve(prob_twobody, Vern7(), reltol=1e-14, abstol=1e-14)

    # # plot
    # fig = Figure(size=(600,600))
    # ax3d = Axis3(fig[1,1])
    # lines!(ax3d, Array(sol_twobody)[1,:], Array(sol_twobody)[2,:], Array(sol_twobody)[3,:], color=:blue)
    # lines!(ax3d, Array(sol)[1,:], Array(sol)[2,:], Array(sol)[3,:], color=:red, linewidth=0.5)
    # fig
end


function test_eom_stm_NbodySH_SPICE(;verbose=false)
    # load gggrd_20x20.tab file
    nmax = 4
    denormalize = true

    filepath_spherical_harmonics = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab")
    #spherical_harmonics_data = HighFidelityEphemerisModel.load_spherical_harmonics(filepath, nmax, denormalize)

    # define parameters
    naif_ids = ["301", "399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 1e5

    et0 = str2et("2026-01-05T00:00:00")
    etf = et0 + 30 * 86400.0
    interpolate_ephem_span = [et0, etf]
    interpolation_time_step = 1000.0
    parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        interpolate_ephem_span=interpolate_ephem_span,
        filepath_spherical_harmonics = filepath_spherical_harmonics,
        nmax = nmax,
        frame_PCPF = "MOON_PA",
        interpolation_time_step = interpolation_time_step,
        include_srp = true,
        srp_Cr = 1.15,
        srp_Am = 0.002,
        srp_P0 = 4.56e-6,
    )

    # initial state (in canonical scale)
    x0_dim, _ = spkezr("-60000", et0, naif_frame, abcorr, naif_ids[1])
    x0 = [x0_dim[1:3]/parameters.DU; x0_dim[4:6]/parameters.VU]
    x0_stm = [x0; reshape(I(6),36)]

    # evaluate Jacobian
    f_eval = zeros(6)
    HighFidelityEphemerisModel.eom_NbodySH_SPICE!(f_eval, x0, parameters, 0.0)
    jac_numerical = zeros(6,6)
    h = 1e-8
    for i = 1:6
        x0_copy = copy(x0)
        x0_copy[i] += h
        _f_eval = zeros(6)
        HighFidelityEphemerisModel.eom_NbodySH_SPICE!(_f_eval, x0_copy, parameters, 0.0)
        jac_numerical[:,i] = (_f_eval - f_eval) / h
    end
    jac_numerical_fd = HighFidelityEphemerisModel.eom_jacobian_fd(
        HighFidelityEphemerisModel.eom_NbodySH_SPICE, x0, 0.0, parameters, 0.0
    )

    if verbose
        println("Numerical Jacobian:")
        print_matrix(jac_numerical)
        println()
        println("ForwardDiff Jacobian:")
        print_matrix(jac_numerical_fd)
        println()
        println("jac_numerical - jac_numerical_fd:")
        print_matrix(jac_numerical - jac_numerical_fd)
        println()
    end
    @test maximum(abs.(jac_numerical_fd - jac_numerical)) < 1e-6

    # time span (in canonical scale)
    tspan = (0.0, 7*86400/parameters.TU)

    # solve just the state
    prob = ODEProblem(HighFidelityEphemerisModel.eom_NbodySH_SPICE!, x0, tspan, parameters)
    sol = solve(prob, Vern8(), reltol=1e-14, abstol=1e-14)
    @test sol.retcode == SciMLBase.ReturnCode.Success

    # solve with ForwardDiff Jacobian
    prob_fd = ODEProblem(HighFidelityEphemerisModel.eom_stm_NbodySH_SPICE_fd!, x0_stm, tspan, parameters)
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
    @test norm(sol.u[end][1:6] - sol_fd.u[end][1:6]) < 1e-12

    # construct numerical STM
    STM_analytical = reshape(sol_fd.u[end][7:42],6,6)'
    STM_numerical = zeros(6,6)
    h = 1e-7
    for i = 1:6
        x0_plus = copy(x0)
        x0_plus[i] += h
        sol_ptrb = solve(ODEProblem(HighFidelityEphemerisModel.eom_NbodySH_SPICE!, x0_plus, tspan, parameters), Vern8(), reltol=1e-14, abstol=1e-14)

        x0_min = copy(x0)
        x0_min[i] -= h
        sol_ptrb_min = solve(ODEProblem(HighFidelityEphemerisModel.eom_NbodySH_SPICE!, x0_min, tspan, parameters), Vern8(), reltol=1e-14, abstol=1e-14)

        STM_numerical[:,i] = (sol_ptrb.u[end][1:6] - sol_ptrb_min.u[end][1:6]) / (2*h)
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
    @test maximum(abs.(STM_analytical - STM_numerical)) < 1e-5
end


test_eom_NbodySH_SPICE()
test_eom_stm_NbodySH_SPICE(;verbose = false)