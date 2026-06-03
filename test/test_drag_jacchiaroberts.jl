"""Tests for Jacchia-Roberts atmospheric density model"""


if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "utils.jl"))
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
    furnsh_kernels()
end


function _drag_stm_parameters(et0)
    naif_ids = ["399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    f_density = HighFidelityEphemerisModel.jacchia_roberts_f_density(frame_PCPF="IAU_EARTH")
    return HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, 6378.0, GMs, naif_ids, "J2000", "NONE";
        filepath_spherical_harmonics = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab"),
        nmax = 2,
        frame_PCPF = "IAU_EARTH",
        include_drag = true,
        drag_Cd = 2.2,
        drag_Am = 0.01,
        f_density = f_density,
    )
end



function _pcpf_position(et, alt_km, naif_frame="J2000", frame_PCPF="IAU_EARTH")
    Re = bodvrd("399", "RADII", 3)[1]
    r_inertial = [Re + alt_km, 0.0, 0.0]
    T = SPICE.pxform(naif_frame, frame_PCPF, et)
    return T * r_inertial
end


function test_jacchia_roberts_f_density_api()
    et = str2et("2020-01-01T00:00:00")
    f_density = HighFidelityEphemerisModel.jacchia_roberts_f_density()
    r_km = _pcpf_position(et, 400.0)
    rho = f_density(et, r_km)
    @test isfinite(rho)
    @test rho > 0.0
end


function test_jacchia_roberts_altitude_guard()
    et = str2et("2020-01-01T00:00:00")
    f_density = HighFidelityEphemerisModel.jacchia_roberts_f_density()
    r_km = _pcpf_position(et, 80.0)
    @test_throws ErrorException f_density(et, r_km)
end


function test_jacchia_roberts_regression()
    et = str2et("2020-01-01T00:00:00")
    f_density = HighFidelityEphemerisModel.jacchia_roberts_f_density()
    @test f_density(et, _pcpf_position(et, 200.0)) ≈ 3.0708677711266465e-10 rtol=1e-6
    @test f_density(et, _pcpf_position(et, 400.0)) ≈ 5.591753480974712e-12 rtol=1e-6
    @test f_density(et, _pcpf_position(et, 800.0)) ≈ 2.5775215118405146e-14 rtol=1e-6
end


function test_jacchia_roberts_vs_harris_priester()
    et = str2et("2020-01-01T00:00:00")
    r_km = _pcpf_position(et, 400.0)
    rho_jr = HighFidelityEphemerisModel.jacchia_roberts_f_density()(et, r_km)
    rho_hp = HighFidelityEphemerisModel.harris_priester_f_density(6378.0; use_min=true)(et, r_km)
    @test rho_jr > 0.0
    @test rho_hp > 0.0
    @test rho_jr != rho_hp
    @test abs(rho_jr - rho_hp) < 1e-11
end


function test_jacchia_roberts_eom()
    et0 = str2et("2020-01-01T00:00:00")
    parameters = _drag_stm_parameters(et0)
    u0 = [1.05, 0.0, 0.01, 0.0, 1.0, 0.0]
    tspan = (0.0, 3600.0 / parameters.TU)
    prob = ODEProblem(HighFidelityEphemerisModel.eom_Nbody_SPICE!, u0, tspan, parameters)
    sol = solve(prob, Vern7(), reltol=1e-12, abstol=1e-12)
    @test all(isfinite, sol.u[end])
end


function test_jacchia_roberts_forwarddiff()
    h = 319.0
    d_fd = ForwardDiff.derivative(h -> HighFidelityEphemerisModel.JacchiaRobertsModel(h), h)
    h_step = 1e-3
    d_num = (
        HighFidelityEphemerisModel.JacchiaRobertsModel(h + h_step) -
        HighFidelityEphemerisModel.JacchiaRobertsModel(h - h_step)
    ) / (2 * h_step)
    @test d_fd ≈ d_num rtol=1e-6

    f_density = HighFidelityEphemerisModel.jacchia_roberts_f_density()
    r_km = [6378.0 + h, 0.0, 0.0]
    grad_fd = ForwardDiff.gradient(r -> f_density(0.0, r), r_km)
    @test all(isfinite, grad_fd)
    @test grad_fd[2] ≈ 0.0 atol=1e-14
    @test grad_fd[3] ≈ 0.0 atol=1e-14
end


function test_jacchia_roberts_eom_stm(; verbose=false)
    et0 = str2et("2020-01-01T00:00:00")
    parameters = _drag_stm_parameters(et0)
    x0 = [1.03, 0.0, 0.001, 0.0, sqrt(1/1.03), 0.0]
    x0_stm = [x0; reshape(I(6), 36)]

    # Short integration: linearized STM should match finite-difference reference closely.
    tspan_short = (0.0, 0.1 * 3600 / parameters.TU)
    sol_fd_short = solve(
        ODEProblem(HighFidelityEphemerisModel.eom_stm_NbodySH_SPICE_fd!, x0_stm, tspan_short, parameters),
        Vern8(), reltol=1e-14, abstol=1e-14,
    )
    STM_analytical_short = reshape(sol_fd_short.u[end][7:42], 6, 6)
    STM_numerical_short = _numerical_stm(x0, tspan_short, parameters)
    @test maximum(abs.(STM_analytical_short - STM_numerical_short)) < 1e-5

    # Longer integration with Jacchia-Roberts density.
    tspan = (0.0, 3 * 3600 / parameters.TU)

    prob = ODEProblem(HighFidelityEphemerisModel.eom_NbodySH_SPICE!, x0, tspan, parameters)
    sol = solve(prob, Vern8(), reltol=1e-14, abstol=1e-14)
    @test sol.retcode == SciMLBase.ReturnCode.Success

    prob_fd = ODEProblem(HighFidelityEphemerisModel.eom_stm_NbodySH_SPICE_fd!, x0_stm, tspan, parameters)
    sol_fd = solve(prob_fd, Vern8(), reltol=1e-14, abstol=1e-14)
    @test sol_fd.retcode == SciMLBase.ReturnCode.Success
    @test norm(sol.u[end] - sol_fd.u[end][1:6]) < 1e-9

    STM_analytical = reshape(sol_fd.u[end][7:42], 6, 6)
    STM_numerical = _numerical_stm(x0, tspan, parameters; h=1e-6)
    if verbose
        println("Analytical STM:")
        print_matrix(STM_analytical)
        println()
        println("Numerical STM:")
        print_matrix(STM_numerical)
        println()
        println("Diff:")
        print_matrix(STM_analytical - STM_numerical)
        println("Abs relative diff:")
        print_matrix(abs.(STM_analytical - STM_numerical) ./ abs.(STM_numerical))
    end
    @test maximum(abs.(STM_analytical - STM_numerical)) < 1e-5
end


test_jacchia_roberts_f_density_api()
test_jacchia_roberts_altitude_guard()
test_jacchia_roberts_regression()
test_jacchia_roberts_vs_harris_priester()
test_jacchia_roberts_eom()
test_jacchia_roberts_forwarddiff()
test_jacchia_roberts_eom_stm()