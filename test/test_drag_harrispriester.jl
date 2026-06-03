"""Tests for atmospheric drag perturbation"""


if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "utils.jl"))
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
    furnsh_kernels()
end


function _drag_stm_parameters(et0)
    naif_ids = ["399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    f_density = HighFidelityEphemerisModel.harris_priester_f_density()
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


function _numerical_stm(x0, tspan, parameters; h=1e-7)
    STM_numerical = zeros(6, 6)
    for i = 1:6
        x0_plus = copy(x0)
        x0_plus[i] += h
        sol_ptrb = solve(
            ODEProblem(HighFidelityEphemerisModel.eom_NbodySH_SPICE!, x0_plus, tspan, parameters),
            Vern8(), reltol=1e-14, abstol=1e-14,
        )

        x0_min = copy(x0)
        x0_min[i] -= h
        sol_ptrb_min = solve(
            ODEProblem(HighFidelityEphemerisModel.eom_NbodySH_SPICE!, x0_min, tspan, parameters),
            Vern8(), reltol=1e-14, abstol=1e-14,
        )

        STM_numerical[:, i] = (sol_ptrb.u[end][1:6] - sol_ptrb_min.u[end][1:6]) / (2 * h)
    end
    return STM_numerical
end


function test_get_drag_coefficient()
    DU = 6378.0
    GM = 398600.4415
    VU = sqrt(GM / DU)
    TU = DU / VU
    drag_Cd = 2.2
    drag_Am = 0.01
    rho = 1e-12
    v_can = [1.0, 0.2, -0.1]
    v_atm = zeros(3)

    k_drag = HighFidelityEphemerisModel.get_drag_coefficient(DU, TU, VU, drag_Cd, drag_Am)
    a_can = HighFidelityEphemerisModel.drag(zeros(3), v_can, v_atm, rho, k_drag) * DU/TU^2

    v_m = v_can * VU * 1e3                                      # [m/s]
    a_m = -0.5 * rho * drag_Cd * drag_Am * norm(v_m) * v_m      # [m/s^2]
    a_can_expected = a_m / 1e3                                  # [km/s^2]

    @test a_can ≈ a_can_expected atol=1e-12
end


function test_atmospheric_velocity()
    r_can = [1.0, 0.3, -0.2]
    TU = 806.0
    omega_atm = [0.0, 0.0, 7.2921159e-5]

    v_atm = HighFidelityEphemerisModel.atmospheric_velocity(r_can, TU, omega_atm)
    @test v_atm ≈ TU * cross(omega_atm, r_can) atol=1e-14
end


function test_drag_opposes_relative_velocity()
    k_drag = 1.0
    rho = 1e-10
    v = [0.5, 0.1, -0.2]
    v_atm = [0.1, 0.0, 0.05]
    a = HighFidelityEphemerisModel.drag(zeros(3), v, v_atm, rho, k_drag)
    v_rel = v - v_atm
    @test dot(a, v_rel) < 0.0
end


function test_harris_priester_eom()
    naif_ids = ["399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 6378.0

    et0 = str2et("2020-01-01T00:00:00")
    # f_density = (et, r) -> 1e-12
    f_density = HighFidelityEphemerisModel.harris_priester_f_density()
    parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        frame_PCPF = "IAU_EARTH",
        include_drag = true,
        drag_Cd = 2.2,
        drag_Am = 0.01,
        f_density = f_density,
    )

    u0 = [1.05, 0.0, 0.01, 0.0, 1.0, 0.0]
    tspan = (0.0, 2 * 86400 / parameters.TU)

    prob = ODEProblem(HighFidelityEphemerisModel.eom_Nbody_SPICE!, u0, tspan, parameters)
    sol = solve(prob, Vern7(), reltol=1e-12, abstol=1e-12)
    u_check = [-0.6474801648845774, 0.9320966427600279, -0.006164827146379468,
               -0.7821155901146892, -0.49572178306666, -0.007445553992251351]
    @test norm(sol.u[end] - u_check) < 1e-10

    et = parameters.et0
    r_km = SPICE.pxform(parameters.naif_frame, parameters.frame_PCPF, et) * u0[1:3] * parameters.DU
    rho = f_density(et, r_km)
    v_atm = HighFidelityEphemerisModel.atmospheric_velocity(u0[1:3], parameters.TU, parameters.omega_atm)
    a = HighFidelityEphemerisModel.drag(u0[1:3], u0[4:6], v_atm, rho, parameters.k_drag)
    v_rel = u0[4:6] - v_atm
    @test dot(a, v_rel) < 0.0
end


function test_harris_priester_model()
    rho_min = HighFidelityEphemerisModel.HarrisPriesterModel(400.0; use_min=true)
    rho_max = HighFidelityEphemerisModel.HarrisPriesterModel(400.0; use_min=false)
    @test rho_min ≈ 2.249e-12 atol=1e-15
    @test rho_max ≈ 7.492e-12 atol=1e-15

    rho_clamped = HighFidelityEphemerisModel.HarrisPriesterModel(50.0; use_min=true)
    @test rho_clamped ≈ 497400.0 * 1e-12 atol=1e-15

    f_density = HighFidelityEphemerisModel.harris_priester_f_density(6378.0; use_min=true)
    r_km = [6378.0 + 400.0, 0.0, 0.0]
    @test f_density(0.0, r_km) ≈ rho_min atol=1e-15
end


function test_harris_priester_frame_invariance()
    et0 = str2et("2020-01-01T00:00:00")
    naif_frame = "J2000"
    frame_PCPF = "IAU_EARTH"
    f_density = HighFidelityEphemerisModel.harris_priester_f_density(6378.0; use_min=true)
    r_inertial = [6378.0 + 400.0, 0.0, 0.0]
    T = SPICE.pxform(naif_frame, frame_PCPF, et0)
    r_pcpf = T * r_inertial
    @test f_density(et0, r_pcpf) ≈ f_density(et0, r_inertial) atol=1e-15
end


function test_harris_priester_forwarddiff()
    h = 319.0
    d_fd = ForwardDiff.derivative(h -> HighFidelityEphemerisModel.HarrisPriesterModel(h), h)
    h_step = 1e-3
    d_num = (
        HighFidelityEphemerisModel.HarrisPriesterModel(h + h_step) -
        HighFidelityEphemerisModel.HarrisPriesterModel(h - h_step)
    ) / (2 * h_step)
    @test d_fd ≈ d_num rtol=1e-6

    f_density = HighFidelityEphemerisModel.harris_priester_f_density(6378.0; use_min=true)
    r_km = [6378.0 + h, 0.0, 0.0]
    grad_fd = ForwardDiff.gradient(r -> f_density(0.0, r), r_km)
    @test all(isfinite, grad_fd)
    @test grad_fd[2] ≈ 0.0 atol=1e-20
    @test grad_fd[3] ≈ 0.0 atol=1e-20
end


function test_harris_priester_jacobian_fd(;verbose=false)
    naif_ids = ["399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 6378.0

    et0 = str2et("2020-01-01T00:00:00")
    f_density = HighFidelityEphemerisModel.harris_priester_f_density()
    parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        filepath_spherical_harmonics = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab"),
        nmax = 2,
        frame_PCPF = "IAU_EARTH",
        include_drag = true,
        drag_Cd = 2.2,
        drag_Am = 0.01,
        f_density = f_density,
    )

    x0 = [1.03, 0.0, 0.001, 0.0, sqrt(1/1.03), 0.0]
    eom = HighFidelityEphemerisModel.eom_NbodySH_SPICE
    jac_numerical_fd = HighFidelityEphemerisModel.eom_jacobian_fd(
        eom, x0, 0.0, parameters, 0.0
    )
    @test all(isfinite, jac_numerical_fd)

    f_eval = eom(x0, parameters, 0.0)
    jac_numerical = zeros(6, 6)
    h = 1e-8
    for i = 1:6
        x0_copy = copy(x0)
        x0_copy[i] += h
        jac_numerical[:, i] = (eom(x0_copy, parameters, 0.0) - f_eval) / h
    end

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
end


function test_harris_priester_eom_stm(; verbose=false)
    et0 = str2et("2030-01-01T00:00:00")
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

    # Longer integration with smooth cubic Harris-Priester density.
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


test_get_drag_coefficient()
test_atmospheric_velocity()
test_drag_opposes_relative_velocity()
test_harris_priester_eom()
test_harris_priester_model()
test_harris_priester_frame_invariance()
test_harris_priester_forwarddiff()
test_harris_priester_jacobian_fd()
test_harris_priester_eom_stm()
