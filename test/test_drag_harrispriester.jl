"""Tests for atmospheric drag perturbation"""


if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "utils.jl"))
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
    furnsh_kernels()
end


function _drag_stm_parameters_harris_priester(et0)
    naif_ids = ["399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    f_density = HighFidelityEphemerisModel.harris_priester_f_density()
    return HighFidelityEphemerisModel.SpiceParameters(
        et0, 6378.0, GMs, naif_ids, "J2000", "NONE";
        filepath_spherical_harmonics = joinpath(@__DIR__, "../data/earth/GGM03S.tab"),
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
            ODEProblem(HighFidelityEphemerisModel.eom_NbodySH!, x0_plus, tspan, parameters),
            Vern8(), reltol=1e-14, abstol=1e-14,
        )

        x0_min = copy(x0)
        x0_min[i] -= h
        sol_ptrb_min = solve(
            ODEProblem(HighFidelityEphemerisModel.eom_NbodySH!, x0_min, tspan, parameters),
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
    f_density = HighFidelityEphemerisModel.harris_priester_f_density()
    parameters = HighFidelityEphemerisModel.SpiceParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        frame_PCPF = "IAU_EARTH",
        include_drag = true,
        drag_Cd = 2.2,
        drag_Am = 0.01,
        f_density = f_density,
    )

    u0 = [1.05, 0.0, 0.01, 0.0, 1.0, 0.0]
    tspan = (0.0, 2 * 86400 / parameters.TU)

    prob = ODEProblem(HighFidelityEphemerisModel.eom_Nbody!, u0, tspan, parameters)
    sol = solve(prob, Vern7(), reltol=1e-12, abstol=1e-12)
    # regression value for the M&G Harris-Priester model (diurnal bulge, n = 6, analytic Sun)
    u_check = [-0.6522673489587052, 0.9289605126960042, -0.006210400132942633,
               -0.7793891515554786, -0.49969787601367466, -0.007419592032811833]
    # Tolerance is set by the numerical noise floor, not the model: over 2 days (~30 revs)
    # a 1-ulp change in u0 moves the final state by ~5e-9 and the Vern7 truncation error is
    # ~1e-9, so platform-dependent rounding (e.g. macOS CI gave 1.8e-9) must be tolerated.
    @test norm(sol.u[end] - u_check) < 1e-7

    # drag must still be a small perturbation of the drag-free orbit, and must decay it
    parameters_nodrag = HighFidelityEphemerisModel.SpiceParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr,
    )
    sol_nodrag = solve(
        ODEProblem(HighFidelityEphemerisModel.eom_Nbody!, u0, tspan, parameters_nodrag),
        Vern7(), reltol=1e-12, abstol=1e-12,
    )
    energy(u) = 0.5 * dot(u[4:6], u[4:6]) - 1 / norm(u[1:3])
    @test energy(sol.u[end]) < energy(sol_nodrag.u[end])

    et = parameters.et0
    r_km = SPICE.pxform(parameters.naif_frame, parameters.frame_PCPF, et) * u0[1:3] * parameters.DU
    rho = f_density(et, r_km)
    @test rho > 0.0
    v_atm = HighFidelityEphemerisModel.atmospheric_velocity(u0[1:3], parameters.TU, parameters.omega_atm)
    a = HighFidelityEphemerisModel.drag(u0[1:3], u0[4:6], v_atm, rho, parameters.k_drag)
    v_rel = u0[4:6] - v_atm
    @test dot(a, v_rel) < 0.0
end


function test_harris_priester_model()
    H = HighFidelityEphemerisModel

    # legacy single-column helper: exact at table heights, clamped outside [100, 1000] km
    rho_min = H.HarrisPriesterModel(400.0; use_min=true)
    rho_max = H.HarrisPriesterModel(400.0; use_min=false)
    @test rho_min ≈ 2.249e-12 atol=1e-15
    @test rho_max ≈ 7.492e-12 atol=1e-15
    rho_clamped = H.HarrisPriesterModel(50.0; use_min=true)
    @test rho_clamped ≈ 497400.0 * 1e-12 atol=1e-15

    # exponential interpolation (constant scale height per interval, M&G §3.5.2):
    # the density at the middle of an interval is the geometric mean of its end values
    @test H.HarrisPriesterModel(410.0; use_min=true) ≈ sqrt(2.249 * 1.558) * 1e-12 rtol=1e-12
    @test H.HarrisPriesterModel(410.0; use_min=false) ≈ sqrt(7.492 * 5.684) * 1e-12 rtol=1e-12
    @test H.HarrisPriesterModel(125.0; use_min=true) ≈ sqrt(24900.0 * 8377.0) * 1e-12 rtol=1e-12

    # full model at the bulge apex (cos ψ = 1) / antapex (cos ψ = -1) = table columns,
    # so the apex/antapex ratio equals rho_max / rho_min at every table height
    for (i, h) in enumerate(H._HARRIS_PRIESTER_HEIGHTS[2:end-1])
        ρ_apex = H.harris_priester_density(h, 1.0)
        ρ_anti = H.harris_priester_density(h, -1.0)
        @test ρ_apex ≈ H._HARRIS_PRIESTER_RHO_MAX[i+1] * 1e-12 rtol=1e-12
        @test ρ_anti ≈ H._HARRIS_PRIESTER_RHO_MIN[i+1] * 1e-12 rtol=1e-12
    end
    @test H.harris_priester_density(400.0, 1.0) / H.harris_priester_density(400.0, -1.0) ≈ 7.492 / 2.249 rtol=1e-12
    @test H.harris_priester_density(800.0, 1.0) / H.harris_priester_density(800.0, -1.0) ≈ 0.08059 / 0.007069 rtol=1e-12

    # cos^n(ψ/2) weighting: at ψ = 90° the weight is 2^(-n/2)
    for n in (2, 4, 6)
        ρ = H.harris_priester_density(400.0, 0.0; n=n)
        @test ρ ≈ (2.249 + (7.492 - 2.249) * 2.0^(-n / 2)) * 1e-12 rtol=1e-12
    end
    @test_throws ArgumentError H.harris_priester_density(400.0, 0.0; n=-1)

    # validity range 100 km < h < 1000 km (zero outside, as M&G)
    @test H.harris_priester_density(99.0, 1.0) == 0.0
    @test H.harris_priester_density(1000.0, 1.0) == 0.0
    @test H.harris_priester_density(1200.0, 1.0) == 0.0
    @test H.harris_priester_density(100.5, 1.0) > 0.0

    # legacy column mode of the callback (no bulge); flattening = 0 -> spherical altitude
    f_density = H.harris_priester_f_density(6378.0; flattening=0.0, use_min=true)
    @test f_density(0.0, [6378.0 + 400.0, 0.0, 0.0]) ≈ rho_min atol=1e-15
    @test f_density(0.0, [0.0, 0.0, 6378.0 + 400.0]) ≈ rho_min atol=1e-15
    @test f_density(0.0, [6378.0 + 50.0, 0.0, 0.0]) == 0.0
end


function test_harris_priester_geodetic_altitude()
    H = HighFidelityEphemerisModel
    Re, flat = 6378.137, 1 / 298.257223563
    f_density = H.harris_priester_f_density(Re; flattening=flat, use_min=true)
    for lat_deg in (0.0, 35.0, 60.0, 89.0), h in (250.0, 400.0, 700.0)
        r = georec(deg2rad(40.0), deg2rad(lat_deg), h, Re, flat)
        # Bowring geodetic height is accurate to ~1 mm here (relative density ~1e-7)
        @test f_density(0.0, r) ≈ H.HarrisPriesterModel(h; use_min=true) rtol=1e-6
    end
    # the same geocentric radius is ~21 km higher above the ellipsoid at the pole
    r_eq = [Re + 400.0, 0.0, 0.0]
    r_pole = [0.0, 0.0, Re + 400.0]
    @test f_density(0.0, r_pole) < f_density(0.0, r_eq)
    @test f_density(0.0, r_pole) ≈ H.HarrisPriesterModel(400.0 + Re * flat; use_min=true) rtol=1e-9
end


# Note: earlier versions had a `test_harris_priester_frame_invariance` test checking that
# the density is the same for a position given in J2000 and in IAU_EARTH. That only held
# for the previous altitude-only model. The M&G model's diurnal bulge is fixed relative to
# the Sun, and with the default analytic Sun the bulge is computed in an Earth-fixed frame,
# so `r_km` must be Earth-fixed (`frame_PCPF` = "ITRF93"/"IAU_EARTH") and the density is
# deliberately *not* invariant under rotations of `r_km`; see the test below.
function test_harris_priester_requires_earth_fixed_input()
    f_density = HighFidelityEphemerisModel.harris_priester_f_density()
    et = str2et("2024-06-01T10:00:00")
    r_j2000 = [6778.0, 0.0, 0.0]
    r_itrf = SPICE.pxform("J2000", "ITRF93", et) * r_j2000
    # the same physical point: only the Earth-fixed input gives the right local time
    ρ_right = f_density(et, r_itrf)
    ρ_wrong = f_density(et, r_j2000)
    @test abs(ρ_right / ρ_wrong - 1) > 0.1
    # the height-only legacy column mode is still invariant under rotations about the pole
    f_column = HighFidelityEphemerisModel.harris_priester_f_density(use_min=true)
    Rz = [cos(1.0) sin(1.0) 0.0; -sin(1.0) cos(1.0) 0.0; 0.0 0.0 1.0]
    @test f_column(et, Rz * r_itrf) ≈ f_column(et, r_itrf) rtol=1e-12
end


function test_harris_priester_diurnal_bulge()
    H = HighFidelityEphemerisModel
    Re = 6378.137
    f_density = H.harris_priester_f_density()
    flat = 1 / 298.257223563
    # point along unit vector `u` at geodetic height `h`
    function at_height(u, h)
        ρ = Re + h
        for _ in 1:5
            _, _, h_now = H._geodetic_lon_lat_alt(ρ * u, Re, flat)
            ρ += h - h_now
        end
        return ρ * u
    end
    for date in ("2020-01-01T00:00:00", "2026-03-20T12:00:00", "2024-06-21T06:00:00")
        et = str2et(date)
        s = H.sun_position_earth_fixed_analytic(et)
        δ = atan(s[3], hypot(s[1], s[2]))
        λs = atan(s[2], s[1])

        # apex: Sun's declination, 30° east of the Sun (lag in right ascension, ≈ 14h LT)
        u_apex = [cos(δ) * cos(λs + deg2rad(30.0)), cos(δ) * sin(λs + deg2rad(30.0)), sin(δ)]
        ρ_apex = f_density(et, at_height(u_apex, 400.0))
        ρ_anti = f_density(et, at_height(-u_apex, 400.0))
        @test ρ_apex ≈ 7.492e-12 rtol=1e-9
        @test ρ_anti ≈ 2.249e-12 rtol=1e-9
        @test ρ_apex / ρ_anti ≈ 7.492 / 2.249 rtol=1e-9

        # sweep longitude along the apex declination circle (constant geodetic height):
        # maximum 30° east of the Sun, minimum 180° away from it
        lons = λs .+ deg2rad.(0:0.5:359.5)
        ρs = [f_density(et, at_height([cos(δ) * cos(λ), cos(δ) * sin(λ), sin(δ)], 400.0)) for λ in lons]
        @test rad2deg(lons[argmax(ρs)] - λs) ≈ 30.0 atol=0.5
        @test rad2deg(lons[argmin(ρs)] - λs) ≈ 210.0 atol=0.5
        @test maximum(ρs) ≈ ρ_apex rtol=1e-6
    end

    # analytic Sun vs a DE440/ITRF93 Sun (SPICE): densities agree closely
    et = str2et("2020-01-01T00:00:00")
    f_spice = H.harris_priester_f_density(
        sun_position = et -> spkpos("SUN", et, "ITRF93", "NONE", "EARTH")[1],
    )
    worst = 0.0
    for lon in 0:30:330, lat in (-50.0, 0.0, 50.0)
        r = georec(deg2rad(lon), deg2rad(lat), 400.0, Re, 1 / 298.257223563)
        worst = max(worst, abs(f_density(et, r) / f_spice(et, r) - 1))
    end
    @test worst < 2e-3

    # larger n concentrates the bulge: lower density away from the apex
    f2 = H.harris_priester_f_density(n=2)
    f6 = H.harris_priester_f_density(n=6)
    s = H.sun_position_earth_fixed_analytic(et)
    λs = atan(s[2], s[1])
    r_90 = georec(λs + deg2rad(120.0), 0.0, 400.0, Re, 1 / 298.257223563)
    @test f2(et, r_90) > f6(et, r_90)
end


function test_harris_priester_forwarddiff()
    H = HighFidelityEphemerisModel
    h = 319.0
    d_fd = ForwardDiff.derivative(h -> H.HarrisPriesterModel(h), h)
    h_step = 1e-3
    d_num = (H.HarrisPriesterModel(h + h_step) - H.HarrisPriesterModel(h - h_step)) / (2 * h_step)
    @test d_fd ≈ d_num rtol=1e-6

    # legacy column mode on the equator: gradient is radial
    f_column = H.harris_priester_f_density(6378.0; use_min=true)
    r_km = [6378.0 + h, 0.0, 0.0]
    grad_fd = ForwardDiff.gradient(r -> f_column(0.0, r), r_km)
    @test all(isfinite, grad_fd)
    @test grad_fd[2] ≈ 0.0 atol=1e-20
    @test grad_fd[3] ≈ 0.0 atol=1e-20

    # full model: ForwardDiff gradient vs central differences
    et = str2et("2020-01-01T00:00:00")
    f_density = H.harris_priester_f_density()
    r_km = [6378.137 + 310.0, 1500.0, 2500.0]
    grad_ad = ForwardDiff.gradient(r -> f_density(et, r), r_km)
    grad_num = map(1:3) do i
        e = zeros(3); e[i] = 1e-3
        (f_density(et, r_km + e) - f_density(et, r_km - e)) / 2e-3
    end
    @test grad_ad ≈ grad_num rtol=1e-6
end


function test_harris_priester_threads()
    # the default callback is SPICE-free and can be evaluated concurrently
    f_density = HighFidelityEphemerisModel.harris_priester_f_density()
    et0 = str2et("2020-01-01T00:00:00")
    ets = et0 .+ 600.0 .* (0:199)
    rs = [[6778.0 * cos(0.1k), 6778.0 * sin(0.1k), 300.0] for k in 0:199]
    serial = [f_density(et, r) for (et, r) in zip(ets, rs)]
    threaded = zeros(length(ets))
    Threads.@threads for i in eachindex(ets)
        threaded[i] = f_density(ets[i], rs[i])
    end
    @test threaded == serial
end


function test_harris_priester_jacobian_fd(;verbose=false)
    naif_ids = ["399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 6378.0

    et0 = str2et("2020-01-01T00:00:00")
    f_density = HighFidelityEphemerisModel.harris_priester_f_density()
    parameters = HighFidelityEphemerisModel.SpiceParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        filepath_spherical_harmonics = joinpath(@__DIR__, "../data/earth/GGM03S.tab"),
        nmax = 2,
        frame_PCPF = "IAU_EARTH",
        include_drag = true,
        drag_Cd = 2.2,
        drag_Am = 0.01,
        f_density = f_density,
    )

    x0 = [1.03, 0.0, 0.001, 0.0, sqrt(1/1.03), 0.0]
    eom = HighFidelityEphemerisModel.eom_NbodySH
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
    parameters = _drag_stm_parameters_harris_priester(et0)
    # J2-corrected near-circular orbit at ~310 km: stays within one Harris-Priester table
    # interval (300-320 km) for the 3 h test, where the exponential profile is smooth
    # (the density gradient is discontinuous at table heights, as in M&G)
    r0 = 1.04862
    x0 = [r0, 0.0, 0.001, 0.0, sqrt((1 + 1.5 * 1.0826e-3 / r0^2) / r0), 0.0]
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

    # Longer integration with the Harris-Priester density.
    tspan = (0.0, 3 * 3600 / parameters.TU)

    prob = ODEProblem(HighFidelityEphemerisModel.eom_NbodySH!, x0, tspan, parameters)
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
test_harris_priester_geodetic_altitude()
test_harris_priester_requires_earth_fixed_input()
test_harris_priester_diurnal_bulge()
test_harris_priester_forwarddiff()
test_harris_priester_threads()
test_harris_priester_jacobian_fd()
test_harris_priester_eom_stm()
