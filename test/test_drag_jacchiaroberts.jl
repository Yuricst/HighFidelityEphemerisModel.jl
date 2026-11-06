"""Tests for Jacchia-Roberts atmospheric density model"""


if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "utils.jl"))
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
    furnsh_kernels()
end


function _drag_stm_parameters_jacchia_roberts(et0)
    naif_ids = ["399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    f_density = HighFidelityEphemerisModel.jacchia_roberts_f_density(frame_PCPF="IAU_EARTH")
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
    # reference values use the UTC-based MJD (et_to_utc_mjd via deltet)
    et = str2et("2020-01-01T00:00:00")
    f_density = HighFidelityEphemerisModel.jacchia_roberts_f_density()
    @test f_density(et, _pcpf_position(et, 200.0)) ≈ 3.0708716391170484e-10 rtol=1e-6
    @test f_density(et, _pcpf_position(et, 400.0)) ≈ 5.591768363368083e-12 rtol=1e-6
    @test f_density(et, _pcpf_position(et, 800.0)) ≈ 2.5775317271832244e-14 rtol=1e-6
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
    parameters = _drag_stm_parameters_jacchia_roberts(et0)
    u0 = [1.05, 0.0, 0.01, 0.0, 1.0, 0.0]
    tspan = (0.0, 3600.0 / parameters.TU)
    prob = ODEProblem(HighFidelityEphemerisModel.eom_Nbody!, u0, tspan, parameters)
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
    parameters = _drag_stm_parameters_jacchia_roberts(et0)
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


# Below 125 km the model uses the roots of a quartic in altitude; check that the
# 100-125 km branch is continuous with the >125 km branch and monotone.
function test_jacchia_roberts_low_altitude()
    H = HighFidelityEphemerisModel
    rho_below = H.JacchiaRobertsModel(125.0 - 1e-6)
    rho_above = H.JacchiaRobertsModel(125.0 + 1e-6)
    # the two branches are separate fits; a ~2e-4 relative mismatch at 125 km is inherent
    @test rho_below ≈ rho_above rtol=1e-3
    hs = 100.5:0.5:130.0
    rhos = [H.JacchiaRobertsModel(h) for h in hs]
    @test all(diff(rhos) .< 0)
    # order of magnitude vs standard atmospheres (~5e-7 kg/m^3 at 100 km, ~1e-8 at 125 km)
    @test 2e-7 < H.JacchiaRobertsModel(101.0) < 1e-6
    @test 5e-9 < H.JacchiaRobertsModel(124.9) < 3e-8

    # roots of the deflated polynomial must also be roots of the original quartic
    for tx in (300.0, 400.0, 500.0)
        root1, root2, x_root, y_root = H._jr_compute_roots(tx)
        c = copy(H._JR_CON_C)
        c[1] += 1.500625e6 * tx / (tx - H._JR_TZERO)
        p(z) = sum(c[i] * z^(i - 1) for i in 1:5)
        scale = sum(abs(c[i]) * 200.0^(i - 1) for i in 1:5)
        @test abs(p(root1)) / scale < 1e-10
        @test abs(p(root2)) / scale < 1e-10
        @test abs(p(complex(x_root, y_root))) / scale < 1e-10
    end
end


function test_geodetic_conversion()
    Re = 6378.137
    flat = 1 / 298.257223563
    for lat_deg in (-80.0, -45.0, 0.0, 30.0, 45.0, 80.0), alt in (100.0, 400.0)
        r = georec(deg2rad(20.0), deg2rad(lat_deg), alt, Re, flat)
        lon, lat, h = HighFidelityEphemerisModel._geodetic_lon_lat_alt(r, Re, flat)
        @test rad2deg(lat) ≈ lat_deg atol=1e-7
        @test h ≈ alt atol=1e-5
    end
end


function test_jacchia_roberts_spice_free()
    H = HighFidelityEphemerisModel
    et = str2et("2020-01-01T00:00:00")
    f_spice = H.jacchia_roberts_f_density()
    f_analytic = H.jacchia_roberts_f_density(sun_model=:analytic)
    f_user = H.jacchia_roberts_f_density(
        sun_position = et -> spkpos("SUN", et, "IAU_EARTH", "NONE", "EARTH")[1],
    )
    @test_throws ArgumentError H.jacchia_roberts_f_density(sun_model=:foo)
    for alt in (200.0, 400.0, 800.0)
        r_km = _pcpf_position(et, alt)
        ρ_spice = f_spice(et, r_km)
        # same Sun and ellipsoid (pck00011 radii) -> only the UTC source differs (built-in
        # leap-second table vs SPICE deltet), which agree to well below a microsecond
        @test f_user(et, r_km) ≈ ρ_spice rtol=1e-10
        # analytic Sun (~0.01-0.1 deg from the SPICE Sun in IAU_EARTH)
        @test f_analytic(et, r_km) ≈ ρ_spice rtol=2e-3
    end
    # regression values of the SPICE path are reproduced by the SPICE-free path
    @test f_analytic(et, _pcpf_position(et, 400.0)) ≈ 5.591768363368083e-12 rtol=2e-3

    # explicit ellipsoid
    f_wgs = H.jacchia_roberts_f_density(sun_model=:analytic, R_equatorial_km=6378.137,
        flattening=1/298.257223563)
    @test isfinite(f_wgs(et, _pcpf_position(et, 400.0)))

    # ForwardDiff through the SPICE-free callback
    r_km = _pcpf_position(et, 350.0) + [0.0, 200.0, 900.0]
    grad_ad = ForwardDiff.gradient(r -> f_analytic(et, r), r_km)
    grad_num = map(1:3) do i
        e = zeros(3); e[i] = 1e-3
        (f_analytic(et, r_km + e) - f_analytic(et, r_km - e)) / 2e-3
    end
    @test grad_ad ≈ grad_num rtol=1e-5

    # thread safety of the SPICE-free callback
    ets = et .+ 900.0 .* (0:99)
    rs = [[6778.0 * cos(0.2k), 6778.0 * sin(0.2k), 500.0 * sin(0.05k)] for k in 0:99]
    serial = [f_analytic(e, r) for (e, r) in zip(ets, rs)]
    threaded = zeros(length(ets))
    Threads.@threads for i in eachindex(ets)
        threaded[i] = f_analytic(ets[i], rs[i])
    end
    @test threaded == serial
end


"""BigFloat reference geodetic height (fixed-point iteration on the latitude)."""
function _geodetic_height_reference(r, Re, flat)
    setprecision(BigFloat, 200) do
        x, y, z = BigFloat.(r)
        a = BigFloat(Re)
        e2 = BigFloat(flat) * (2 - BigFloat(flat))
        p = sqrt(x^2 + y^2)
        φ = atan(z, p * (1 - e2))
        for _ in 1:60
            N = a / sqrt(1 - e2 * sin(φ)^2)
            φ = atan(z + e2 * N * sin(φ), p)
        end
        return p * cos(φ) + z * sin(φ) - a * sqrt(1 - e2 * sin(φ)^2)
    end
end


function test_geodetic_conversion_near_axis()
    H = HighFidelityEphemerisModel
    Re = 6378.137
    flat = 1 / 298.257223563
    geo_h(r) = H._geodetic_lon_lat_alt(r, Re, flat)[3]
    # height accurate to ~1e-12 km over all latitudes (incl. the poles) and heights
    worst = 0.0
    for lat_deg in -90.0:1.5:90.0, alt in (-50.0, 100.0, 500.0, 1000.0)
        r = georec(0.3, deg2rad(lat_deg), alt, Re, flat)
        worst = max(worst, abs(geo_h(r) - Float64(_geodetic_height_reference(r, Re, flat))))
    end
    @test worst < 1e-10

    # gradient of the height vs a BigFloat central difference, on / near / away from the axis
    # (on both sides of the series switch at p = 10 m)
    for p in (0.0, 1e-12, 1e-9, 1e-6, 1e-3, 9.99e-3, 1.001e-2, 1.0, 100.0),
        z in (6356.752 + 400.0, -(6356.752 + 50.0))
        r = [p * cos(0.7), p * sin(0.7), z]
        g_ad = ForwardDiff.gradient(geo_h, r)
        @test all(isfinite, g_ad)
        g_ref = setprecision(BigFloat, 200) do
            h = big"1e-30"
            [(_geodetic_height_reference(big.(r) .+ h .* (1:3 .== i), Re, flat) -
              _geodetic_height_reference(big.(r) .- h .* (1:3 .== i), Re, flat)) / (2h) for i in 1:3]
        end
        @test norm(g_ad - Float64.(g_ref)) < 1e-12
        @test abs(geo_h(r) - Float64(_geodetic_height_reference(r, Re, flat))) < 1e-10
    end

    # exactly on the axis with dual numbers: no MethodError / NaN, zero lateral derivative
    J = ForwardDiff.jacobian(r -> collect(H._geodetic_lon_lat_alt(r, Re, flat)), [0.0, 0.0, 6800.0])
    @test all(isfinite, J)
    @test J[3, :] ≈ [0.0, 0.0, 1.0] atol=1e-15

    # density callbacks over the pole: finite gradients that match central differences
    et = str2et("2024-06-01T10:00:00")
    for f in (HighFidelityEphemerisModel.harris_priester_f_density(),
              HighFidelityEphemerisModel.jacchia_roberts_f_density(sun_model=:analytic))
        for r in ([0.0, 0.0, 6756.75], [1e-9, 0.0, 6756.75], [0.3, -0.2, -6756.75])
            g = ForwardDiff.gradient(q -> f(et, q), r)
            @test all(isfinite, g)
        end
        r = [0.0, 0.0, 6756.75]
        g = ForwardDiff.gradient(q -> f(et, q), r)
        dz = (f(et, r + [0, 0, 1e-3]) - f(et, r - [0, 0, 1e-3])) / 2e-3
        @test g[3] ≈ dz rtol=1e-5
    end
    # HP is a smooth function of position across the axis: gradient continuous
    f = HighFidelityEphemerisModel.harris_priester_f_density()
    g0 = ForwardDiff.gradient(q -> f(et, q), [0.0, 0.0, 6756.75])
    g1 = ForwardDiff.gradient(q -> f(et, q), [1e-6, 1e-6, 6756.75])
    @test norm(g1 - g0) < 1e-6 * norm(g0)
end


test_jacchia_roberts_f_density_api()
test_jacchia_roberts_low_altitude()
test_geodetic_conversion()
test_jacchia_roberts_altitude_guard()
test_jacchia_roberts_regression()
test_jacchia_roberts_vs_harris_priester()
test_jacchia_roberts_eom()
test_jacchia_roberts_forwarddiff()
test_jacchia_roberts_eom_stm()
test_jacchia_roberts_spice_free()
test_geodetic_conversion_near_axis()
