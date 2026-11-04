"""
Test spherical harmonics
"""

using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq
using SPICE
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end


function test_earth_spherical_harmonics(nmax::Int=8)
    filepath = joinpath(@__DIR__, "../data/earth/GGM03S.tab")
    # filepath = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab")
    denormalize = true
    spherical_harmonics_data = HighFidelityEphemerisModel.load_spherical_harmonics(filepath, nmax, denormalize)
    @test spherical_harmonics_data["Cnm"][2,0] ≈ -0.00108263 atol=1e-6
    return
end


function test_spherical_harmonics()
    # load gggrd_20x20.tab file
    nmax = 8
    denormalize = true

    filepath = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab")
    spherical_harmonics_data = HighFidelityEphemerisModel.load_spherical_harmonics(filepath, nmax, denormalize)

    rvec = [0.0, 0.0, 1838.0]
    lmb, phi, r = HighFidelityEphemerisModel.cart2sph(rvec)

    # evaluate a single acceleration term in the infinite series
    a_nm = HighFidelityEphemerisModel.spherical_harmonics_nm_accel_PCPF(
        phi, lmb, r,
        spherical_harmonics_data["Cnm"],
        spherical_harmonics_data["Snm"],
        spherical_harmonics_data["GM"],
        spherical_harmonics_data["REFERENCE RADIUS"],
        2,0
    )

    # evaluate the cumulative acceleration due to spherical harmonics
    a_full = HighFidelityEphemerisModel.spherical_harmonics_accel_PCPF(
        rvec,
        spherical_harmonics_data["Cnm"],
        spherical_harmonics_data["Snm"],
        spherical_harmonics_data["GM"],
        spherical_harmonics_data["REFERENCE RADIUS"],
        nmax
    )
    a_full_check = [
        3.0575156328702336e-7,
        -1.812482284272607e-8,
        4.304294506669716e-7
    ]
    @test all(isapprox.(a_full, a_full_check, atol=1e-10))

    # nmax >= 10 evaluates Legendre terms of degree >= 11 (parameters then select `factorial_safe`).
    high_degree_nmax = 10
    high_degree_data = HighFidelityEphemerisModel.load_spherical_harmonics(
        filepath, high_degree_nmax, denormalize
    )
    high_degree_accel = HighFidelityEphemerisModel.spherical_harmonics_accel_PCPF(
        rvec,
        high_degree_data["Cnm"],
        high_degree_data["Snm"],
        high_degree_data["GM"],
        high_degree_data["REFERENCE RADIUS"],
        high_degree_nmax
    )
    @test all(isfinite, high_degree_accel)

    params = HighFidelityEphemerisModel.SpiceParameters(
        0.0,
        1.0,
        [1.0],
        ["301"];
        filepath_spherical_harmonics = filepath,
        nmax = high_degree_nmax,
    )
    @test params.factorial_alias === HighFidelityEphemerisModel.factorial_safe
end


"""
Reference associated Legendre function (no Condon-Shortley phase) from the explicit
sum, M&G eqn (3.11) expanded, evaluated exactly with BigInt factorials in BigFloat.
"""
function legendre_reference(n::Int, m::Int, t::Real)
    setprecision(BigFloat, 512) do
        tb = BigFloat(t)
        s = zero(BigFloat)
        for k = 0:div(n - m, 2)
            coeff = factorial(big(2n - 2k)) //
                (factorial(big(k)) * factorial(big(n - k)) * factorial(big(n - m - 2k)))
            s += (-1)^k * coeff * tb^(n - m - 2k)
        end
        return BigFloat(2)^(-n) * (1 - tb^2)^(BigFloat(m) / 2) * s
    end
end


"""
Spherical-harmonics part of the gravitational potential (degrees 2..nmax), M&G eqn (3.27),
evaluated in BigFloat with the reference Legendre functions.
"""
function sh_potential_reference(rvec, Cnm, Snm, GM, R, nmax)
    setprecision(BigFloat, 512) do
        x, y, z = BigFloat.(rvec)
        r = sqrt(x^2 + y^2 + z^2)
        sinφ = z / r
        λ = atan(y, x)
        U = zero(BigFloat)
        for n = 2:nmax, m = 0:n
            Pnm = legendre_reference(n, m, sinφ)
            U += (R / r)^(n + 1) * Pnm * (Cnm[n, m] * cos(m * λ) + Snm[n, m] * sin(m * λ))
        end
        return GM / R * U
    end
end


function test_legendre_recurrence()
    legendre = HighFidelityEphemerisModel.legendre

    # closed forms (M&G Table 3.1 convention, no Condon-Shortley phase)
    for t in (-0.7, 0.0, 0.3, 0.95)
        c = sqrt(1 - t^2)
        @test legendre(0, 0, t) ≈ 1.0
        @test legendre(1, 0, t) ≈ t
        @test legendre(1, 1, t) ≈ c
        @test legendre(2, 0, t) ≈ (3t^2 - 1) / 2
        @test legendre(2, 1, t) ≈ 3t * c
        @test legendre(2, 2, t) ≈ 3 * c^2
        @test legendre(3, 1, t) ≈ 1.5 * (5t^2 - 1) * c
        @test legendre(3, 3, t) ≈ 15 * c^3
    end

    # regression: the former explicit sum overflowed Int64 for n >= 13
    @test legendre(13, 0, 0.3) ≈ -0.18391780445464623 rtol = 1e-13
    @test legendre(15, 2, 0.3) ≈ -49.78676279834872 rtol = 1e-13
    @test legendre(21, 5, 0.3) ≈ 782504.6838764548 rtol = 1e-13

    # agreement with the exact reference up to degree 30, error scaled by max |P_nm| over t
    ts = [-1.0, -0.999, -0.8, -0.31, 0.0, 0.2, 0.55, 0.9, 0.9999, 1.0]
    worst = 0.0
    for n = 0:30, m = 0:n
        ref = [legendre_reference(n, m, t) for t in ts]
        scale = max(maximum(abs, ref), floatmin(Float64))
        for (t, P_ref) in zip(ts, ref)
            P = legendre(n, m, t)
            @test P isa Float64
            worst = max(worst, Float64(abs(P - P_ref) / scale))
        end
    end
    @test worst < 1e-12

    # ForwardDiff: finite derivatives at the poles for m = 0 and even m, matching the reference
    for (n, m) in ((2, 0), (9, 0), (10, 4))
        for t in (-1.0, 1.0)
            @test isfinite(ForwardDiff.derivative(τ -> legendre(n, m, τ), t))
        end
    end
    h = 1e-20
    for (n, m, t) in ((7, 0, 0.4), (12, 3, -0.6), (20, 11, 0.25))
        dP = ForwardDiff.derivative(τ -> legendre(n, m, τ), t)
        dP_ref = (legendre_reference(n, m, big(t) + h) - legendre_reference(n, m, big(t) - h)) / (2h)
        @test dP ≈ Float64(dP_ref) rtol = 1e-11
    end
end


function test_high_degree_spherical_harmonics()
    # nmax = 20 exercises Legendre degrees up to 21 (previously corrupted by Int64 overflow)
    nmax = 20
    filepath = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab")
    data = HighFidelityEphemerisModel.load_spherical_harmonics(filepath, nmax, true)
    Cnm, Snm = data["Cnm"], data["Snm"]
    GM, R = data["GM"], data["REFERENCE RADIUS"]
    accel(rvec) = HighFidelityEphemerisModel.spherical_harmonics_accel_PCPF(rvec, Cnm, Snm, GM, R, nmax)

    for rvec in ([0.0, 0.0, 1838.0], [0.5, -0.3, 1838.0], [50.0, -30.0, 1838.0],
                 [1300.0, -1000.0, 800.0], [-1750.0, 30.0, -200.0], [10.0, 5.0, -1790.0])
        a = accel(rvec)
        @test eltype(a) == Float64

        # the acceleration must be the gradient of the potential (central differences in BigFloat)
        grad = setprecision(BigFloat, 512) do
            h = BigFloat(10)^-30
            map(1:3) do i
                e = zeros(BigFloat, 3); e[i] = h
                rb = BigFloat.(rvec)
                (sh_potential_reference(rb + e, Cnm, Snm, GM, R, nmax) -
                 sh_potential_reference(rb - e, Cnm, Snm, GM, R, nmax)) / (2h)
            end
        end
        @test norm(a - Float64.(grad)) < 5e-12 * norm(Float64.(grad))

        # the Jacobian of a gradient field is symmetric; also checks ForwardDiff compatibility
        # (skipped within a few km of the polar axis: the spherical-coordinate formulation is
        # singular on the axis and ill-conditioned next to it, independently of the degree)
        hypot(rvec[1], rvec[2]) > 10.0 || continue
        J = ForwardDiff.jacobian(accel, rvec)
        @test all(isfinite, J)
        @test norm(J - J') < 1e-11 * norm(J)
    end
end

test_earth_spherical_harmonics()
test_spherical_harmonics()
test_legendre_recurrence()
test_high_degree_spherical_harmonics()