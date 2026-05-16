"""
Test spherical harmonics
"""

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

    # nmax >= 10 reaches factorial(22) in the n + 1 Legendre terms.
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

    params = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        0.0,
        1.0,
        [1.0],
        ["301"];
        filepath_spherical_harmonics = filepath,
        nmax = high_degree_nmax,
        get_jacobian_func = false,
    )
    @test params.factorial_alias === HighFidelityEphemerisModel.factorial_safe
end

test_earth_spherical_harmonics()
test_spherical_harmonics()
