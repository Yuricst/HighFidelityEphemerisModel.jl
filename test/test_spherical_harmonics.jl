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
end

test_earth_spherical_harmonics()
test_spherical_harmonics()
