"""
Tests for parameter display/validation paths, `factorial_safe`, the in-place third-body
acceleration and the cannonball SRP acceleration.
"""

using ForwardDiff
using LinearAlgebra
using Printf
using SPICE
using Test

if !@isdefined(furnsh_kernels)
    include(joinpath(@__DIR__, "utils.jl"))
end

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
    furnsh_kernels()
end


function _miscA_error_message(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end


@testset "parameters: show" begin
    naif_ids = ["399", "301", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    et0 = str2et("2026-01-05T00:00:00")

    params = HighFidelityEphemerisModel.SpiceParameters(
        et0, 6378.0, GMs, naif_ids, "J2000", "NONE";
        include_srp = true, include_drag = true,
        f_density = (args...) -> 0.0, frame_PCPF = "ITRF93",
        omega_atm = [0.0, 0.0, 7.5e-5],
    )
    out = sprint(show, params)
    lines = split(chomp(out), '\n')
    @test lines[1] == "SpiceParameters"
    @test occursin(et2utc(et0, "ISOC", 3), out)
    @test occursin(@sprintf("DU               : %1.8f", params.DU), out)
    @test occursin(@sprintf("TU               : %1.8f", params.TU), out)
    @test occursin(@sprintf("VU               : %1.8f", params.VU), out)
    @test occursin("include_srp      : true", out)
    @test occursin(@sprintf("k_srp_cannonball : %1.8f", params.k_srp_cannonball), out)
    @test occursin("idx_sun          : 3", out)
    @test occursin("include_drag     : true", out)
    @test occursin(@sprintf("k_drag           : %1.8f", params.k_drag), out)
    @test occursin("omega_atm        : [0.00000000, 0.00000000, 0.00007500]", out)

    # without SRP/drag the corresponding detail lines are omitted
    params_plain = HighFidelityEphemerisModel.SpiceParameters(et0, 6378.0, GMs[1:2], naif_ids[1:2])
    out_plain = sprint(show, params_plain)
    @test occursin("include_srp      : false", out_plain)
    @test occursin("include_drag     : false", out_plain)
    @test !occursin("k_srp_cannonball", out_plain)
    @test !occursin("idx_sun", out_plain)
    @test !occursin("k_drag", out_plain)
    @test !occursin("omega_atm", out_plain)
    @test length(split(chomp(out_plain), '\n')) == 7
end


@testset "parameters: non-canonical scales and backend validation" begin
    naif_ids = ["399", "301"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    et0 = 0.0
    spk = hfem_test_kernel("spk", "de440.bsp")

    # use_canonical_scales = false keeps dimensional (km, s) units
    params = HighFidelityEphemerisModel.SpiceParameters(
        et0, 6378.0, GMs, naif_ids; use_canonical_scales = false)
    @test (params.DU, params.TU, params.VU) == (1.0, 1.0, 1.0)
    @test params.mus == GMs
    params_c = HighFidelityEphemerisModel.SpiceParameters(et0, 6378.0, GMs, naif_ids)
    @test params_c.DU == 6378.0
    @test params_c.mus ≈ GMs / GMs[1]

    # nmax > 9 switches the factorial to the overflow-safe variant
    @test HighFidelityEphemerisModel.SpiceParameters(et0, 1.0, GMs, naif_ids; nmax = 12).factorial_alias ===
        HighFidelityEphemerisModel.factorial_safe
    @test HighFidelityEphemerisModel.SpiceParameters(et0, 1.0, GMs, naif_ids; nmax = 9).factorial_alias === factorial

    # provider and files are mutually exclusive
    provider = HighFidelityEphemerisModel.Ephemerides.EphemerisProvider(spk)
    err = _miscA_error_message(() -> HighFidelityEphemerisModel.EphemeridesParameters(
        et0, 1.0, GMs, naif_ids; ephemerides_provider = provider, ephemerides_files = spk))
    @test err isa ErrorException
    @test occursin("not both", err.msg)

    # the backend factory itself needs some ephemeris input
    err = _miscA_error_message(() ->
        HighFidelityEphemerisModel._make_ephemerides_backend(nothing, nothing, nothing, nothing))
    @test err isa ErrorException
    @test occursin("require `ephemerides_provider`", err.msg)

    # unknown backend symbol
    err = _miscA_error_message(() -> HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, 1.0, GMs, naif_ids; backend = :bogus))
    @test err isa ErrorException
    @test occursin("Unsupported HFEM parameter backend: bogus", err.msg)
end


@testset "factorial_safe" begin
    fs = HighFidelityEphemerisModel.factorial_safe
    @test fs(0) == 1
    @test fs(5) === 120
    @test fs(20) === factorial(20)
    @test fs(20) isa Int
    @test fs(21) isa BigInt
    @test fs(21) == 21 * big(factorial(20))
    @test fs(30) == factorial(big(30))
end


@testset "third_body_accel! (in place)" begin
    r_sc = [7000.0, -1200.0, 300.0]           # km, w.r.t. central body
    r_3b = [-384400.0, 12000.0, -3000.0]      # km
    mu_3b = 4902.800066

    a_classical = HighFidelityEphemerisModel.third_body_accel_classical(r_sc, r_3b, mu_3b)
    dx = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
    dx0 = copy(dx)
    @test isnothing(HighFidelityEphemerisModel.third_body_accel!(dx, r_sc, r_3b, mu_3b))
    @test dx[1:3] == dx0[1:3]                 # velocity part untouched
    @test isapprox(dx[4:6] - dx0[4:6], a_classical; rtol = 1e-10)
    @test isapprox(dx[4:6] - dx0[4:6], HighFidelityEphemerisModel.third_body_accel(r_sc, r_3b, mu_3b); rtol = 1e-14)

    # accumulates: a second call adds the same acceleration again
    HighFidelityEphemerisModel.third_body_accel!(dx, r_sc, r_3b, mu_3b)
    @test isapprox(dx[4:6] - dx0[4:6], 2 * a_classical; rtol = 1e-10)
end


@testset "srp_cannonball" begin
    k = 1.7e-3
    r = [1.0, 0.2, -0.1]
    r_sun = [-150.0, 30.0, 4.0]
    a = HighFidelityEphemerisModel.srp_cannonball(r, r_sun, k)
    d = r - r_sun
    @test isapprox(a, k * d / norm(d)^3; rtol = 1e-14)
    @test isapprox(norm(a), k / norm(d)^2; rtol = 1e-14)
    @test dot(a, d) > 0                       # pushes away from the Sun
    @test isapprox(collect(HighFidelityEphemerisModel._srp_cannonball3(r..., r_sun..., k)), a; rtol = 1e-14)

    # generic in the spacecraft position type (used under ForwardDiff)
    J = ForwardDiff.jacobian(x -> HighFidelityEphemerisModel.srp_cannonball(x, r_sun, k), r)
    dn = norm(d)
    J_analytic = k * (I / dn^3 - 3 * d * d' / dn^5)
    @test isapprox(J, J_analytic; rtol = 1e-12)
end
