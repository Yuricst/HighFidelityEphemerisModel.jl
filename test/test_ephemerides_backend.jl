"""
Tests for the Ephemerides.jl backend helpers in `src/ephemerides.jl`: the provider
fallback chain (direct / SSB / Earth-Moon-barycenter bridge), provider-based state
queries, `EphemeridesBackend` construction and frame-system point registration.
"""

using LinearAlgebra
using SPICE
using Test

if !@isdefined(furnsh_kernels)
    include(joinpath(@__DIR__, "utils.jl"))
end

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
    furnsh_kernels()
end

const _ephA_HFEM = HighFidelityEphemerisModel
const _ephA_Eph = HighFidelityEphemerisModel.Ephemerides
const _ephA_FT = HighFidelityEphemerisModel.FrameTransformations


# ----------------------------------------------------------------------------------------
# Mock provider: knows the exact SSB-relative state of every point, but only answers
# queries for the (from, to) pairs it was given (in either orientation), mimicking a
# kernel set where only some SPK segments are loaded. Every query is logged.
# ----------------------------------------------------------------------------------------
struct _EphAMissingPair <: Exception
    from::Int
    to::Int
end

struct _EphAMockProvider
    pairs::Set{Tuple{Int,Int}}
    calls::Vector{Tuple{Int,Int}}
end

_EphAMockProvider(pairs) = _EphAMockProvider(Set{Tuple{Int,Int}}(pairs), Tuple{Int,Int}[])

# deterministic, point- and time-dependent SSB-relative state (SSB itself is the origin)
function _ephA_ssb_state(point::Int, et::Number)
    point == 0 && return zeros(6)
    p = Float64(point)
    return [p + 1e-3 * et, 2p - 7.0, -0.5p + sin(et), 1e-2p, 0.3 + 1e-4p, cos(et)]
end

function _ephA_mock_state(m::_EphAMockProvider, from::Int, to::Int, et::Number)
    push!(m.calls, (from, to))
    ((from, to) in m.pairs || (to, from) in m.pairs) || throw(_EphAMissingPair(from, to))
    return _ephA_ssb_state(to, et) - _ephA_ssb_state(from, et)
end

_ephA_Eph.ephem_vector6(m::_EphAMockProvider, from::Int, to::Int, et::Number) =
    _ephA_mock_state(m, from, to, et)
_ephA_Eph.ephem_vector3(m::_EphAMockProvider, from::Int, to::Int, et::Number) =
    _ephA_mock_state(m, from, to, et)[1:3]

# expected `to` w.r.t. `from` in the first `n` components
_ephA_expected(from, to, et, n) = (_ephA_ssb_state(to, et) - _ephA_ssb_state(from, et))[1:n]

# the (vector, n) variants of each internal helper
const _ephA_variants = (
    (n = 3, fallback = _ephA_HFEM._ephem_vector3_with_fallback,
        ssb = _ephA_HFEM._ephem_vector3_wrt_ssb, bridge = _ephA_HFEM._ephem_vector3_wrt_bridge),
    (n = 6, fallback = _ephA_HFEM._ephem_vector6_with_fallback,
        ssb = _ephA_HFEM._ephem_vector6_wrt_ssb, bridge = _ephA_HFEM._ephem_vector6_wrt_bridge),
)


function _ephA_spice_kernels()
    spk = hfem_test_kernel("spk", "de440.bsp")
    sc = haskey(ENV, "SPICE") ?
        joinpath(ENV["SPICE"], "misc", "dsg_naif", "receding_horiz_3189_1burnApo_DiffCorr_15yr.bsp") :
        joinpath(@__DIR__, "../spice/test", "receding_horiz_3189_1burnApo_DiffCorr_15yr.bsp")
    moon_bpc = hfem_test_kernel("pck", "moon_pa_de440_200625.bpc")
    return (; spk, sc, moon_bpc)
end

# minimum per-thread cache length over every cached segment of `provider`
function _ephA_min_cache_length(provider)
    nmin = typemax(Int)
    for daf in provider.files, fname in fieldnames(typeof(daf.seglist))
        segs = getfield(daf.seglist, fname)
        segs isa AbstractVector || continue
        for seg in segs
            hasfield(typeof(seg), :cache) || continue
            caches = getfield(seg, :cache)
            (caches isa AbstractVector && !isempty(caches)) || continue
            nmin = min(nmin, length(caches))
        end
    end
    return nmin
end


@testset "SPK parent-point classification" begin
    parent = _ephA_HFEM._ephemerides_parent_point
    @test parent(301) == 3
    @test parent(399) == 3
    for body in (199, 299, 499, 599, 699, 799, 899, 999)
        @test parent(body) == div(body, 100)
    end
    @test isnothing(parent(3))
    @test isnothing(parent(-60000))
    @test isnothing(parent(401))      # Phobos is not on the DE planet tree

    direct = _ephA_HFEM._ephemerides_direct_ssb_point
    @test direct(10)
    @test all(direct, 1:9)
    @test !direct(0)
    @test !direct(399)
    @test !direct(-60000)

    chain = _ephA_HFEM._ephemerides_common_ssb_chain_point
    @test chain(0)
    @test chain(5)
    @test chain(10)
    @test chain(599)
    @test chain(301)
    @test !chain(-60000)
    @test !chain(401)
end


@testset "provider fallback chain (mock provider, n = $(v.n))" for v in _ephA_variants
    n = v.n
    et = 123.456

    # identical points / SSB / bridge itself: zeros without querying the provider
    m = _EphAMockProvider([])
    @test v.fallback(m, 399, 399, et) == zeros(n)
    @test v.ssb(m, 0, et) == zeros(n)
    @test v.bridge(m, 3, 3, et) == zeros(n)
    @test isempty(m.calls)

    # SSB recursion through the DE tree: 301 -> 3 -> SSB
    m = _EphAMockProvider([(0, 3), (3, 301)])
    @test v.ssb(m, 301, et) ≈ _ephA_expected(0, 301, et, n)
    @test m.calls == [(0, 3), (3, 301)]

    # point outside the DE tree is asked directly w.r.t. the SSB
    m = _EphAMockProvider([(0, -100)])
    @test v.ssb(m, -100, et) ≈ _ephA_expected(0, -100, et, n)
    @test m.calls == [(0, -100)]

    # both points on the DE tree: resolved through the SSB without a direct query
    m = _EphAMockProvider([(0, 3), (3, 399), (3, 301)])
    @test v.fallback(m, 399, 301, et) ≈ _ephA_expected(399, 301, et, n)
    @test !((399, 301) in m.calls)

    # DE-tree route fails (no EMB w.r.t. SSB) -> direct query succeeds
    m = _EphAMockProvider([(399, 301)])
    @test v.fallback(m, 399, 301, et) ≈ _ephA_expected(399, 301, et, n)
    @test m.calls[end] == (399, 301)

    # non-tree target available directly
    m = _EphAMockProvider([(399, -60000)])
    @test v.fallback(m, 399, -60000, et) ≈ _ephA_expected(399, -60000, et, n)
    @test m.calls == [(399, -60000)]

    # direct query fails -> both points resolved w.r.t. the SSB
    m = _EphAMockProvider([(0, -100), (0, 3), (3, 399)])
    @test v.fallback(m, -100, 399, et) ≈ _ephA_expected(-100, 399, et, n)
    @test m.calls[1] == (-100, 399)
    @test (0, -100) in m.calls

    # direct and SSB routes fail -> EMB (3) bridge; `from` is not available w.r.t. the
    # bridge either, so `_wrt_bridge` itself falls back to its SSB difference
    m = _EphAMockProvider([(0, -100), (0, 3), (3, -200)])
    @test v.fallback(m, -100, -200, et) ≈ _ephA_expected(-100, -200, et, n)
    @test (3, -100) in m.calls      # bridge-direct attempt for `from` (failed)
    @test (3, -200) in m.calls      # bridge-direct query for `to` (succeeded)
    @test v.bridge(m, -100, 3, et) ≈ _ephA_expected(3, -100, et, n)
    @test v.bridge(m, -200, 3, et) ≈ _ephA_expected(3, -200, et, n)

    # nothing works -> the error of the *direct* query is rethrown
    m = _EphAMockProvider([(0, 3)])
    @test_throws _EphAMissingPair v.fallback(m, -100, -200, et)
    err = try
        v.fallback(m, -100, -200, et)
    catch e
        e
    end
    @test err isa _EphAMissingPair
    @test (err.from, err.to) == (-100, -200)
end


@testset "get_pos/get_state_ephemerides with a provider (mock)" begin
    et = -42.0
    m = _EphAMockProvider([(0, 3), (3, 399), (3, 301)])
    # target = 301, center = 399
    @test _ephA_HFEM.get_pos_ephemerides(m, 301, 399, et) ≈ _ephA_expected(399, 301, et, 3)
    @test _ephA_HFEM.get_pos_ephemerides(m, "301", "399", et) ≈ _ephA_expected(399, 301, et, 3)
    @test _ephA_HFEM.get_state_ephemerides(m, 301, 399, et) ≈ _ephA_expected(399, 301, et, 6)
    @test _ephA_HFEM.get_state_ephemerides(m, "301", "399", et) ≈ _ephA_expected(399, 301, et, 6)
    @test _ephA_HFEM.get_pos_ephemerides(m, Int32(399), Int32(301), et) ≈ _ephA_expected(301, 399, et, 3)

    @test_throws ErrorException _ephA_HFEM.get_pos_ephemerides(nothing, 301, 399, et)
    @test_throws ErrorException _ephA_HFEM.get_state_ephemerides(nothing, 301, 399, et)
end


@testset "get_pos/get_state_ephemerides with a provider vs SPICE" begin
    k = _ephA_spice_kernels()
    provider = _ephA_Eph.EphemerisProvider([k.spk, k.sc])
    et = 8.0e8   # inside the coverage of the -60000 spacecraft kernel

    for (target, center) in (("301", "399"), ("10", "399"), ("399", "10"), ("5", "301"),
                             ("-60000", "399"), ("399", "-60000"))
        r_spice, _ = spkpos(target, et, "J2000", "NONE", center)
        x_spice, _ = spkezr(target, et, "J2000", "NONE", center)

        r_int = _ephA_HFEM.get_pos_ephemerides(provider, parse(Int, target), parse(Int, center), et)
        x_int = _ephA_HFEM.get_state_ephemerides(provider, parse(Int, target), parse(Int, center), et)
        r_str = _ephA_HFEM.get_pos_ephemerides(provider, target, center, et)
        x_str = _ephA_HFEM.get_state_ephemerides(provider, target, center, et)

        scale = max(norm(r_spice), 1.0)
        @test isapprox(collect(r_int), r_spice; atol = 1e-12 * scale + 1e-6)
        @test isapprox(collect(x_int)[1:3], x_spice[1:3]; atol = 1e-12 * scale + 1e-6)
        @test isapprox(collect(x_int)[4:6], x_spice[4:6]; atol = 1e-9)
        @test collect(r_str) == collect(r_int)
        @test collect(x_str) == collect(x_int)
    end

    # same point: zero vector
    @test _ephA_HFEM.get_state_ephemerides(provider, 399, 399, et) == zeros(6)
end


@testset "EphemeridesBackend construction" begin
    k = _ephA_spice_kernels()
    et = 8.0e8

    err = try
        _ephA_HFEM.EphemeridesBackend(nothing); nothing
    catch e
        e
    end
    @test err isa ErrorException && occursin("Pass `ephemerides_provider` or `ephemerides_files`.", err.msg)
    @test_throws ErrorException _ephA_HFEM.build_ephemerides_frame_system(nothing)

    # provider only -> frame system built from it, with the requested body-fixed axes
    provider = _ephA_Eph.EphemerisProvider([k.spk, k.moon_bpc])
    backend = _ephA_HFEM.EphemeridesBackend(provider; frame_PCPF = "MOON_PA")
    @test backend.provider === provider
    @test backend.frame_system isa _ephA_FT.FrameSystem
    @test _ephA_min_cache_length(provider) >= _ephA_HFEM._ephemerides_cache_slots_needed()
    r_spice, _ = spkpos("399", et, "J2000", "NONE", "301")
    @test isapprox(collect(_ephA_HFEM.get_pos_ephemerides(backend.frame_system, 399, 301, et)),
        r_spice; rtol = 1e-12)
    R = _ephA_FT.rotation3(backend.frame_system, :ICRF, :MOON_PA, et)[1]
    @test isapprox(Matrix(R), pxform("J2000", "MOON_PA", et); atol = 1e-12)

    # user-supplied frame system backed by a *different* provider: it is used as is,
    # and the provider reachable from it gets its per-thread caches padded
    provider_fs = _ephA_Eph.EphemerisProvider(k.spk)
    frames = _ephA_HFEM.build_ephemerides_frame_system(provider_fs)
    backend2 = _ephA_HFEM.EphemeridesBackend(provider; frame_system = frames)
    @test backend2.provider === provider
    @test backend2.frame_system === frames
    found = _ephA_HFEM.find_ephemerides_providers(frames)
    @test any(p -> p === provider_fs, found)
    @test _ephA_min_cache_length(provider_fs) >= _ephA_HFEM._ephemerides_cache_slots_needed()
end


@testset "frame system: non-DE points and registration failures" begin
    k = _ephA_spice_kernels()
    et = 8.0e8

    # the spacecraft (-60000 w.r.t. 399) is not in the priority list and is registered
    # by the generic pass once its parent is available
    provider = _ephA_Eph.EphemerisProvider([k.spk, k.sc])
    frames = _ephA_HFEM.build_ephemerides_frame_system(provider)
    @test _ephA_FT.has_point(frames, -60000)
    for center in (399, 301, 10)
        r_spice, _ = spkpos("-60000", et, "J2000", "NONE", string(center))
        r = _ephA_HFEM.get_pos_ephemerides(frames, -60000, center, et; axes = "J2000")
        @test isapprox(collect(r), r_spice; rtol = 1e-12, atol = 1e-6)
    end
    x_spice, _ = spkezr("-60000", et, "ECLIPJ2000", "NONE", "301")
    x = _ephA_HFEM.get_state_ephemerides(frames, "-60000", "301", et; axes = "ECLIPJ2000")
    @test isapprox(collect(x)[1:3], x_spice[1:3]; rtol = 1e-12, atol = 1e-6)
    @test isapprox(collect(x)[4:6], x_spice[4:6]; rtol = 1e-10, atol = 1e-9)

    # a spacecraft-only kernel: neither 399 nor -60000 can be attached to the SSB
    provider_sc = _ephA_Eph.EphemerisProvider(k.sc)
    frames_sc = @test_logs (:warn, r"could not be registered") match_mode = :any begin
        _ephA_HFEM.build_ephemerides_frame_system(provider_sc)
    end
    @test frames_sc isa _ephA_FT.FrameSystem
    @test !_ephA_FT.has_point(frames_sc, -60000)
    @test !_ephA_FT.has_point(frames_sc, 399)
    @test _ephA_FT.has_point(frames_sc, 0)

    # single-point registration helper
    @test _ephA_HFEM._try_add_point_ephemeris!(frames, provider, 0)          # SSB root
    @test _ephA_HFEM._try_add_point_ephemeris!(frames, provider, 399)        # already there
    @test !_ephA_HFEM._try_add_point_ephemeris!(frames, provider, 12345)     # not in provider
    @test !_ephA_FT.has_point(frames, 12345)

    # `_frame_has_point` reports `false` instead of propagating lookup errors
    @test _ephA_HFEM._frame_has_point(frames, 399)
    @test !_ephA_HFEM._frame_has_point(frames, 12345)
    @test !_ephA_HFEM._frame_has_point(nothing, 399)
end


@testset "axes helpers" begin
    @test _ephA_HFEM.ephemerides_axes_symbol(:MOON_PA) === :MOON_PA
    @test _ephA_HFEM.ephemerides_axes_symbol(Int32(31008)) === 31008
    @test _ephA_HFEM.ephemerides_axes_symbol(17) === 17
    @test _ephA_HFEM.ephemerides_axes_symbol("j2000") === :ICRF
    @test _ephA_HFEM.ephemerides_axes_symbol("iau_earth") === :IAU_EARTH
    @test _ephA_HFEM.ephemerides_axes_symbol("MY_FRAME") === :MY_FRAME

    k = _ephA_spice_kernels()
    provider_moon = _ephA_Eph.EphemerisProvider(k.moon_bpc)
    # MOON_PA (31008) is defined relative to ICRF (1) in the lunar binary PCK
    @test _ephA_HFEM._ephemerides_orientation_parent(provider_moon, 31008) == 1
    # ITRF93 orientation data is not in the lunar PCK
    @test isnothing(_ephA_HFEM._ephemerides_orientation_parent(provider_moon, 3000))
end


function _ephA_all_segment_caches(provider)
    out = AbstractVector[]
    for daf in provider.files, fname in fieldnames(typeof(daf.seglist))
        segs = getfield(daf.seglist, fname)
        segs isa AbstractVector || continue
        for seg in segs
            hasfield(typeof(seg), :cache) || continue
            caches = getfield(seg, :cache)
            (caches isa AbstractVector && !isempty(caches)) && push!(out, caches)
        end
    end
    return out
end


@testset "thread-cache padding creates independent caches" begin
    k = _ephA_spice_kernels()
    provider = _ephA_Eph.EphemerisProvider(k.spk)
    nslots = Threads.maxthreadid() + 3
    @test _ephA_HFEM.ensure_ephemerides_thread_caches!(provider; nslots = nslots) === provider
    all_caches = _ephA_all_segment_caches(provider)
    @test !isempty(all_caches)
    for c in all_caches
        @test length(c) >= nslots
        # padded slots must be deep copies, otherwise threads would share (and race on) one cache
        @test allunique(map(objectid, c))
    end
    # the provider is still usable and consistent with SPICE after padding
    et = 8.0e8
    r_spice, _ = spkpos("301", et, "J2000", "NONE", "399")
    @test isapprox(collect(_ephA_HFEM.get_pos_ephemerides(provider, 301, 399, et)), r_spice; rtol = 1e-12)
end
