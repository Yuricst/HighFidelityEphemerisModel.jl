"""
    _ephemerides_parent_point(point)

Return the SPK tree parent used by common DE planetary ephemerides.

These provider-only helpers are retained for backward compatibility. The
preferred path for arbitrary target/center queries in the dynamics is the
`FrameTransformations.FrameSystem` method below.
"""
@inline function _ephemerides_parent_point(point::Int)
    if point == 301 || point == 399
        return 3
    elseif point in (199, 299, 499, 599, 699, 799, 899, 999)
        return div(point, 100)
    else
        return nothing
    end
end


@inline _ephemerides_direct_ssb_point(point::Int) = point == 10 || 1 <= point <= 9


@inline function _ephemerides_common_ssb_chain_point(point::Int)
    return point == 0 || _ephemerides_direct_ssb_point(point) || !isnothing(_ephemerides_parent_point(point))
end


function _ephem_vector3_wrt_ssb(provider, point::Int, et::Number)
    point == 0 && return zeros(3)

    if _ephemerides_direct_ssb_point(point)
        return Ephemerides.ephem_vector3(provider, 0, point, et)
    end

    parent = _ephemerides_parent_point(point)
    if !isnothing(parent)
        parent_ssb = _ephem_vector3_wrt_ssb(provider, parent, et)
        point_parent = Ephemerides.ephem_vector3(provider, parent, point, et)
        return parent_ssb + point_parent
    end

    return Ephemerides.ephem_vector3(provider, 0, point, et)
end


function _ephem_vector6_wrt_ssb(provider, point::Int, et::Number)
    point == 0 && return zeros(6)

    if _ephemerides_direct_ssb_point(point)
        return Ephemerides.ephem_vector6(provider, 0, point, et)
    end

    parent = _ephemerides_parent_point(point)
    if !isnothing(parent)
        parent_ssb = _ephem_vector6_wrt_ssb(provider, parent, et)
        point_parent = Ephemerides.ephem_vector6(provider, parent, point, et)
        return parent_ssb + point_parent
    end

    return Ephemerides.ephem_vector6(provider, 0, point, et)
end


function _ephem_vector3_wrt_bridge(provider, point::Int, bridge::Int, et::Number)
    point == bridge && return zeros(3)

    try
        return Ephemerides.ephem_vector3(provider, bridge, point, et)
    catch
        point_ssb = _ephem_vector3_wrt_ssb(provider, point, et)
        bridge_ssb = _ephem_vector3_wrt_ssb(provider, bridge, et)
        return point_ssb - bridge_ssb
    end
end


function _ephem_vector6_wrt_bridge(provider, point::Int, bridge::Int, et::Number)
    point == bridge && return zeros(6)

    try
        return Ephemerides.ephem_vector6(provider, bridge, point, et)
    catch
        point_ssb = _ephem_vector6_wrt_ssb(provider, point, et)
        bridge_ssb = _ephem_vector6_wrt_ssb(provider, bridge, et)
        return point_ssb - bridge_ssb
    end
end


function _ephem_vector3_with_fallback(provider, from::Int, to::Int, et::Number)
    from == to && return zeros(3)

    if _ephemerides_common_ssb_chain_point(from) && _ephemerides_common_ssb_chain_point(to)
        try
            from_ssb = _ephem_vector3_wrt_ssb(provider, from, et)
            to_ssb = _ephem_vector3_wrt_ssb(provider, to, et)
            return to_ssb - from_ssb
        catch
        end
    end

    direct_error = nothing
    try
        return Ephemerides.ephem_vector3(provider, from, to, et)
    catch error
        direct_error = error
    end

    try
        from_ssb = _ephem_vector3_wrt_ssb(provider, from, et)
        to_ssb = _ephem_vector3_wrt_ssb(provider, to, et)
        return to_ssb - from_ssb
    catch
    end

    try
        bridge = 3
        from_bridge = _ephem_vector3_wrt_bridge(provider, from, bridge, et)
        to_bridge = _ephem_vector3_wrt_bridge(provider, to, bridge, et)
        return to_bridge - from_bridge
    catch
        throw(direct_error)
    end
end


function _ephem_vector6_with_fallback(provider, from::Int, to::Int, et::Number)
    from == to && return zeros(6)

    if _ephemerides_common_ssb_chain_point(from) && _ephemerides_common_ssb_chain_point(to)
        try
            from_ssb = _ephem_vector6_wrt_ssb(provider, from, et)
            to_ssb = _ephem_vector6_wrt_ssb(provider, to, et)
            return to_ssb - from_ssb
        catch
        end
    end

    direct_error = nothing
    try
        return Ephemerides.ephem_vector6(provider, from, to, et)
    catch error
        direct_error = error
    end

    try
        from_ssb = _ephem_vector6_wrt_ssb(provider, from, et)
        to_ssb = _ephem_vector6_wrt_ssb(provider, to, et)
        return to_ssb - from_ssb
    catch
    end

    try
        bridge = 3
        from_bridge = _ephem_vector6_wrt_bridge(provider, from, bridge, et)
        to_bridge = _ephem_vector6_wrt_bridge(provider, to, bridge, et)
        return to_bridge - from_bridge
    catch
        throw(direct_error)
    end
end


"""
    EphemeridesBackend(provider, frame_system)

Small container for Julia-native ephemeris state.
"""
struct EphemeridesBackend{P,F}
    provider::P
    frame_system::F
end


function EphemeridesBackend(provider; frame_PCPF::Union{Nothing,String}=nothing, frame_system=nothing)
    isnothing(provider) && error(
        "No Ephemerides.jl provider was supplied. Pass `ephemerides_provider` or `ephemerides_files`."
    )

    ensure_ephemerides_thread_caches!(provider)

    if isnothing(frame_system)
        frame_system = build_ephemerides_frame_system(provider, frame_PCPF)
    else
        # a user-supplied frame system may be backed by other provider(s)
        ensure_ephemerides_thread_caches!(frame_system)
    end

    return EphemeridesBackend(provider, frame_system)
end


@inline _ephemerides_cache_slots_needed() =
    isdefined(Threads, :maxthreadid) ? Threads.maxthreadid() : Threads.nthreads()


"""
    ensure_ephemerides_thread_caches!(x; nslots = Threads.maxthreadid())

Pad Ephemerides.jl SPK/PCK segment caches to `nslots` (default
`Threads.maxthreadid()`) entries.

Ephemerides allocates `Threads.nthreads()` cache slots but indexes them with
`Threads.threadid()`. With an interactive thread pool (the default from Julia
1.12 on) those IDs are not `1:nthreads()`, so `@threads` / `EnsembleThreads`
workers can `BoundsError` on the cache vector.

`x` can be an `Ephemerides.EphemerisProvider`, `nothing` (no-op), or any other
object such as a `FrameTransformations.FrameSystem`. In the latter case every
`EphemerisProvider` reachable from `x` through its fields is padded. For a
frame system built with `FrameTransformations.add_point_ephemeris!` /
`add_axes_ephemeris!` (as `build_ephemerides_frame_system` does), the providers
are captured by the closures stored in the point/axes nodes and are therefore
found. A provider that is only referenced through a global variable (e.g. from
a user-defined `add_point_dynamical!` function) cannot be found this way; pass
it explicitly as `ephemerides_provider` in that case.

Returns `x`. Padding only covers thread IDs that exist when this is called.
"""
function ensure_ephemerides_thread_caches!(
    provider::Ephemerides.EphemerisProvider; nslots::Integer = _ephemerides_cache_slots_needed()
)
    for daf in provider.files
        for fname in fieldnames(typeof(daf.seglist))
            segs = getfield(daf.seglist, fname)
            segs isa AbstractVector || continue
            for seg in segs
                hasfield(typeof(seg), :cache) || continue
                caches = getfield(seg, :cache)
                caches isa AbstractVector || continue
                isempty(caches) && continue
                while length(caches) < nslots
                    push!(caches, deepcopy(caches[1]))
                end
            end
        end
    end
    return provider
end

ensure_ephemerides_thread_caches!(::Nothing; nslots::Integer = 0) = nothing

function ensure_ephemerides_thread_caches!(x; nslots::Integer = _ephemerides_cache_slots_needed())
    for provider in find_ephemerides_providers(x)
        ensure_ephemerides_thread_caches!(provider; nslots = nslots)
    end
    return x
end


@inline function _skip_in_provider_search(obj)
    return isbits(obj) || obj isa Module || obj isa Type || obj isa Symbol ||
        obj isa AbstractString || obj isa Core.MethodInstance || obj isa Method ||
        obj isa Core.CodeInstance || obj isa Task
end


"""
    find_ephemerides_providers(x; max_objects = 10^6)

Return the distinct `Ephemerides.EphemerisProvider`s reachable from `x` by
following struct fields and array elements (including closure captures, e.g.
the point/axes functions stored in a `FrameTransformations.FrameSystem`).
"""
function find_ephemerides_providers(root; max_objects::Integer = 10^6)
    found = Ephemerides.EphemerisProvider[]
    seen = IdDict{Any,Nothing}()
    stack = Any[root]
    while !isempty(stack) && length(seen) < max_objects
        obj = pop!(stack)
        _skip_in_provider_search(obj) && continue
        haskey(seen, obj) && continue
        seen[obj] = nothing

        if obj isa Ephemerides.EphemerisProvider
            push!(found, obj)
            continue
        end

        if obj isa AbstractArray || (isdefined(Core, :GenericMemory) && obj isa Core.GenericMemory)
            isbitstype(eltype(obj)) && continue
            if obj isa Array || (isdefined(Core, :GenericMemory) && obj isa Core.GenericMemory)
                for i in eachindex(obj)
                    isassigned(obj, i) && push!(stack, obj[i])
                end
                continue
            end
        end

        for i in 1:nfields(obj)
            isdefined(obj, i) && push!(stack, getfield(obj, i))
        end
    end
    return found
end


@inline ephemerides_backend(params) = params.ephemerides_backend


"""
    ephemerides_axes_symbol(frame_name)

Map common SPICE frame names to FrameTransformations axes symbols.

SPICE treats `J2000` as the ICRF/J2000 inertial axes for the use cases in this
package, so the Ephemerides/FrameTransformations backend maps `"J2000"` to
`:ICRF`.
"""
function ephemerides_axes_symbol(frame_name::String)
    # Non-allocating (ASCII case-insensitive comparisons): this is also called by the
    # equations of motion on every evaluation.
    if _ascii_iequal(frame_name, "J2000") || _ascii_iequal(frame_name, "ICRF")
        return :ICRF
    elseif _ascii_iequal(frame_name, "EME2000")
        return :EME2000
    else
        for (sym, name) in _EPHEMERIDES_KNOWN_AXES
            _ascii_iequal(frame_name, name) && return sym
        end
        return Symbol(frame_name)
    end
end

# body-fixed / other axes whose FrameTransformations name is their upper-case SPICE name
const _EPHEMERIDES_KNOWN_AXES = (
    (:MOON_PA, "MOON_PA"), (:ITRF93, "ITRF93"), (:IAU_EARTH, "IAU_EARTH"), (:ECLIPJ2000, "ECLIPJ2000"),
)


# ASCII case-insensitive string equality, without allocating
@inline function _ascii_iequal(a::AbstractString, b::AbstractString)
    ncodeunits(a) == ncodeunits(b) || return false
    @inbounds for i in 1:ncodeunits(a)
        ca, cb = codeunit(a, i), codeunit(b, i)
        ca = UInt8('a') <= ca <= UInt8('z') ? ca - 0x20 : ca
        cb = UInt8('a') <= cb <= UInt8('z') ? cb - 0x20 : cb
        ca == cb || return false
    end
    return true
end

ephemerides_axes_symbol(frame_name::Symbol) = frame_name
ephemerides_axes_symbol(frame_name::Integer) = Int(frame_name)


@inline ephemerides_point_id(id::Integer) = Int(id)
@inline ephemerides_point_id(id::String) = parse(Int, id)


"""
    get_pos_ephemerides(provider, target_id, center_id, et)

Query a directly available Ephemerides.jl position vector.

This method is intentionally a thin wrapper around
`Ephemerides.ephem_vector3(provider, center_id, target_id, et)`. It does not
manually reconstruct missing point chains. If the loaded kernels/provider cannot
answer the query, the backend error is surfaced to the caller.
"""
function get_pos_ephemerides(provider, target_id::Integer, center_id::Integer, et::Number)
    isnothing(provider) && error(
        "No Ephemerides.jl provider was supplied. Pass `ephemerides_provider` or `ephemerides_files` to EphemeridesParameters."
    )

    return _ephem_vector3_with_fallback(provider, Int(center_id), Int(target_id), et)
end


function get_pos_ephemerides(provider, target_id::String, center_id::String, et::Number)
    return get_pos_ephemerides(provider, parse(Int, target_id), parse(Int, center_id), et)
end


"""
    get_state_ephemerides(provider, target_id, center_id, et)

Query a directly available Ephemerides.jl state vector.

This method is intentionally a thin wrapper around
`Ephemerides.ephem_vector6(provider, center_id, target_id, et)`. It does not
manually reconstruct missing point chains.
"""
function get_state_ephemerides(provider, target_id::Integer, center_id::Integer, et::Number)
    isnothing(provider) && error(
        "No Ephemerides.jl provider was supplied. Pass `ephemerides_provider` or `ephemerides_files` to EphemeridesParameters."
    )

    return _ephem_vector6_with_fallback(provider, Int(center_id), Int(target_id), et)
end


function get_state_ephemerides(provider, target_id::String, center_id::String, et::Number)
    return get_state_ephemerides(provider, parse(Int, target_id), parse(Int, center_id), et)
end


@inline _ephemerides_point_symbol(point::Integer) = Symbol("P$(Int(point))")


function _frame_has_point(frames, point)
    try
        return FrameTransformations.has_point(frames, point)
    catch
        return false
    end
end


function _try_add_point_ephemeris!(frames, provider, point::Int)
    point == 0 && return true
    _frame_has_point(frames, point) && return true

    try
        FrameTransformations.add_point_ephemeris!(
            frames,
            provider,
            _ephemerides_point_symbol(point),
            point,
        )
        return true
    catch
        return false
    end
end


function _register_ephemerides_points!(frames, provider)
    points = Int.(collect(Ephemerides.ephem_get_points(provider)))

    # SSB is the root point and is registered manually before this helper.
    remaining = Set(filter(!=(0), points))

    # Register common DE ephemeris parent points before their children. This is
    # only frame-system setup; it does not reconstruct missing states.
    priority = [10; collect(1:9); 199; 299; 301; 399]

    for point in priority
        if point in remaining && _try_add_point_ephemeris!(frames, provider, point)
            delete!(remaining, point)
        end
    end

    # Generic registration pass for any other points the backend can register.
    while !isempty(remaining)
        progressed = false

        for point in collect(remaining)
            if _try_add_point_ephemeris!(frames, provider, point)
                delete!(remaining, point)
                progressed = true
            end
        end

        if !progressed
            @warn "Some Ephemerides points could not be registered in the FrameTransformations graph" collect(remaining)
            break
        end
    end

    return frames
end


# NAIF frame IDs used for the body-fixed axes supported by `build_ephemerides_frame_system`
const _EPHEM_AXESID_ECLIPJ2000 = 17
const _EPHEM_AXESID_ITRF93 = 3000
const _EPHEM_AXESID_IAU_EARTH = 10013
const _EPHEM_AXESID_MOON_PA_DE440 = 31008

# SPICE's J2000 -> ECLIPJ2000 obliquity (IAU 1976), 84381.448 arcsec
const _EPHEM_ECLIPJ2000_OBLIQUITY_RAD = deg2rad(84381.448 / 3600)

const _EPHEM_SUPPORTED_PCPF_MSG = """
Supported `frame_PCPF` values for the Ephemerides.jl backend:
  "MOON_PA"   - DE440 lunar principal axes; needs a lunar binary PCK in the provider
                (e.g. moon_pa_de440_200625.bpc)
  "ITRF93"    - high-precision Earth body-fixed frame; needs a binary Earth PCK in the
                provider (e.g. earth_latest_high_prec.bpc, earth_000101_*.bpc,
                earth_200101_*_predict.bpc)
  "IAU_EARTH" - IAU 2009 Earth rotation model (same constants as pck00010/pck00011);
                no kernel needed
  "J2000"/"ICRF", "EME2000", "ECLIPJ2000" - inertial axes (always available)
For other frames, build a FrameTransformations.FrameSystem yourself and pass it as
`ephemerides_frame_system`."""


"""
    _iau_earth_dcm(t)

Rotation matrix from ICRF (SPICE J2000) to `IAU_EARTH` at `t` TDB seconds past J2000,
using the IAU 2009 Earth rotational elements (identical in NAIF `pck00010.tpc` and
`pck00011.tpc`; Earth has no nutation/precession trigonometric terms):

    α₀ = 0° - 0.641° T,   δ₀ = 90° - 0.557° T,   W = 190.147° + 360.9856235° d

with `T` in Julian centuries and `d` in days (TDB) past J2000, and
`R = R₃(W) R₁(90° - δ₀) R₃(90° + α₀)` as in SPICE. Generic over `t` so that
FrameTransformations can differentiate it with ForwardDiff for the rotation rate.
"""
function _iau_earth_dcm(t)
    d = t / 86400
    T = d / 36525
    α = deg2rad(-0.641 * T)
    δ = deg2rad(90 - 0.557 * T)
    W = deg2rad(190.147 + 360.9856235 * d)
    return FrameTransformations.angle_to_dcm(π / 2 + α, π / 2 - δ, W, :ZXZ)
end


function _ensure_axes_eclipj2000!(frames)
    FrameTransformations.has_axes(frames, _EPHEM_AXESID_ECLIPJ2000) && return frames
    # Defined directly from ICRF (= SPICE J2000) with the IAU 1976 obliquity, exactly as
    # SPICE's ECLIPJ2000; FrameTransformations' `add_axes_ecl2000!` would also apply the
    # ICRF -> EME2000 frame bias, which SPICE does not.
    FrameTransformations.add_axes_fixedoffset!(
        frames, :ECLIPJ2000, _EPHEM_AXESID_ECLIPJ2000, FrameTransformations.AXESID_ICRF,
        FrameTransformations.angle_to_dcm(_EPHEM_ECLIPJ2000_OBLIQUITY_RAD, :X),
    )
    return frames
end


function _ephemerides_orientation_parent(provider, axes_id::Int)
    for rec in Ephemerides.jEph.ephem_orient_records(provider)
        rec.target == axes_id && return Int(rec.axes)
    end
    return nothing
end


function _add_axes_from_binary_pck!(frames, provider, name::Symbol, axes_id::Int, kernel_hint::String)
    available = Int.(collect(Ephemerides.ephem_get_axes(provider)))
    if !(axes_id in available)
        throw(ArgumentError(
            "frame_PCPF = \"$(name)\" (NAIF frame ID $(axes_id)) requires orientation data " *
            "that is not in the Ephemerides.jl provider: load $(kernel_hint) through " *
            "`ephemerides_files`/`ephemerides_provider`. Axes available in the provider: " *
            "$(available).\n" * _EPHEM_SUPPORTED_PCPF_MSG
        ))
    end
    parent = _ephemerides_orientation_parent(provider, axes_id)
    if parent == _EPHEM_AXESID_ECLIPJ2000
        _ensure_axes_eclipj2000!(frames)
    end
    # binary PCK type 2/3 angles are (φ, δ, w) with R = R₃(w) R₁(δ) R₃(φ)
    FrameTransformations.add_axes_ephemeris!(frames, provider, name, axes_id, :ZXZ)
    return frames
end


"""
    build_ephemerides_frame_system(provider, frame_PCPF=nothing)

Build a FrameTransformations frame system for Ephemerides-backed point and axes
transforms.

The frame system is built with order 2 so it supports both `vector3` and
`vector6` queries. All SPK points of `provider` are registered, together with the
ICRF (`"J2000"`), EME2000 and SPICE-compatible ECLIPJ2000 axes, and the body-fixed
axes named by `frame_PCPF`:

- `"MOON_PA"`: DE440 lunar principal axes from a lunar binary PCK in `provider`
  (e.g. `moon_pa_de440_200625.bpc`).
- `"ITRF93"`: Earth body-fixed axes from a binary Earth PCK in `provider`
  (e.g. `earth_latest_high_prec.bpc`; NAIF frame 3000, defined relative to ECLIPJ2000).
  Matches SPICE `pxform("J2000", "ITRF93", et)` to ~1e-15.
- `"IAU_EARTH"`: IAU 2009 Earth rotation model ([`_iau_earth_dcm`](@ref)); needs no
  kernel and matches SPICE `IAU_EARTH` with `pck00010.tpc`/`pck00011.tpc` loaded.

Any other body-fixed `frame_PCPF` raises an `ArgumentError` listing the supported frames
and the kernels they need; a custom frame system can always be passed to
`EphemeridesParameters` through `ephemerides_frame_system` instead. HFEM does not
otherwise work around missing kernel data: unsupported point/axes queries error.
"""
function build_ephemerides_frame_system(provider, frame_PCPF::Union{Nothing,String}=nothing)
    isnothing(provider) && error(
        "No Ephemerides.jl provider was supplied. Pass `ephemerides_provider` or `ephemerides_files` to EphemeridesParameters."
    )

    frames = FrameTransformations.FrameSystem{2,Float64}()

    FrameTransformations.add_axes_icrf!(frames)

    if isdefined(FrameTransformations, :add_axes_eme2000!)
        try
            FrameTransformations.add_axes_eme2000!(frames)
        catch error
            msg = sprint(showerror, error)
            occursin("already registered", msg) || rethrow(error)
        end
    end
    _ensure_axes_eclipj2000!(frames)

    # Root point for the SPK point graph. The docs use ICRF axes ID 1.
    FrameTransformations.add_point!(frames, :SSB, 0, 1)
    _register_ephemerides_points!(frames, provider)

    if !isnothing(frame_PCPF)
        frame_symbol = ephemerides_axes_symbol(frame_PCPF)

        if frame_symbol == :MOON_PA
            _add_axes_from_binary_pck!(
                frames, provider, :MOON_PA, _EPHEM_AXESID_MOON_PA_DE440,
                "a DE440 lunar binary PCK (e.g. moon_pa_de440_200625.bpc)",
            )
        elseif frame_symbol == :ITRF93
            _add_axes_from_binary_pck!(
                frames, provider, :ITRF93, _EPHEM_AXESID_ITRF93,
                "a binary Earth PCK (e.g. earth_latest_high_prec.bpc)",
            )
        elseif frame_symbol == :IAU_EARTH
            FrameTransformations.add_axes_rotating!(
                frames, :IAU_EARTH, _EPHEM_AXESID_IAU_EARTH, FrameTransformations.AXESID_ICRF,
                _iau_earth_dcm,
            )
        elseif frame_symbol in (:ICRF, :EME2000, :ECLIPJ2000)
            # inertial axes, already registered
        else
            throw(ArgumentError(
                "frame_PCPF = \"$(frame_PCPF)\" is not supported by the Ephemerides.jl " *
                "backend's built-in frame system.\n" * _EPHEM_SUPPORTED_PCPF_MSG
            ))
        end
    end

    return frames
end


"""
    get_pos_ephemerides(frames, target_id, center_id, et; axes=:ICRF)

Query a target body's position relative to a center body through a populated
FrameTransformations frame system.
"""
function get_pos_ephemerides(
    frames::FrameTransformations.FrameSystem,
    target_id::Integer,
    center_id::Integer,
    et::Number;
    axes = :ICRF,
)
    return FrameTransformations.vector3(
        frames,
        ephemerides_point_id(center_id),
        ephemerides_point_id(target_id),
        ephemerides_axes_symbol(axes),
        et,
    )
end


function get_pos_ephemerides(
    frames::FrameTransformations.FrameSystem,
    target_id::String,
    center_id::String,
    et::Number;
    axes = :ICRF,
)
    return get_pos_ephemerides(
        frames,
        parse(Int, target_id),
        parse(Int, center_id),
        et;
        axes = axes,
    )
end


"""
    get_state_ephemerides(frames, target_id, center_id, et; axes=:ICRF)

Query a target body's state relative to a center body through a populated
FrameTransformations frame system.
"""
function get_state_ephemerides(
    frames::FrameTransformations.FrameSystem,
    target_id::Integer,
    center_id::Integer,
    et::Number;
    axes = :ICRF,
)
    return FrameTransformations.vector6(
        frames,
        ephemerides_point_id(center_id),
        ephemerides_point_id(target_id),
        ephemerides_axes_symbol(axes),
        et,
    )
end


function get_state_ephemerides(
    frames::FrameTransformations.FrameSystem,
    target_id::String,
    center_id::String,
    et::Number;
    axes = :ICRF,
)
    return get_state_ephemerides(
        frames,
        parse(Int, target_id),
        parse(Int, center_id),
        et;
        axes = axes,
    )
end


"""
    pxform_ephemerides(params, frame_from, frame_to, et)

Ephemerides.jl/FrameTransformations-backed replacement for

    SPICE.pxform(frame_from, frame_to, et)

for supported axes in `params.ephemerides_frame_system`.
"""
function pxform_ephemerides(params, frame_from::String, frame_to::String, et::Number)
    backend = ephemerides_backend(params)
    isnothing(backend.frame_system) && error(
        "No Ephemerides.jl frame system was supplied. Pass `ephemerides_frame_system`, or pass `ephemerides_files`/`ephemerides_provider` together with a supported `frame_PCPF`."
    )

    rotation = FrameTransformations.rotation3(
        backend.frame_system,
        ephemerides_axes_symbol(frame_from),
        ephemerides_axes_symbol(frame_to),
        et,
    )

    return Matrix(rotation[1])
end