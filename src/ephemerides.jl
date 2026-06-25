"""
    _ephem_vector3_with_fallback(provider, from, to, et)

Query an Ephemerides.jl position vector with small SPK segment-chain fallbacks.

Ephemerides.jl queries only directly available point pairs. SPICE can concatenate
SPK segments automatically. These fallbacks cover common DE440 chains through the
solar-system barycenter (NAIF 0) and Earth-Moon barycenter (NAIF 3).
"""
function _ephem_vector3_wrt_bridge(provider, point::Int, bridge::Int, et::Number)
    point == bridge && return zeros(3)

    try
        return Ephemerides.ephem_vector3(provider, bridge, point, et)
    catch
        point_ssb = Ephemerides.ephem_vector3(provider, 0, point, et)
        bridge_ssb = Ephemerides.ephem_vector3(provider, 0, bridge, et)
        return point_ssb - bridge_ssb
    end
end


function _ephem_vector3_with_fallback(provider, from::Int, to::Int, et::Number)
    from == to && return zeros(3)

    direct_error = nothing

    try
        return Ephemerides.ephem_vector3(provider, from, to, et)
    catch error
        direct_error = error
    end

    try
        from_ssb = Ephemerides.ephem_vector3(provider, 0, from, et)
        to_ssb = Ephemerides.ephem_vector3(provider, 0, to, et)
        return to_ssb - from_ssb
    catch
    end

    try
        bridge = 3  # Earth-Moon barycenter
        from_bridge = _ephem_vector3_wrt_bridge(provider, from, bridge, et)
        to_bridge = _ephem_vector3_wrt_bridge(provider, to, bridge, et)
        return to_bridge - from_bridge
    catch
        throw(direct_error)
    end
end


"""
    _ephem_vector6_with_fallback(provider, from, to, et)

Query an Ephemerides.jl state vector with small SPK segment-chain fallbacks.
"""
function _ephem_vector6_wrt_bridge(provider, point::Int, bridge::Int, et::Number)
    point == bridge && return zeros(6)

    try
        return Ephemerides.ephem_vector6(provider, bridge, point, et)
    catch
        point_ssb = Ephemerides.ephem_vector6(provider, 0, point, et)
        bridge_ssb = Ephemerides.ephem_vector6(provider, 0, bridge, et)
        return point_ssb - bridge_ssb
    end
end


function _ephem_vector6_with_fallback(provider, from::Int, to::Int, et::Number)
    from == to && return zeros(6)

    direct_error = nothing

    try
        return Ephemerides.ephem_vector6(provider, from, to, et)
    catch error
        direct_error = error
    end

    try
        from_ssb = Ephemerides.ephem_vector6(provider, 0, from, et)
        to_ssb = Ephemerides.ephem_vector6(provider, 0, to, et)
        return to_ssb - from_ssb
    catch
    end

    try
        bridge = 3  # Earth-Moon barycenter
        from_bridge = _ephem_vector6_wrt_bridge(provider, from, bridge, et)
        to_bridge = _ephem_vector6_wrt_bridge(provider, to, bridge, et)
        return to_bridge - from_bridge
    catch
        throw(direct_error)
    end
end


"""

    get_pos_ephemerides(provider, target_id, center_id, et)

Query a target body's position relative to a center body using Ephemerides.jl.

This is intended to mirror the position part of

    SPICE.spkpos(target_id, et, frame, abcorr, center_id)

for common SPK object pairs.

Ephemerides.jl uses integer NAIF IDs and the convention

    ephem_vector3(provider, from, to, et)

where `from` is the center and `to` is the target.
"""
function get_pos_ephemerides(provider, target_id::String, center_id::String, et::Number)
    isnothing(provider) && error(
        "No Ephemerides.jl provider was supplied. Pass `ephemerides_provider` or `ephemerides_files` to HighFidelityEphemerisModelParameters."
    )

    from = parse(Int, center_id)
    to = parse(Int, target_id)

    return _ephem_vector3_with_fallback(provider, from, to, et)
end


"""
    get_state_ephemerides(provider, target_id, center_id, et)

Query a target body's state relative to a center body using Ephemerides.jl.
"""
function get_state_ephemerides(provider, target_id::String, center_id::String, et::Number)
    isnothing(provider) && error(
        "No Ephemerides.jl provider was supplied. Pass `ephemerides_provider` or `ephemerides_files` to HighFidelityEphemerisModelParameters."
    )

    from = parse(Int, center_id)
    to = parse(Int, target_id)

    return _ephem_vector6_with_fallback(provider, from, to, et)
end


"""
    ephemerides_axes_symbol(frame_name)

Map common SPICE frame names to FrameTransformations axes symbols.

SPICE treats `J2000` as the ICRF/J2000 inertial axes for the use cases in this
package, so the Ephemerides/FrameTransformations backend maps `"J2000"` to
`:ICRF`.
"""
function ephemerides_axes_symbol(frame_name::String)
    frame_upper = uppercase(frame_name)

    if frame_upper in ("J2000", "ICRF")
        return :ICRF
    elseif frame_upper == "EME2000"
        return :EME2000
    elseif frame_upper == "MOON_PA"
        return :MOON_PA
    else
        return Symbol(frame_name)
    end
end


"""
    build_ephemerides_frame_system(provider, frame_PCPF=nothing)

Build a FrameTransformations frame system for Ephemerides-backed frame transforms.

Currently, this registers inertial ICRF/EME2000 axes and the DE440 Moon
principal-axes frame when `frame_PCPF == "MOON_PA"`.
"""
function build_ephemerides_frame_system(provider, frame_PCPF::Union{Nothing,String}=nothing)
    isnothing(provider) && error(
        "No Ephemerides.jl provider was supplied. Pass `ephemerides_provider` or `ephemerides_files` to HighFidelityEphemerisModelParameters."
    )

    frames = FrameTransformations.FrameSystem{1,Float64}()
    FrameTransformations.add_axes_icrf!(frames)
    FrameTransformations.add_axes_eme2000!(frames)

    if !isnothing(frame_PCPF)
        frame_symbol = ephemerides_axes_symbol(frame_PCPF)

        if frame_symbol == :MOON_PA
            FrameTransformations.add_axes_pa440!(frames, provider, frame_symbol)
        end
    end

    return frames
end


"""
    pxform_ephemerides(params, frame_from, frame_to, et)

Ephemerides.jl/FrameTransformations-backed replacement for

    SPICE.pxform(frame_from, frame_to, et)

for supported axes in `params.ephemerides_frame_system`.
"""
function pxform_ephemerides(params, frame_from::String, frame_to::String, et::Number)
    isnothing(params.ephemerides_frame_system) && error(
        "No Ephemerides.jl frame system was supplied. Pass `ephemerides_frame_system`, or pass `ephemerides_files`/`ephemerides_provider` together with a supported `frame_PCPF`."
    )

    axes_from = ephemerides_axes_symbol(frame_from)
    axes_to = ephemerides_axes_symbol(frame_to)

    rotation = FrameTransformations.rotation3(
        params.ephemerides_frame_system,
        axes_from,
        axes_to,
        et,
    )

    return Matrix(rotation[1])
end