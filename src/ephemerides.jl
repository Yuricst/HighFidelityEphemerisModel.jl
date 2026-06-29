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
        "No Ephemerides.jl provider was supplied. Pass `ephemerides_provider` or `ephemerides_files` to HighFidelityEphemerisModelParameters."
    )

    return Ephemerides.ephem_vector3(
        provider,
        ephemerides_point_id(center_id),
        ephemerides_point_id(target_id),
        et,
    )
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
        "No Ephemerides.jl provider was supplied. Pass `ephemerides_provider` or `ephemerides_files` to HighFidelityEphemerisModelParameters."
    )

    return Ephemerides.ephem_vector6(
        provider,
        ephemerides_point_id(center_id),
        ephemerides_point_id(target_id),
        et,
    )
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


"""
    build_ephemerides_frame_system(provider, frame_PCPF=nothing)

Build a FrameTransformations frame system for Ephemerides-backed point and axes
transforms.

The frame system is built with order 2 so it supports both `vector3` and
`vector6` queries. HFEM does not manually work around missing kernel data: if the
requested point/axes transform is unsupported by the loaded kernels/backend, the
query should error clearly.
"""
function build_ephemerides_frame_system(provider, frame_PCPF::Union{Nothing,String}=nothing)
    isnothing(provider) && error(
        "No Ephemerides.jl provider was supplied. Pass `ephemerides_provider` or `ephemerides_files` to HighFidelityEphemerisModelParameters."
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

    # Root point for the SPK point graph. The docs use ICRF axes ID 1.
    FrameTransformations.add_point!(frames, :SSB, 0, 1)
    _register_ephemerides_points!(frames, provider)

    if !isnothing(frame_PCPF)
        frame_symbol = ephemerides_axes_symbol(frame_PCPF)

        if frame_symbol == :MOON_PA
            FrameTransformations.add_axes_pa440!(frames, provider, frame_symbol)
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
    isnothing(params.ephemerides_frame_system) && error(
        "No Ephemerides.jl frame system was supplied. Pass `ephemerides_frame_system`, or pass `ephemerides_files`/`ephemerides_provider` together with a supported `frame_PCPF`."
    )

    rotation = FrameTransformations.rotation3(
        params.ephemerides_frame_system,
        ephemerides_axes_symbol(frame_from),
        ephemerides_axes_symbol(frame_to),
        et,
    )

    return Matrix(rotation[1])
end