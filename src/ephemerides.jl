"""
    get_pos_ephemerides(provider, target_id, center_id, et)

Query a target body's position relative to a center body using Ephemerides.jl.

This is intended to mirror the position part of

    SPICE.spkpos(target_id, et, frame, abcorr, center_id)

for directly available SPK object pairs.

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

    return Ephemerides.ephem_vector3(provider, from, to, et)
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

    return Ephemerides.ephem_vector6(provider, from, to, et)
end