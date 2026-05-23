"""Interpolate ephemeris"""


"""
InterpolatedEphemeris struct

# Fields
- `naif_id::String`: NAIF ID of the body
- `et_range::Tuple{Float64, Float64}`: span of epochs to interpolate
- `splines::Array{Spline1D, 1}`: splines for the interpolated ephemeris
- `rescale_epoch::Bool`: whether to rescale the epoch to the canonical time unit
- `tstar::Float64`: canonical time unit

# Arguments
- `naif_id::String`: NAIF ID of the body
- `ets::Vector{Float64}`: epochs to interpolate
- `rvs::Array{Float64, 2}`: position and velocity vectors of the body, in km and km/s
- `rescale_epoch::Bool`: whether to rescale the epoch to the canonical time unit
- `tstar::Float64`: canonical time unit
- `spline_order::Int`: order of the spline
"""
struct InterpolatedEphemeris
    naif_id::String
    et_range::Tuple{Float64, Float64}
    splines::Array{Spline1D, 1}
    rescale_epoch::Bool
    tstar::Float64

    function InterpolatedEphemeris(
        naif_id::String,
        ets,
        rvs,
        rescale_epoch::Bool,
        tstar::Float64,
        spline_order::Int = 3,
    )
        @assert 1 <= spline_order <= 5
        if rescale_epoch
            times_input = (ets .- ets[1]) / tstar
        else
            times_input = ets
        end
        splines = [
            Spline1D(times_input, rvs[1,:]; k=spline_order, bc="error"),
            Spline1D(times_input, rvs[2,:]; k=spline_order, bc="error"),
            Spline1D(times_input, rvs[3,:]; k=spline_order, bc="error"),
            Spline1D(times_input, rvs[4,:]; k=spline_order, bc="error"),
            Spline1D(times_input, rvs[5,:]; k=spline_order, bc="error"),
            Spline1D(times_input, rvs[6,:]; k=spline_order, bc="error"),
        ]
        new(naif_id, (ets[1], ets[end]), splines, rescale_epoch, tstar)
    end
end


"""
Overload method for showing InterpolatedEphemeris
"""
function Base.show(io::IO, ephem::InterpolatedEphemeris)
    println("Interpolated ephemeris struct")
    @printf("    et0        : %s (%1.8f)\n", et2utc(ephem.et_range[1], "ISOC", 3), ephem.et_range[1])
    @printf("    etf        : %s (%1.8f)\n", et2utc(ephem.et_range[2], "ISOC", 3), ephem.et_range[2])
    @printf("    naif_id    : %s\n", ephem.naif_id)
end


"""
Interpolate ephemeris position at a given epoch

# Arguments
- `ephem::InterpolatedEphemeris`: interpolated ephemeris struct
- `et::Float64`: epoch to interpolate
"""
function get_pos(ephem::InterpolatedEphemeris, et::Float64)
    if ephem.rescale_epoch
        @assert ephem.et_range[1] <= et <= ephem.et_range[2]
        et_eval = (et - ephem.et_range[1]) / ephem.tstar
    else
        et_eval = et
        @assert ephem.et_range[1] <= et <= ephem.et_range[2]
    end
    return [Dierckx.evaluate(ephem.splines[1], et_eval),
            Dierckx.evaluate(ephem.splines[2], et_eval),
            Dierckx.evaluate(ephem.splines[3], et_eval)]
end


"""Interpolate ephemeris state at a given epoch

# Arguments
- `ephem::InterpolatedEphemeris`: interpolated ephemeris struct
- `et::Float64`: epoch to interpolate
"""
function get_state(ephem::InterpolatedEphemeris, et::Float64)
    if ephem.rescale_epoch
        @assert ephem.et_range[1] <= et <= ephem.et_range[2]
        et_eval = (et - ephem.et_range[1]) / ephem.tstar
    else
        et_eval = et
        @assert ephem.et_range[1] <= et <= ephem.et_range[2]
    end
    return [Dierckx.evaluate(ephem.splines[1], et_eval),
            Dierckx.evaluate(ephem.splines[2], et_eval),
            Dierckx.evaluate(ephem.splines[3], et_eval),
            Dierckx.evaluate(ephem.splines[4], et_eval),
            Dierckx.evaluate(ephem.splines[5], et_eval),
            Dierckx.evaluate(ephem.splines[6], et_eval)]
end

