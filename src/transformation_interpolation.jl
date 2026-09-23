"""Interpolate transformation matrix"""


function rotmat1(phi::Float64)
    return [1.0  0.0      0.0;
            0.0  cos(phi) sin(phi);
            0.0 -sin(phi) cos(phi)]
end


function rotmat2(phi::Float64)
    return [cos(phi) 0.0 -sin(phi);
            0.0      1.0  0.0;
            sin(phi) 0.0  cos(phi)]
end


function rotmat3(phi::Float64)
    return [ cos(phi) sin(phi) 0.0;
            -sin(phi) cos(phi) 0.0;
             0.0      0.0      1.0]
end


"""
    unwrap_euler_angles!(angles)

Remove 2π discontinuities along each Euler-angle time history in-place.

`angles` is a 3-by-N matrix whose columns are successive samples. SPICE `m2eul`
returns each angle in (-π, π], so a steadily spinning body-fixed frame (for
example `IAU_EARTH`) produces sawtooth jumps. Spline interpolation across those
jumps yields nonsensical rotation matrices between knots; unwrapping restores a
continuous history before the splines are built.
"""
function unwrap_euler_angles!(angles::AbstractMatrix{<:Real})
    size(angles, 1) == 3 || error("Expected a 3-by-N Euler-angle history, got size $(size(angles)).")
    n = size(angles, 2)
    n == 0 && return angles

    @inbounds for component in 1:3
        for i in 2:n
            Δ = angles[component, i] - angles[component, i - 1]
            if Δ > π
                angles[component, i] -= 2π * round(Δ / 2π)
            elseif Δ < -π
                angles[component, i] -= 2π * round(Δ / 2π)
            end
            # Guard residual multi-turn gaps from coarse sampling.
            while angles[component, i] - angles[component, i - 1] > π
                angles[component, i] -= 2π
            end
            while angles[component, i] - angles[component, i - 1] < -π
                angles[component, i] += 2π
            end
        end
    end

    return angles
end


"""
InterpolatedTransformation struct

# Fields
- `et_range::Tuple{Float64, Float64}`: span of epochs to interpolate
- `frame_from::String`: frame from which the transformation is computed
- `frame_to::String`: frame to which the transformation is computed
- `axis_sequence::Tuple{Int, Int, Int}`: sequence of axes for the Euler angles
- `splines::Array{Spline1D, 1}`: splines for the interpolated transformation
- `rescale_epoch::Bool`: whether to rescale the epoch to the canonical time unit
- `TU::Float64`: canonical time unit

# Arguments
- `ets::Vector{Float64}`: epochs to interpolate
- `frame_from::String`: frame from which the transformation is computed
- `frame_to::String`: frame to which the transformation is computed
- `rescale_epoch::Bool`: whether to rescale the epoch to the canonical time unit
- `TU::Float64`: canonical time unit
- `spline_order::Int`: order of the spline
"""
struct InterpolatedTransformation
    et_range::Tuple{Float64, Float64}
    frame_from::String
    frame_to::String
    axis_sequence::Tuple{Int, Int, Int}
    splines::Array{Spline1D, 1}
    rescale_epoch::Bool
    TU::Float64

    function InterpolatedTransformation(
        ets,
        frame_from::String,
        frame_to::String,
        rescale_epoch::Bool,
        TU::Float64;
        spline_order::Int = 3,
    )
        @assert 1 <= spline_order <= 5
        if rescale_epoch
            @warn "rescale_epoch == true is buggy"
            times_input = (ets .- ets[1]) / TU
        else
            times_input = ets
        end
        euler_angles = zeros(3, length(ets))
        axis_sequence = (3, 1, 3)
        for (idx,et) in enumerate(ets)
            T = SPICE.pxform(frame_from, frame_to, et)
            euler_angles[:,idx] .= m2eul(T, axis_sequence...)
        end
        # m2eul angles lie in (-π, π]; unwrap before spline fitting so steadily
        # rotating frames (e.g. IAU_EARTH) do not introduce 2π discontinuities.
        unwrap_euler_angles!(euler_angles)
        splines = [
            Spline1D(times_input, euler_angles[1,:]; k=spline_order, bc="error"),
            Spline1D(times_input, euler_angles[2,:]; k=spline_order, bc="error"),
            Spline1D(times_input, euler_angles[3,:]; k=spline_order, bc="error"),
        ]
        new((ets[1], ets[end]), frame_from, frame_to, axis_sequence, splines, rescale_epoch, TU)
    end
end


"""
Overload method for showing InterpolatedTransformation
"""
function Base.show(io::IO, transformation::InterpolatedTransformation)
    println("Interpolated transformation struct")
    @printf("    et0           : %s (%1.8f)\n", et2utc(transformation.et_range[1], "ISOC", 3), transformation.et_range[1])
    @printf("    etf           : %s (%1.8f)\n", et2utc(transformation.et_range[2], "ISOC", 3), transformation.et_range[2])
    @printf("    frame from    : %s\n", transformation.frame_from)
    @printf("    frame to      : %s\n", transformation.frame_to)
    @printf("    axis sequence : %s\n", transformation.axis_sequence)
end


"""Interpolate Euler angles at a given epoch

# Arguments
- `transformation::InterpolatedTransformation`: interpolated transformation struct
- `et::Float64`: epoch to interpolate
"""
function get_euler_angles(transformation::InterpolatedTransformation, et::Float64)
    if transformation.rescale_epoch
        et_eval = et * transformation.TU + transformation.et_range[1]
        @assert transformation.et_range[1] <= et <= transformation.et_range[2]
    else
        et_eval = et
        @assert transformation.et_range[1] <= et <= transformation.et_range[2]
    end
    euler_angles = [Dierckx.evaluate(transformation.splines[1], et_eval),
                    Dierckx.evaluate(transformation.splines[2], et_eval),
                    Dierckx.evaluate(transformation.splines[3], et_eval)]
    return euler_angles
end


"""Interpolate transformation matrix at a given epoch

# Arguments
- `transformation::InterpolatedTransformation`: interpolated transformation struct
- `et::Float64`: epoch to interpolate
"""
function pxform(transformation::InterpolatedTransformation, et::Float64)
    euler_angles = get_euler_angles(transformation, et)
    T = rotmat3(euler_angles[1]) * rotmat1(euler_angles[2]) * rotmat3(euler_angles[3])
    return T
end