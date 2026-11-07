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
    _unwrap_angles!(angles)

Remove 2π jumps from a sequence of angles in place, so that consecutive samples
differ by less than π.
"""
function _unwrap_angles!(angles::AbstractVector{<:Real})
    offset = 0.0
    for i in 2:length(angles)
        raw_prev = angles[i-1] - offset
        d = angles[i] - raw_prev
        if d > π
            offset -= 2π
        elseif d < -π
            offset += 2π
        end
        angles[i] += offset
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
        # m2eul returns the first and third angles in [-π, π]; for a rotating body
        # these wrap by 2π every revolution, which a spline cannot follow.
        # Unwrap them so the splined histories are continuous.
        _unwrap_angles!(view(euler_angles, 1, :))
        _unwrap_angles!(view(euler_angles, 3, :))
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
    println(io, "Interpolated transformation struct")
    @printf(io, "    et0           : %s (%1.8f)\n", et2utc(transformation.et_range[1], "ISOC", 3), transformation.et_range[1])
    @printf(io, "    etf           : %s (%1.8f)\n", et2utc(transformation.et_range[2], "ISOC", 3), transformation.et_range[2])
    @printf(io, "    frame from    : %s\n", transformation.frame_from)
    @printf(io, "    frame to      : %s\n", transformation.frame_to)
    @printf(io, "    axis sequence : %s\n", transformation.axis_sequence)
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
    return _matrix33(_pxform_tuple(transformation, et))
end


# 3x3 `Matrix` from a row-major 9-tuple (avoids the boxing of the `[a b c; ...]` literal)
function _matrix33(R::NTuple{9,T}) where {T}
    M = Matrix{T}(undef, 3, 3)
    @inbounds begin
        M[1, 1] = R[1]; M[1, 2] = R[2]; M[1, 3] = R[3]
        M[2, 1] = R[4]; M[2, 2] = R[5]; M[2, 3] = R[6]
        M[3, 1] = R[7]; M[3, 2] = R[8]; M[3, 3] = R[9]
    end
    return M
end


@inline _rotmat1_tuple(s, c) = (one(c), zero(c), zero(c), zero(c), c, s, zero(c), -s, c)
@inline _rotmat3_tuple(s, c) = (c, s, zero(c), -s, c, zero(c), zero(c), zero(c), one(c))

# product of two 3x3 matrices stored as row-major 9-tuples
@inline function _mul33(A::NTuple{9}, B::NTuple{9})
    return (
        A[1] * B[1] + A[2] * B[4] + A[3] * B[7],
        A[1] * B[2] + A[2] * B[5] + A[3] * B[8],
        A[1] * B[3] + A[2] * B[6] + A[3] * B[9],
        A[4] * B[1] + A[5] * B[4] + A[6] * B[7],
        A[4] * B[2] + A[5] * B[5] + A[6] * B[8],
        A[4] * B[3] + A[5] * B[6] + A[6] * B[9],
        A[7] * B[1] + A[8] * B[4] + A[9] * B[7],
        A[7] * B[2] + A[8] * B[5] + A[9] * B[8],
        A[7] * B[3] + A[8] * B[6] + A[9] * B[9],
    )
end


"""
    _pxform_tuple(transformation, et)

Allocation-free form of [`pxform`](@ref); returns the 3x3 rotation as a
row-major 9-tuple.
"""
@inline function _pxform_tuple(transformation::InterpolatedTransformation, et::Float64)
    if transformation.rescale_epoch
        et_eval = et * transformation.TU + transformation.et_range[1]
        @assert transformation.et_range[1] <= et <= transformation.et_range[2]
    else
        et_eval = et
        @assert transformation.et_range[1] <= et <= transformation.et_range[2]
    end
    splines = transformation.splines
    s1, c1 = sincos(Dierckx.evaluate(splines[1], et_eval))
    s2, c2 = sincos(Dierckx.evaluate(splines[2], et_eval))
    s3, c3 = sincos(Dierckx.evaluate(splines[3], et_eval))
    return _mul33(_mul33(_rotmat3_tuple(s1, c1), _rotmat1_tuple(s2, c2)), _rotmat3_tuple(s3, c3))
end
