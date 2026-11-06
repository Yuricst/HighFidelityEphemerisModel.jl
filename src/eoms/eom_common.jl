# Backend-agnostic, allocation-free kernels shared by the N-body equations of motion.
#
# All vector quantities are handled as scalars / tuples, so the in-place
# equations of motion do not allocate (apart from what the backend itself
# allocates, e.g. SPICE.jl's `spkpos`/`pxform` wrappers, the user's `f_density`
# callback, and the spherical-harmonics routine), and the same code works for
# `Float64` and ForwardDiff `Dual` states. No buffers are stored in the
# parameters, so a single parameter object can be shared across threads.


abstract type _HFEMBackend end
struct _SpiceBackend <: _HFEMBackend end
struct _EphemeridesBackend <: _HFEMBackend end

_hfem_backend(::SpiceParameters) = _SpiceBackend()
_hfem_backend(::EphemeridesParameters) = _EphemeridesBackend()


@inline _hfem_et(params, t) = params.et0 + t * params.TU


# Frame-name -> FrameTransformations axes mapping used by the EOMs. This is the same
# (non-allocating, case-insensitive) mapping applied at construction time.
@inline _ephemerides_axes(frame_name) = ephemerides_axes_symbol(frame_name)


# --- third-body positions (canonical units, relative to `naif_ids[1]`), as 3-tuples

@inline function _third_body_pos(::_SpiceBackend, params, i::Int, et)
    naif_ids = params.naif_ids
    pos, _ = spkpos(naif_ids[i], et, params.naif_frame, params.abcorr, naif_ids[1])
    DU = params.DU
    return (pos[1] / DU, pos[2] / DU, pos[3] / DU)
end

@inline function _third_body_pos(::_EphemeridesBackend, params, i::Int, et)
    naif_ids = params.naif_ids
    pos = FrameTransformations.vector3(
        ephemerides_backend(params).frame_system,
        ephemerides_point_id(naif_ids[1]),
        ephemerides_point_id(naif_ids[i]),
        _ephemerides_axes(params.naif_frame),
        et,
    )
    DU = params.DU
    return (pos[1] / DU, pos[2] / DU, pos[3] / DU)
end


# --- inertial (`naif_frame`) -> body-fixed (`frame_PCPF`) rotation, as a row-major 9-tuple

@inline _rowmajor9(M) = (M[1, 1], M[1, 2], M[1, 3], M[2, 1], M[2, 2], M[2, 3], M[3, 1], M[3, 2], M[3, 3])

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

function _pcpf_rotation(::_SpiceBackend, params, et)
    frame_PCPF = params.frame_PCPF
    isnothing(frame_PCPF) && error(
        "`params.frame_PCPF` must be set when drag or spherical harmonics are enabled."
    )
    return _rowmajor9(SPICE.pxform(params.naif_frame, frame_PCPF, et))
end

function _pcpf_rotation(::_EphemeridesBackend, params, et)
    frame_PCPF = params.frame_PCPF
    isnothing(frame_PCPF) && error(
        "EphemeridesParameters require `params.frame_PCPF` when drag or spherical harmonics are enabled."
    )
    frame_system = ephemerides_backend(params).frame_system
    isnothing(frame_system) && error(
        "No Ephemerides.jl frame system was supplied. Pass `ephemerides_frame_system`, or pass " *
        "`ephemerides_files`/`ephemerides_provider` together with a supported `frame_PCPF`."
    )
    rotation = FrameTransformations.rotation3(
        frame_system,
        _ephemerides_axes(params.naif_frame),
        _ephemerides_axes(frame_PCPF),
        et,
    )
    return _rowmajor9(rotation[1])
end


@inline function _rotate3(R::NTuple{9}, x, y, z)
    return (R[1] * x + R[2] * y + R[3] * z,
            R[4] * x + R[5] * y + R[6] * z,
            R[7] * x + R[8] * y + R[9] * z)
end


# --- force-model pieces that call user / other-module code (function barriers)

@inline function _drag_accel(params, R::NTuple{9}, et, rx, ry, rz, vx, vy, vz)
    DU = params.DU
    px, py, pz = _rotate3(R, rx, ry, rz)
    # `f_density(et, r_km)` receives a `Vector` (public callback API)
    r_km = [px * DU, py * DU, pz * DU]
    rho = params.f_density(et, r_km)
    return _drag3(vx, vy, vz, rx, ry, rz, rho, params.TU, params.omega_atm, params.k_drag)
end


function _spherical_harmonics_accel_canonical(params, R::NTuple{9}, rx, ry, rz)
    sh = params.spherical_harmonics_data
    if sh isa AbstractDict
        field = get(sh, "FIELD", nothing)
        GM = get(sh, "GM", nothing)
        Rref = get(sh, "REFERENCE RADIUS", nothing)
        nmax = get(sh, "nmax", nothing)
        # Fast path only while "Cnm"/"Snm" are still the views of this very field (so a
        # replaced coefficient set is never silently ignored) and the scalars are plain
        # Float64/Int (e.g. a Dual GM goes through the generic dictionary path).
        if field isa SphericalHarmonicsField{Float64} && GM isa Float64 && Rref isa Float64 && nmax isa Int &&
                _sh_views_of_same_field(get(sh, "Cnm", nothing), get(sh, "Snm", nothing)) &&
                (get(sh, "Cnm", nothing)::SphericalHarmonicsCoefficients).field === field
            return _spherical_harmonics_accel_field(params, field, GM, Rref, nmax, R, rx, ry, rz)
        end
    end
    return _spherical_harmonics_accel_legacy(params, R, rx, ry, rz)
end


function _spherical_harmonics_accel_field(params, field::SphericalHarmonicsField, GM::Float64, Rref::Float64,
                                          nmax::Int, R::NTuple{9}, rx, ry, rz)
    DU = params.DU
    x, y, z = _rotate3(R, rx * DU, ry * DU, rz * DU)             # inertial -> PCPF, km
    ax, ay, az = spherical_harmonics_accel_normalized(x, y, z, field, GM, Rref, nmax)
    scale = params.VU / params.TU
    # PCPF -> inertial (transpose), then to canonical units
    return ((R[1] * ax + R[4] * ay + R[7] * az) / scale,
            (R[2] * ax + R[5] * ay + R[8] * az) / scale,
            (R[3] * ax + R[6] * ay + R[9] * az) / scale)
end


# The EOM result has the element type of the state (type stability). Coefficients or GM
# of another type (e.g. ForwardDiff duals) are supported when the state carries the same
# type, e.g. `x .+ zero(c)` inside the differentiated closure.
function _sh_convert(::Type{Ta}, a) where {Ta}
    a isa Ta && return a
    try
        return convert(Ta, a)::Ta
    catch
        throw(ArgumentError(
            "the spherical-harmonics acceleration has element type $(typeof(a)), which cannot be " *
            "converted to the state element type $Ta. When the coefficients or GM are e.g. " *
            "ForwardDiff duals, pass a state of the same element type (e.g. `x .+ zero(c)`), or " *
            "call `spherical_harmonics_accel` directly."
        ))
    end
end


# Plain-Dict coefficient data (e.g. user-built `Cnm`/`Snm` dictionaries without a "FIELD").
function _spherical_harmonics_accel_legacy(params, R::NTuple{9}, rx, ry, rz)
    sh = params.spherical_harmonics_data
    DU = params.DU
    T_inr2pcpf = _matrix33(promote(R...))
    rvec = [rx * DU, ry * DU, rz * DU]
    Ta = promote_type(eltype(T_inr2pcpf), eltype(rvec))
    # The dictionary entries are untyped, so this call is dynamically dispatched (once
    # per EOM evaluation). Only heap-allocated arguments are passed so that nothing
    # extra is boxed, and the result is narrowed back to a concrete type below.
    a_SH = spherical_harmonics_accel(
        T_inr2pcpf,
        rvec,
        sh["Cnm"],
        sh["Snm"],
        sh["GM"],
        sh["REFERENCE RADIUS"],
        sh["nmax"],
        params.factorial_alias,
    )
    scale = params.VU / params.TU
    if a_SH isa Vector{Ta}
        return (a_SH[1] / scale, a_SH[2] / scale, a_SH[3] / scale)
    else
        return (_sh_convert(Ta, a_SH[1]) / scale,
                _sh_convert(Ta, a_SH[2]) / scale,
                _sh_convert(Ta, a_SH[3]) / scale)
    end
end


#
#     _hfem_accel(backend, x, params, t, Val(include_sh), Val(include_central))
#
# Acceleration of the N-body (+ SRP, drag and optionally spherical harmonics)
# force model at state `x` (canonical units), returned as a 3-tuple. With
# `include_central = false` only the perturbing (non-Keplerian) part is returned.
# The terms are accumulated in the same order as the original vector
# implementation: central body, then for each body third-body and SRP terms, then
# drag, then spherical harmonics.
@inline function _hfem_accel(backend::_HFEMBackend, x, params, t, ::Val{include_sh}, ::Val{include_central}) where {include_sh,include_central}
    rx, ry, rz = x[1], x[2], x[3]
    vx, vy, vz = x[4], x[5], x[6]
    mus = params.mus
    naif_ids = params.naif_ids
    et = _hfem_et(params, t)

    if include_central
        c = -mus[1] / sqrt(rx * rx + ry * ry + rz * rz)^3
        ax, ay, az = c * rx, c * ry, c * rz
    else
        z = zero(rx * mus[1])
        ax, ay, az = z, z, z
    end

    include_srp = params.include_srp
    for i in eachindex(naif_ids)
        apply_srp = include_srp && naif_ids[i] == "10"
        i == 1 && !apply_srp && continue

        # the central body sits at the origin (needed for Sun-centered SRP)
        sx, sy, sz = i == 1 ? (0.0, 0.0, 0.0) : _third_body_pos(backend, params, i, et)

        if i >= 2
            tx, ty, tz = _third_body_accel3(rx, ry, rz, sx, sy, sz, mus[i])
            ax += tx; ay += ty; az += tz
        end

        if apply_srp
            px, py, pz = _srp_cannonball3(rx, ry, rz, sx, sy, sz, params.k_srp_cannonball)
            ax += px; ay += py; az += pz
        end
    end

    include_drag = params.include_drag
    has_sh = include_sh && !isnothing(params.spherical_harmonics_data)
    if include_drag || has_sh
        R = _pcpf_rotation(backend, params, et)
        if include_drag
            dx_, dy_, dz_ = _drag_accel(params, R, et, rx, ry, rz, vx, vy, vz)
            ax += dx_; ay += dy_; az += dz_
        end
        if has_sh
            hx, hy, hz = _spherical_harmonics_accel_canonical(params, R, rx, ry, rz)
            ax += hx; ay += hy; az += hz
        end
    end

    return (ax, ay, az)
end


@inline function _hfem_eom!(dx, x, params, t, backend::_HFEMBackend, include_sh::Val)
    ax, ay, az = _hfem_accel(backend, x, params, t, include_sh, Val(true))
    dx[1] = x[4]
    dx[2] = x[5]
    dx[3] = x[6]
    dx[4] = ax
    dx[5] = ay
    dx[6] = az
    return nothing
end


@inline function _hfem_eom(x, params, t, backend::_HFEMBackend, include_sh::Val)
    ax, ay, az = _hfem_accel(backend, x, params, t, include_sh, Val(true))
    return [x[4], x[5], x[6], ax, ay, az]
end


#
# Right-hand side of the 42-element state + STM system, with the Jacobian of the
# out-of-place `eom(x, params, t)` evaluated by ForwardDiff.
function _hfem_eom_stm_fd!(dx_stm, x_stm, params, t, eom::F) where {F}
    x = view(x_stm, 1:6)
    dx = eom(x, params, t)
    for k in 1:6
        dx_stm[k] = dx[k]
    end
    A = eom_jacobian_fd(eom, x, 0.0, params, t)
    for j in 1:3, i in 1:3
        A[i, j+3] = i == j ? one(eltype(A)) : zero(eltype(A))   # force identity for linear map
    end
    mul!(reshape(view(dx_stm, 7:42), 6, 6), A, reshape(view(x_stm, 7:42), 6, 6))
    return nothing
end
