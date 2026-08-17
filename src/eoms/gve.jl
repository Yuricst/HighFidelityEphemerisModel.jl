"""Gauss variational equations in orbital-element coordinates."""


"""
    pxform_inr2rtn(r, v)

Construct the inertial-to-RTN direction-cosine matrix for a Cartesian position
`r` and velocity `v`. Its rows are the radial direction `r / norm(r)`, the
transverse direction `cross(r, v) / norm(cross(r, v)) × r̂`, and the orbit-normal
direction `cross(r, v) / norm(cross(r, v))`.

The transformation is undefined for a zero position or zero angular momentum
and throws `DomainError` in either case.
"""
function pxform_inr2rtn(r::AbstractVector, v::AbstractVector)
    length(r) == 3 || throw(DimensionMismatch("r must contain three components"))
    length(v) == 3 || throw(DimensionMismatch("v must contain three components"))

    rx, ry, rz = r
    vx, vy, vz = v
    rnorm = sqrt(rx * rx + ry * ry + rz * rz)
    iszero(rnorm) && throw(DomainError(rnorm, "RTN is undefined for zero position"))

    hx = ry * vz - rz * vy
    hy = rz * vx - rx * vz
    hz = rx * vy - ry * vx
    hnorm = sqrt(hx * hx + hy * hy + hz * hz)
    iszero(hnorm) && throw(DomainError(hnorm, "RTN is undefined for zero angular momentum"))

    rhatx, rhaty, rhatz = rx / rnorm, ry / rnorm, rz / rnorm
    nhatx, nhaty, nhatz = hx / hnorm, hy / hnorm, hz / hnorm
    thatx = nhaty * rhatz - nhatz * rhaty
    thaty = nhatz * rhatx - nhatx * rhatz
    thatz = nhatx * rhaty - nhaty * rhatx

    T = Matrix{promote_type(eltype(r), eltype(v))}(undef, 3, 3)
    @inbounds begin
        T[1, 1] = rhatx; T[1, 2] = rhaty; T[1, 3] = rhatz
        T[2, 1] = thatx; T[2, 2] = thaty; T[2, 3] = thatz
        T[3, 1] = nhatx; T[3, 2] = nhaty; T[3, 3] = nhatz
    end
    return T
end


"""Construct the RTN-to-inertial transformation, the transpose of `pxform_inr2rtn`."""
pxform_rtn2inr(r::AbstractVector, v::AbstractVector) = copy(transpose(pxform_inr2rtn(r, v)))


"""
    project_inr_to_rtn!(a_rtn, a_inr, r, v)

Project an inertial acceleration directly onto RTN without constructing a
matrix. Component order is `[radial, transverse, normal]`.
"""
function project_inr_to_rtn!(a_rtn, a_inr, r, v)
    length(a_rtn) == 3 || throw(DimensionMismatch("a_rtn must contain three components"))
    length(a_inr) == 3 || throw(DimensionMismatch("a_inr must contain three components"))
    length(r) == 3 || throw(DimensionMismatch("r must contain three components"))
    length(v) == 3 || throw(DimensionMismatch("v must contain three components"))

    rx, ry, rz = r
    vx, vy, vz = v
    ax, ay, az = a_inr
    rnorm = sqrt(rx * rx + ry * ry + rz * rz)
    iszero(rnorm) && throw(DomainError(rnorm, "RTN is undefined for zero position"))
    hx = ry * vz - rz * vy
    hy = rz * vx - rx * vz
    hz = rx * vy - ry * vx
    hnorm = sqrt(hx * hx + hy * hy + hz * hz)
    iszero(hnorm) && throw(DomainError(hnorm, "RTN is undefined for zero angular momentum"))

    rhatx, rhaty, rhatz = rx / rnorm, ry / rnorm, rz / rnorm
    nhatx, nhaty, nhatz = hx / hnorm, hy / hnorm, hz / hnorm
    thatx = nhaty * rhatz - nhatz * rhaty
    thaty = nhatz * rhatx - nhatx * rhatz
    thatz = nhatx * rhaty - nhaty * rhatx
    @inbounds begin
        a_rtn[1] = rhatx * ax + rhaty * ay + rhatz * az
        a_rtn[2] = thatx * ax + thaty * ay + thatz * az
        a_rtn[3] = nhatx * ax + nhaty * ay + nhatz * az
    end
    return a_rtn
end


function _project_inr_to_rtn_mee!(a_rtn, a_inr, mee)
    length(a_rtn) == 3 || throw(DimensionMismatch("a_rtn must contain three components"))
    length(a_inr) == 3 || throw(DimensionMismatch("a_inr must contain three components"))
    length(mee) == 6 || throw(DimensionMismatch("mee must contain six components"))

    hmee, kmee, longitude = mee[4], mee[5], mee[6]
    sinL, cosL = sincos(longitude)
    s2 = one(hmee) + hmee * hmee + kmee * kmee
    hk2 = 2 * hmee * kmee
    h2mk2 = hmee * hmee - kmee * kmee

    rhatx = ((one(hmee) + h2mk2) * cosL + hk2 * sinL) / s2
    rhaty = ((one(hmee) - h2mk2) * sinL + hk2 * cosL) / s2
    rhatz = 2 * (hmee * sinL - kmee * cosL) / s2
    thatx = (-(one(hmee) + h2mk2) * sinL + hk2 * cosL) / s2
    thaty = ((one(hmee) - h2mk2) * cosL - hk2 * sinL) / s2
    thatz = 2 * (hmee * cosL + kmee * sinL) / s2
    nhatx = 2 * kmee / s2
    nhaty = -2 * hmee / s2
    nhatz = (one(hmee) - hmee * hmee - kmee * kmee) / s2

    ax, ay, az = a_inr
    @inbounds begin
        a_rtn[1] = rhatx * ax + rhaty * ay + rhatz * az
        a_rtn[2] = thatx * ax + thaty * ay + thatz * az
        a_rtn[3] = nhatx * ax + nhaty * ay + nhatz * az
    end
    return a_rtn
end


"""
    gve_mee_derivs!(dmee, mee, a_rtn, mu)

Evaluate the prograde modified-equinoctial Gauss variational equations.
`mee = [p, f, g, h, k, L]` and `a_rtn = [a_radial, a_transverse, a_normal]`.
The central two-body contribution appears only in `L̇`; `a_rtn` must contain
non-Keplerian acceleration only. All quantities must use one consistent unit
system and angles are radians.

This standard prograde MEE convention is singular exactly at retrograde-planar
inclination (`i = pi`). A nonpositive `p` or zero
`w = 1 + f*cos(L) + g*sin(L)` is rejected.
"""
function gve_mee_derivs!(dmee, mee, a_rtn, mu)
    length(dmee) == 6 || throw(DimensionMismatch("dmee must contain six components"))
    length(mee) == 6 || throw(DimensionMismatch("mee must contain six components"))
    length(a_rtn) == 3 || throw(DimensionMismatch("a_rtn must contain three components"))

    pmee, fmee, gmee, hmee, kmee, longitude = mee
    pmee > zero(pmee) || throw(DomainError(pmee, "MEE semi-latus rectum must be positive"))
    mu > zero(mu) || throw(DomainError(mu, "central-body gravitational parameter must be positive"))

    a_radial, a_transverse, a_normal = a_rtn
    sinL, cosL = sincos(longitude)
    w = one(pmee) + fmee * cosL + gmee * sinL
    iszero(w) && throw(DomainError(w, "MEE radius denominator w must be nonzero"))

    s2 = one(pmee) + hmee * hmee + kmee * kmee
    sqrt_p_over_mu = sqrt(pmee / mu)
    normal_coupling = hmee * sinL - kmee * cosL

    @inbounds begin
        dmee[1] = sqrt_p_over_mu * (2 * pmee / w) * a_transverse
        dmee[2] = sqrt_p_over_mu * (
            sinL * a_radial + (((w + one(w)) * cosL + fmee) / w) * a_transverse -
            (gmee * normal_coupling / w) * a_normal
        )
        dmee[3] = sqrt_p_over_mu * (
            -cosL * a_radial + (((w + one(w)) * sinL + gmee) / w) * a_transverse +
            (fmee * normal_coupling / w) * a_normal
        )
        dmee[4] = sqrt_p_over_mu * (s2 * cosL / (2 * w)) * a_normal
        dmee[5] = sqrt_p_over_mu * (s2 * sinL / (2 * w)) * a_normal
        dmee[6] = sqrt(mu * pmee) * (w / pmee)^2 +
            sqrt_p_over_mu * (normal_coupling / w) * a_normal
    end
    return nothing
end


function _perturbing_accel!(a_pert, rv, params, t, ::Val{include_sh}) where {include_sh}
    dstate = include_sh ? eom_NbodySH(rv, params, t) : eom_Nbody(rv, params, t)
    rx, ry, rz = rv[1], rv[2], rv[3]
    r2 = rx * rx + ry * ry + rz * rz
    iszero(r2) && throw(DomainError(r2, "central-body-relative position must be nonzero"))
    central_scale = params.mus[1] / (r2 * sqrt(r2))
    @inbounds begin
        a_pert[1] = dstate[4] + central_scale * rx
        a_pert[2] = dstate[5] + central_scale * ry
        a_pert[3] = dstate[6] + central_scale * rz
    end
    return a_pert
end


function _gve_mee!(dmee, mee, params, t, include_sh)
    mu = params.mus[1]
    rv = AstrodynamicsCore.mee2rv(mee, mu)
    a_pert = similar(rv, 3)
    a_rtn = similar(rv, 3)
    _perturbing_accel!(a_pert, rv, params, t, include_sh)
    project_inr_to_rtn!(a_rtn, a_pert, view(rv, 1:3), view(rv, 4:6))
    return gve_mee_derivs!(dmee, mee, a_rtn, mu)
end


"""MEE Gauss variational equations for the same force model as `eom_Nbody!`."""
gve_mee_Nbody!(dmee, mee, params::AbstractHFEMParameters, t) =
    _gve_mee!(dmee, mee, params, t, Val(false))

"""Out-of-place form of [`gve_mee_Nbody!`](@ref)."""
function gve_mee_Nbody(mee, params::AbstractHFEMParameters, t)
    dmee = similar(mee)
    gve_mee_Nbody!(dmee, mee, params, t)
    return dmee
end

"""MEE Gauss variational equations for the same force model as `eom_NbodySH!`."""
gve_mee_NbodySH!(dmee, mee, params::AbstractHFEMParameters, t) =
    _gve_mee!(dmee, mee, params, t, Val(true))

"""Out-of-place form of [`gve_mee_NbodySH!`](@ref)."""
function gve_mee_NbodySH(mee, params::AbstractHFEMParameters, t)
    dmee = similar(mee)
    gve_mee_NbodySH!(dmee, mee, params, t)
    return dmee
end


"""
    eom_kep_twobody!(dkep, kep, mu, t)

Two-body drift for AstrodynamicsCore's Keplerian ordering
`[a, e, i, Omega, omega, true_anomaly]`. Only true anomaly changes. This model
is useful for reference, validation, simplified analysis, and propagation when
one central body dominates. All quantities must use one consistent unit system
and angles are radians.
"""
function eom_kep_twobody!(dkep, kep, mu, t = zero(eltype(kep)))
    length(dkep) == 6 || throw(DimensionMismatch("dkep must contain six components"))
    length(kep) == 6 || throw(DimensionMismatch("kep must contain six components"))
    semimajor_axis, eccentricity, _, _, _, true_anomaly = kep
    p = semimajor_axis * (one(eccentricity) - eccentricity^2)
    p > zero(p) || throw(DomainError(p, "Keplerian semi-latus rectum must be positive"))
    mu > zero(mu) || throw(DomainError(mu, "central-body gravitational parameter must be positive"))
    radius = p / (one(p) + eccentricity * cos(true_anomaly))
    fill!(dkep, zero(eltype(dkep)))
    dkep[6] = sqrt(mu * p) / radius^2
    return nothing
end


"""Out-of-place form of [`eom_kep_twobody!`](@ref)."""
function eom_kep_twobody(kep, mu, t = zero(eltype(kep)))
    dkep = similar(kep)
    eom_kep_twobody!(dkep, kep, mu, t)
    return dkep
end


"""
    gve_kep_derivs!(dkep, kep, a_rtn, mu)

Classical Keplerian Gauss variational equations for
`[a, e, i, Omega, omega, true_anomaly]`. They are singular for circular
or equatorial orbits and cannot smoothly cross the ill-conditioned parabolic
boundary `e = 1`. `a_rtn` contains non-Keplerian acceleration in
`[radial, transverse, normal]` order. All quantities must use one consistent
unit system and angles are radians.
"""
function gve_kep_derivs!(dkep, kep, a_rtn, mu)
    length(dkep) == 6 || throw(DimensionMismatch("dkep must contain six components"))
    length(kep) == 6 || throw(DimensionMismatch("kep must contain six components"))
    length(a_rtn) == 3 || throw(DimensionMismatch("a_rtn must contain three components"))

    semimajor_axis, eccentricity, inclination, _, argument_periapsis, true_anomaly = kep
    iszero(eccentricity) && throw(DomainError(eccentricity, "Keplerian GVE is singular for circular orbits"))
    sin_i = sin(inclination)
    iszero(sin_i) && throw(DomainError(inclination, "Keplerian GVE is singular for equatorial orbits"))
    p = semimajor_axis * (one(eccentricity) - eccentricity^2)
    p > zero(p) || throw(DomainError(p, "Keplerian semi-latus rectum must be positive"))
    mu > zero(mu) || throw(DomainError(mu, "central-body gravitational parameter must be positive"))

    a_radial, a_transverse, a_normal = a_rtn
    sin_nu, cos_nu = sincos(true_anomaly)
    radius = p / (one(p) + eccentricity * cos_nu)
    hmag = sqrt(mu * p)
    argument_latitude = argument_periapsis + true_anomaly
    sin_u, cos_u = sincos(argument_latitude)

    @inbounds begin
        dkep[1] = 2 * semimajor_axis^2 / hmag * (
            eccentricity * sin_nu * a_radial + (p / radius) * a_transverse)
        dkep[2] = (p * sin_nu * a_radial +
            ((p + radius) * cos_nu + radius * eccentricity) * a_transverse) / hmag
        dkep[3] = radius * cos_u / hmag * a_normal
        dkep[4] = radius * sin_u / (hmag * sin_i) * a_normal
        dkep[5] = (-p * cos_nu * a_radial + (p + radius) * sin_nu * a_transverse) /
            (hmag * eccentricity) - radius * sin_u * cos(inclination) /
            (hmag * sin_i) * a_normal
        dkep[6] = hmag / radius^2 +
            (p * cos_nu * a_radial - (p + radius) * sin_nu * a_transverse) /
            (hmag * eccentricity)
    end
    return nothing
end


function _gve_kep!(dkep, kep, params, t, include_sh)
    mu = params.mus[1]
    rv = AstrodynamicsCore.kep2rv(kep, mu)
    a_pert = similar(rv, 3)
    a_rtn = similar(rv, 3)
    _perturbing_accel!(a_pert, rv, params, t, include_sh)
    project_inr_to_rtn!(a_rtn, a_pert, view(rv, 1:3), view(rv, 4:6))
    return gve_kep_derivs!(dkep, kep, a_rtn, mu)
end


"""Classical Keplerian GVE for the same force model as `eom_Nbody!`."""
gve_kep_Nbody!(dkep, kep, params::AbstractHFEMParameters, t) =
    _gve_kep!(dkep, kep, params, t, Val(false))

"""Out-of-place form of [`gve_kep_Nbody!`](@ref)."""
function gve_kep_Nbody(kep, params::AbstractHFEMParameters, t)
    dkep = similar(kep)
    gve_kep_Nbody!(dkep, kep, params, t)
    return dkep
end

"""Classical Keplerian GVE for the same force model as `eom_NbodySH!`."""
gve_kep_NbodySH!(dkep, kep, params::AbstractHFEMParameters, t) =
    _gve_kep!(dkep, kep, params, t, Val(true))

"""Out-of-place form of [`gve_kep_NbodySH!`](@ref)."""
function gve_kep_NbodySH(kep, params::AbstractHFEMParameters, t)
    dkep = similar(kep)
    gve_kep_NbodySH!(dkep, kep, params, t)
    return dkep
end


"""
    gve_eq_derivs!(deq, eq, a_rtn, mu)

Ordinary-equinoctial Gauss variational equations for
`eq = [a, f, g, h, k, lambda]`, where `lambda = Omega + omega + M` is mean
longitude. The acceleration order is `[radial, transverse, normal]`. This
elliptic formulation requires `a > 0` and `e < 1` and is singular exactly at
`i = pi` in the standard prograde inclination coordinates. All quantities must
use one consistent unit system and angles are radians.
"""
function gve_eq_derivs!(deq, eq, a_rtn, mu)
    length(deq) == 6 || throw(DimensionMismatch("deq must contain six components"))
    length(eq) == 6 || throw(DimensionMismatch("eq must contain six components"))
    length(a_rtn) == 3 || throw(DimensionMismatch("a_rtn must contain three components"))

    semimajor_axis, fmee, gmee, hmee, kmee, _ = eq
    semimajor_axis > zero(semimajor_axis) ||
        throw(DomainError(semimajor_axis, "equinoctial semimajor axis must be positive"))
    mu > zero(mu) || throw(DomainError(mu, "central-body gravitational parameter must be positive"))
    e2 = fmee * fmee + gmee * gmee
    beta2 = one(e2) - e2
    beta2 > zero(beta2) ||
        throw(DomainError(e2, "ordinary equinoctial GVE require eccentricity less than one"))

    mee = AstrodynamicsCore.eq2mee(eq)
    dmee = similar(mee)
    gve_mee_derivs!(dmee, mee, a_rtn, mu)

    pmee, longitude = mee[1], mee[6]
    pdot, fdot, gdot, hdot, kdot = dmee[1], dmee[2], dmee[3], dmee[4], dmee[5]
    a_radial = a_rtn[1]
    sinL, cosL = sincos(longitude)
    w = one(e2) + fmee * cosL + gmee * sinL
    radius = pmee / w
    beta = sqrt(beta2)
    alpha = inv(one(beta) + beta)
    s2 = one(e2) + hmee * hmee + kmee * kmee
    mean_motion = sqrt(mu / semimajor_axis^3)

    @inbounds begin
        deq[1] = (pdot + 2 * semimajor_axis * (fmee * fdot + gmee * gdot)) / beta2
        deq[2] = fdot
        deq[3] = gdot
        deq[4] = hdot
        deq[5] = kdot
        deq[6] = mean_motion - 2 * radius / (mean_motion * semimajor_axis^2) * a_radial +
            alpha * (fmee * gdot - gmee * fdot) +
            2 * beta / s2 * (hmee * kdot - kmee * hdot)
    end
    return nothing
end


function _gve_eq!(deq, eq, params, t, include_sh)
    mu = params.mus[1]
    rv = AstrodynamicsCore.eq2rv(eq, mu)
    a_pert = similar(rv, 3)
    a_rtn = similar(rv, 3)
    _perturbing_accel!(a_pert, rv, params, t, include_sh)
    project_inr_to_rtn!(a_rtn, a_pert, view(rv, 1:3), view(rv, 4:6))
    return gve_eq_derivs!(deq, eq, a_rtn, mu)
end


"""Ordinary-equinoctial GVE for the same force model as `eom_Nbody!`."""
gve_eq_Nbody!(deq, eq, params::AbstractHFEMParameters, t) =
    _gve_eq!(deq, eq, params, t, Val(false))

"""Out-of-place form of [`gve_eq_Nbody!`](@ref)."""
function gve_eq_Nbody(eq, params::AbstractHFEMParameters, t)
    deq = similar(eq)
    gve_eq_Nbody!(deq, eq, params, t)
    return deq
end

"""Ordinary-equinoctial GVE for the same force model as `eom_NbodySH!`."""
gve_eq_NbodySH!(deq, eq, params::AbstractHFEMParameters, t) =
    _gve_eq!(deq, eq, params, t, Val(true))

"""Out-of-place form of [`gve_eq_NbodySH!`](@ref)."""
function gve_eq_NbodySH(eq, params::AbstractHFEMParameters, t)
    deq = similar(eq)
    gve_eq_NbodySH!(deq, eq, params, t)
    return deq
end