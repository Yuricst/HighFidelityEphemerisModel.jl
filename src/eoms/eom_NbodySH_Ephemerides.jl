"""Ephemerides.jl/FrameTransformations-based N-body equations of motion with spherical harmonics"""


"""
    eom_NbodySH_Ephemerides!(dx, x, params, t)

Right-hand side of N-body equations of motion with spherical harmonics compatible
with `DifferentialEquations.jl`.

Third-body positions and frame rotations are queried through the Ephemerides
backend attached to `params`.
"""
function eom_NbodySH_Ephemerides!(dx, x, params::EphemeridesParameters, t)
    return _hfem_eom!(dx, x, params, t, _EphemeridesBackend(), Val(true))
end


"""
    eom_NbodySH_Ephemerides(x, params, t)

Right-hand side of N-body equations of motion with spherical harmonics compatible
with `DifferentialEquations.jl`.
"""
function eom_NbodySH_Ephemerides(x, params::EphemeridesParameters, t)
    return _hfem_eom(x, params, t, _EphemeridesBackend(), Val(true))
end


"""
    eom_stm_NbodySH_Ephemerides_fd!(dx_stm, x_stm, params, t)

Right-hand side of N-body equations of motion with spherical harmonics and STM
compatible with `DifferentialEquations.jl`.
"""
function eom_stm_NbodySH_Ephemerides_fd!(dx_stm, x_stm, params::EphemeridesParameters, t)
    return _hfem_eom_stm_fd!(dx_stm, x_stm, params, t, eom_NbodySH_Ephemerides)
end
