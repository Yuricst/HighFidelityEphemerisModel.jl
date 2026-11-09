"""Ephemerides.jl/FrameTransformations-based N-body equations of motion"""


"""
    eom_Nbody_Ephemerides!(dx, x, params, t)

Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`.

Third-body positions are queried directly through `FrameTransformations.vector3`
using the Ephemerides-backed frame system in `params.ephemerides_frame_system`.
HFEM uses the Ephemerides backend attached to `params`.
"""
function eom_Nbody_Ephemerides!(dx, x, params::EphemeridesParameters, t)
    return _hfem_eom!(dx, x, params, t, _EphemeridesBackend(), Val(false))
end


"""
    eom_Nbody_Ephemerides(x, params, t)

Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`.
"""
function eom_Nbody_Ephemerides(x, params::EphemeridesParameters, t)
    return _hfem_eom(x, params, t, _EphemeridesBackend(), Val(false))
end


"""
    eom_stm_Nbody_Ephemerides!(dx_stm, x_stm, params, t)

Right-hand side of N-body equations of motion with STM compatible with `DifferentialEquations.jl`.
"""
function eom_stm_Nbody_Ephemerides!(dx_stm, x_stm, params::EphemeridesParameters, t)
    return eom_stm_Nbody_Ephemerides_fd!(dx_stm, x_stm, params, t)
end


"""
    eom_stm_Nbody_Ephemerides_fd!(dx_stm, x_stm, params, t)

Right-hand side of N-body equations of motion with STM compatible with `DifferentialEquations.jl`.
"""
function eom_stm_Nbody_Ephemerides_fd!(dx_stm, x_stm, params::EphemeridesParameters, t)
    return _hfem_eom_stm_fd!(dx_stm, x_stm, params, t, eom_Nbody_Ephemerides)
end
