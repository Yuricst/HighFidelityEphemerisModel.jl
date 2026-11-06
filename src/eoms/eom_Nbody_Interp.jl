"""Interpolated ephemeris-based N-body equations of motion"""


"""
    eom_Nbody_Interp!(dx, x, params, t)

Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`
"""
function eom_Nbody_Interp!(dx, x, params, t)
    return _hfem_eom!(dx, x, params, t, _InterpBackend(), Val(false))
end


"""
    eom_Nbody_Interp(x, params, t)
    
Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`
"""
function eom_Nbody_Interp(x, params, t)
    return _hfem_eom(x, params, t, _InterpBackend(), Val(false))
end


"""
    eom_stm_Nbody_Interp!(dx_stm, x_stm, params, t)

Right-hand side of N-body equations of motion with STM compatible with `DifferentialEquations.jl`
"""
function eom_stm_Nbody_Interp!(dx_stm, x_stm, params, t)
    return eom_stm_Nbody_Interp_fd!(dx_stm, x_stm, params, t)
end


"""
    eom_stm_Nbody_Interp_fd!(dx_stm, x_stm, params, t)
    
Right-hand side of N-body equations of motion with STM compatible with `DifferentialEquations.jl`
"""
function eom_stm_Nbody_Interp_fd!(dx_stm, x_stm, params, t)
    return _hfem_eom_stm_fd!(dx_stm, x_stm, params, t, eom_Nbody_Interp)
end
