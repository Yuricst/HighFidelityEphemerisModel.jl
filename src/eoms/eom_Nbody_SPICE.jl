"""Equations of motion"""


"""
    eom_Nbody_SPICE!(dx, x, params, t)

Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`
"""
function eom_Nbody_SPICE!(dx, x, params, t)
    return _hfem_eom!(dx, x, params, t, _SpiceBackend(), Val(false))
end


"""
    eom_Nbody_SPICE(x, params, t)

Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`
"""
function eom_Nbody_SPICE(x, params, t)
    return _hfem_eom(x, params, t, _SpiceBackend(), Val(false))
end


"""
    eom_stm_Nbody_SPICE!(dx_stm, x_stm, params, t)

Right-hand side of N-body equations of motion with STM compatible with `DifferentialEquations.jl`
"""
function eom_stm_Nbody_SPICE!(dx_stm, x_stm, params, t)
    return eom_stm_Nbody_SPICE_fd!(dx_stm, x_stm, params, t)
end


"""
    eom_stm_Nbody_SPICE_fd!(dx_stm, x_stm, params, t)
    
Right-hand side of N-body equations of motion with STM compatible with `DifferentialEquations.jl`
"""
function eom_stm_Nbody_SPICE_fd!(dx_stm, x_stm, params, t)
    return _hfem_eom_stm_fd!(dx_stm, x_stm, params, t, eom_Nbody_SPICE)
end
