"""Equations of motion for N-body problem with spherical harmonics using SPICE"""


"""
    eom_NbodySH_SPICE!(dx, x, params, t)
    
Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`
"""
function eom_NbodySH_SPICE!(dx, x, params, t)
    return _hfem_eom!(dx, x, params, t, _SpiceBackend(), Val(true))
end


"""
    eom_NbodySH_SPICE(x, params, t)
    
Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`
"""
function eom_NbodySH_SPICE(x, params, t)
    return _hfem_eom(x, params, t, _SpiceBackend(), Val(true))
end


"""
    eom_stm_NbodySH_SPICE_fd!(dx_stm, x_stm, params, t)
    
Right-hand side of N-body equations of motion with STM compatible with `DifferentialEquations.jl`
"""
function eom_stm_NbodySH_SPICE_fd!(dx_stm, x_stm, params, t)
    return _hfem_eom_stm_fd!(dx_stm, x_stm, params, t, eom_NbodySH_SPICE)
end
