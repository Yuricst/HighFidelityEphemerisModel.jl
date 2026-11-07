"""Equations of motion for N-body problem with spherical harmonics using interpolated structs"""


"""
    eom_NbodySH_Interp!(dx, x, params, t)
    
Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`
"""
function eom_NbodySH_Interp!(dx, x, params, t)
    return _hfem_eom!(dx, x, params, t, _InterpBackend(), Val(true))
end


"""
    eom_NbodySH_Interp(x, params, t)
    
Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`
"""
function eom_NbodySH_Interp(x, params, t)
    return _hfem_eom(x, params, t, _InterpBackend(), Val(true))
end


"""
    dfdx_NbodySH_Interp_fd(x, u, params, t)
    
Evaluate Jacobian of N-body problem
"""
function dfdx_NbodySH_Interp_fd(x, u, params, t)
    return ForwardDiff.jacobian(x -> HighFidelityEphemerisModel.eom_NbodySH_Interp(x, params, t), x)
end



"""
    eom_stm_NbodySH_Interp_fd!(dx_stm, x_stm, params, t)
    
Right-hand side of N-body equations of motion with STM compatible with `DifferentialEquations.jl`
"""
function eom_stm_NbodySH_Interp_fd!(dx_stm, x_stm, params, t)
    return _hfem_eom_stm_fd!(dx_stm, x_stm, params, t, eom_NbodySH_Interp)
end
