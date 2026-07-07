"""Generic EOM front-ends selected by parameter type."""


function eom_Nbody!(dx, x, params::SpiceParameters, t)
    return eom_Nbody_SPICE!(dx, x, params, t)
end


function eom_Nbody!(dx, x, params::InterpParameters, t)
    return eom_Nbody_Interp!(dx, x, params, t)
end


function eom_Nbody!(dx, x, params::EphemeridesParameters, t)
    return eom_Nbody_Ephemerides!(dx, x, params, t)
end


function eom_Nbody(x, params::SpiceParameters, t)
    return eom_Nbody_SPICE(x, params, t)
end


function eom_Nbody(x, params::InterpParameters, t)
    return eom_Nbody_Interp(x, params, t)
end


function eom_Nbody(x, params::EphemeridesParameters, t)
    return eom_Nbody_Ephemerides(x, params, t)
end


function eom_NbodySH!(dx, x, params::SpiceParameters, t)
    return eom_NbodySH_SPICE!(dx, x, params, t)
end


function eom_NbodySH!(dx, x, params::InterpParameters, t)
    return eom_NbodySH_Interp!(dx, x, params, t)
end


function eom_NbodySH!(dx, x, params::EphemeridesParameters, t)
    return eom_NbodySH_Ephemerides!(dx, x, params, t)
end


function eom_NbodySH(x, params::SpiceParameters, t)
    return eom_NbodySH_SPICE(x, params, t)
end


function eom_NbodySH(x, params::InterpParameters, t)
    return eom_NbodySH_Interp(x, params, t)
end


function eom_NbodySH(x, params::EphemeridesParameters, t)
    return eom_NbodySH_Ephemerides(x, params, t)
end