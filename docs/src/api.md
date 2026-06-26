# Core routines

## Parameters

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:type, :constant, :function]
Pages   = [
    "parameters.jl",
]
```

## Ephemerides and interpolation

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:type, :constant, :function]
Pages   = [
    "ephemerides.jl",
    "ephemeris_interpolation.jl",
    "transformation_interpolation.jl",
]
```

## Equations of motion

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:type, :constant, :function]
Pages   = [
    "eoms/eom_Nbody_SPICE.jl",
    "eoms/eom_Nbody_Interp.jl",
    "eoms/eom_Nbody_Ephemerides.jl",
    "eoms/eom_NbodySH_SPICE.jl",
    "eoms/eom_NbodySH_Interp.jl",
    "eoms/eom_NbodySH_Ephemerides.jl",
]
```

## Jacobian and Hessian routines

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:type, :constant, :function]
Pages   = [
    "utils.jl",
    "jacobians_symbolic.jl",
    "jacobians_sparsediff.jl",
]
```

## Perturbation routines

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:type, :constant, :function]
Pages   = [
    "perturbations/third_body.jl",
    "perturbations/spherical_harmonics.jl",
    "perturbations/solar_radiation_pressure.jl",
    "perturbations/drag.jl",
    "perturbations/harrispriester.jl",
    "perturbations/jacchiaroberts.jl",
]
```

## Events

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:type, :constant, :function]
Pages   = [
    "events.jl",
]
```

## SPK generation helpers

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:type, :constant, :function]
Pages   = [
    "spk/utils.jl",
    "spk/states.jl",
    "spk/spkw13.jl",
    "spk/maneuvers.jl",
    "spk/metadata.jl",
    "spk/incremental.jl",
    "spk/ode_sol_to_spk.jl",
]
```
