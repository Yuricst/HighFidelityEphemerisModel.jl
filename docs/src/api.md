# Core routines

## Parameters & Interpolations

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:function, :type, :struct]
Pages   = [
  "parameters.jl",
  "ephemeris_interpolation.jl",
  "transformation_interpolation.jl",
]
```

## Equations of motion

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:function, :type]
Pages   = [
  "eoms/eom_Nbody_Interp.jl",
  "eoms/eom_Nbody_SPICE.jl",
  "eoms/eom_NbodySH_Interp.jl",
  "eoms/eom_NbodySH_SPICE.jl"
]
```

## Jacobians & Hessians
```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:function, :type]
Pages   = [
  "utils.jl"
]
```

## Perturbations

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:function, :type]
Pages   = [
  "perturbations/spherical_harmonics.jl",
  "perturbations/third_body.jl", 
  "perturbations/solar_radiation_pressure.jl"
]
```

## Events

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:function, :type]
Pages   = [
  "events.jl",
]
```