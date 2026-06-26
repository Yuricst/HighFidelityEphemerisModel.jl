# Core routines

## Parameters & Interpolations

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:function, :type, :struct]
Pages   = [
  "parameters.jl",
  "ephemerides.jl",
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
  "eoms/eom_Nbody_Ephemerides.jl",
  "eoms/eom_NbodySH_Interp.jl",
  "eoms/eom_NbodySH_SPICE.jl",
  "eoms/eom_NbodySH_Ephemerides.jl",
]
```

## Jacobian, Hessian, and utility routines

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:function, :type]
Pages   = [
  "utils.jl",
]
```

## Perturbation routines

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:function, :type, :struct]
Pages   = [
  "perturbations/spherical_harmonics.jl",
  "perturbations/third_body.jl",
  "perturbations/solar_radiation_pressure.jl",
  "perturbations/drag.jl",
  "perturbations/harrispriester.jl",
  "perturbations/jacchiaroberts.jl",
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

## SPK generation helpers

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:type, :constant, :function]
Pages   = [
  "spk/states.jl",
  "spk/spkw13.jl",
  "spk/maneuvers.jl",
  "spk/metadata.jl",
  "spk/incremental.jl",
  "spk/ode_sol_to_spk.jl",
]
```
