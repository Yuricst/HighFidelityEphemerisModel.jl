module HighFidelityEphemerisModel

using Ephemerides
using FrameTransformations
using Dierckx
using ForwardDiff
using Interpolations
using LinearAlgebra
using Printf
using SPICE

include("utils.jl")
include("perturbations/third_body.jl")
include("perturbations/spherical_harmonics.jl")
include("perturbations/solar_radiation_pressure.jl")
include("perturbations/drag.jl")
include("perturbations/harrispriester.jl")
include("perturbations/jacchiaroberts.jl")

include("ephemeris_interpolation.jl")
include("transformation_interpolation.jl")
include("ephemerides.jl")
include("parameters.jl")

include("eoms/eom_Nbody_SPICE.jl")
include("eoms/eom_Nbody_Interp.jl")
include("eoms/eom_Nbody_Ephemerides.jl")

include("eoms/eom_NbodySH_SPICE.jl")
include("eoms/eom_NbodySH_Interp.jl")
include("eoms/eom_NbodySH_Ephemerides.jl")

include("events.jl")
include("spk/utils.jl")
include("spk/states.jl")
include("spk/spkw13.jl")
include("spk/maneuvers.jl")
include("spk/metadata.jl")
include("spk/incremental.jl")
include("spk/ode_sol_to_spk.jl")

export eom_jacobian_fd, eom_hessian_fd, et_to_utc_mjd

export InterpolatedEphemeris
export InterpolatedTransformation
export get_pos_ephemerides, get_state_ephemerides, pxform_ephemerides
export HighFidelityEphemerisModelParameters

export eom_Nbody_SPICE!, eom_Nbody_SPICE, eom_stm_Nbody_SPICE!, eom_stm_Nbody_SPICE_fd!
export eom_Nbody_Interp!, eom_Nbody_Interp, dfdx_Nbody_Interp, eom_stm_Nbody_Interp!, eom_stm_Nbody_Interp_fd!
export eom_Nbody_Ephemerides!, eom_Nbody_Ephemerides, dfdx_Nbody_Ephemerides, eom_stm_Nbody_Ephemerides!, eom_stm_Nbody_Ephemerides_fd!

export eom_NbodySH_SPICE!, eom_NbodySH_SPICE, eom_stm_NbodySH_SPICE_fd!
export eom_NbodySH_Interp!, eom_NbodySH_Interp, eom_stm_NbodySH_Interp_fd!
export eom_NbodySH_Ephemerides!, eom_NbodySH_Ephemerides, dfdx_NbodySH_Ephemerides_fd, eom_stm_NbodySH_Ephemerides_fd!

export get_trueanomaly_event
export HarrisPriesterModel, harris_priester_f_density
export JacchiaRobertsModel, jacchia_roberts_f_density, JacchiaRobertsGeomagneticExposphericParams

export ode_sol_to_spk
export prepare_spk_output!
export append_solution_segment_to_spk!
export append_state_file_to_spk!
export write_spk_metadata_json

end # module HighFidelityEphemerisModel