// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

/*! \file CosineAngleForceGPU.cuh
    \brief Declares GPU kernel code for calculating the sine squared angle forces. Used
    by CosineAngleForceComputeGPU.
*/

#ifndef __COSINEANGLEFORCEGPU_CUH__
#define __COSINEANGLEFORCEGPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes sine squared angle forces for CosineAngleForceComputeGPU
hipError_t gpu_compute_cosine_angle_forces(Scalar4* d_force,
                                             Scalar* d_virial,
                                             const size_t virial_pitch,
                                             const unsigned int N,
                                             const Scalar4* d_pos,
                                             const BoxDim& box,
                                             const group_storage<3>* atable,
                                             const unsigned int* apos_list,
                                             const unsigned int pitch,
                                             const unsigned int* n_angles_list,
                                             Scalar2* d_params,
                                             unsigned int n_angle_types,
                                             int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif