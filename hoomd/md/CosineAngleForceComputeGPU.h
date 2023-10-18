// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "SineSqAngleForceCompute.h"
#include "SineSqAngleForceGPU.cuh"
#include "hoomd/Autotuner.h"

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <memory>

/*! \file SineSqAngleForceComputeGPU.h
    \brief Declares the SineSqAngleForceGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __SINESQANGLEFORCECOMPUTEGPU_H__
#define __SINESQANGLEFORCECOMPUTEGPU_H__

namespace hoomd
    {
namespace md
    {
//! Implements the sine squared angle force calculation on the GPU
/*! SineSqAngleForceComputeGPU implements the same calculations as SineSqAngleForceCompute,
    but executing on the GPU.

    Per-type parameters are stored in a simple global memory area pointed to by
    \a m_gpu_params. They are stored as Scalar2's with the \a x component being a and the
    \a y component being b.

    The GPU kernel can be found in angleforce_kernel.cu.

    \ingroup computes
*/
class PYBIND11_EXPORT SineSqAngleForceComputeGPU : public SineSqAngleForceCompute
    {
    public:
    //! Constructs the compute
    SineSqAngleForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef);
    //! Destructor
    ~SineSqAngleForceComputeGPU();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar a, Scalar b);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size
    GPUArray<Scalar2> m_params;            //!< Parameters stored on the GPU

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
