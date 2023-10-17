// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.h"
#include "hoomd/ForceCompute.h"

#include <memory>
#include <vector>

/*! \file SineSqAngleForceCompute.h
    \brief Declares a class for computing sine squared angles
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __SINESQANGLEFORCECOMPUTE_H__
#define __SINESQANGLEFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
struct sinesq_params
    {
    Scalar a;
    Scalar b;

#ifndef __HIPCC__
    sinesq_params() : a(0), b(0) { }

    sinesq_params(pybind11::dict params)
        : a(params["a"].cast<Scalar>()), b(params["b"].cast<Scalar>())
        {
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["a"] = a;
        v["b"] = b;
        return v;
        }
#endif
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(8)));
#else
    __attribute__((aligned(16)));
#endif

//! Computes sine squared angle forces on each particle
/*! Sine squared angle forces are computed on every particle in the simulation.

    The angles which forces are computed on are accessed from ParticleData::getAngleData
    \ingroup computes
*/
class PYBIND11_EXPORT SineSqAngleForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    SineSqAngleForceCompute(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~SineSqAngleForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar a, Scalar b);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a given type
    virtual pybind11::dict getParams(std::string type);

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    /*! \param timestep Current time step
     */
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    protected:
    Scalar* m_a;   //!< a parameter
    Scalar* m_b;   //!< b parameter

    std::shared_ptr<AngleData> m_angle_data; //!< Angle data to use in computing angles

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
