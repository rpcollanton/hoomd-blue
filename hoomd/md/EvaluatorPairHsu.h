// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_HSU_H__
#define __PAIR_EVALUATOR_HSU_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairHsu.h
    \brief Defines the pair evaluator class for HSU potential
*/

// need to declare these class methods with __device__ qualifiers when building
// in nvcc DEVICE is __host__ __device__ when included in nvcc and blank when
// included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hoomd
    {
namespace md
    {

class EvaluatorPairHsu
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar a;
        Scalar r0;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        //! Set CUDA memory hints
        void set_memory_hint() const
            {
            // default implementation does nothing
            }
#endif

#ifndef __HIPCC__
        param_type() : a(0), r0(0) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            a = v["a"].cast<Scalar>();
            r0 = v["r0"].cast<Scalar>();
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["a"] = a;
            v["r0"] = r0;
            return v;
            }
#endif
        };

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairHsu(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), params(_params)
        {
        }

    //! HSU doesn't use charge
    DEVICE static bool needsCharge()
        {
        return false;
        }

    //! Accept the optional charge values.
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    DEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Evaluate the force and energy
    /*! \param force_divr Output parameter to write the computed force
     * divided by r.
     *  \param pair_eng Output parameter to write the computed pair energy
     *  \param energy_shift If true, the potential must be shifted so that
     *      V(r) is continuous at the cutoff

     *  \return True if they are evaluated or false if they are not because
     *  we are beyond the cutoff
     */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {
        if (rsq < rcutsq)
            {
            Scalar r0sq = fast::pow(params.r0,2);
            if (rsq < r0sq)
                {
                pair_eng = 0;
                force_divr = 0;
                }
            else
                {
                // Get quantities need for both energy and force calculation
                Scalar eval_sin, eval_cos;
                fast::sincospi(rsq/r0sq, eval_sin, eval_cos);

                // Compute energy
                pair_eng = params.a * eval_cos;

                // Compute force
                force_divr = 2*M_PI*params.a / r0sq * eval_sin;
                }
            return true;
            }
        else
            {
            return false;
            }
        }

    DEVICE Scalar evalPressureLRCIntegral()
        {
        return 0;
        }

    DEVICE Scalar evalEnergyLRCIntegral()
        {
        return 0;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("hsu");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;        /// Stored rsq from the constructor
    Scalar rcutsq;     /// Stored rcutsq from the constructor
    param_type params; /// Stored pair parameters for a given type pair
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_HSU_H__
