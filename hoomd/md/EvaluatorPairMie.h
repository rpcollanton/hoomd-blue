// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __PAIR_EVALUATOR_MIE_H__
#define __PAIR_EVALUATOR_MIE_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairMie.h
    \brief Defines the pair evaluator class for Mie potentials
    \details As the prototypical example of a MD pair potential, this also serves as the primary documentation and
    base reference for the implementation of pair evaluators.
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating the Mie pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ.

    <b>Mie specifics</b>

    EvaluatorPairMie evaluates the function:
    \f[ V_{\mathrm{mie}}(r)  = \left( \frac{n}{n-m} \right) {\left( \frac{n}{m} \right)}^{\frac{m}{n-m}} \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{n} -
                \left( \frac{\sigma}{r} \right)^{m} \right] \f]

    The Mie potential does not need diameter or charge. Four parameters are specified and stored in a Scalar4. \a mie1 \a mie2
    \a mie3 and \a mie4 are stored in \a params.x \a params.y \a params.z and \a params.w respectively.

    These are related to the standard lj parameters sigma and epsilon and the variable exponents n and m by:
    - \a mie1 = epsilon * pow(sigma,n) * (n/(n-m)) * power(n/m,m/(n-m))
    - \a mie2 = epsilon * pow(sigma,m) * (n/(n-m)) * power(n/m,m/(n-m))
    - \a mie3 = n
    - \a mie4 = m

*/
class EvaluatorPairMie
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        struct param_type
            {
            // params given by user
            Scalar epsilon, sigma, n, m;

            // params of interest for evaluating the potential
            Scalar m1, m2, m3, m4;

            #ifdef ENABLE_HIP
            // set CUDA memory hints
            void set_memory_hint() const {}
            #endif

            #ifndef __HIPCC__
            param_type()
                {
                n = m = epsilon = sigma = 0;
                m1 = m2 = m3 = m4 = 0;
                }

            param_type(Scalar eps, Scalar sig, Scalar n_, Scalar m_) :
                epsilon(eps), sigma(sig), n(n_), m(m_)
                {
                computeParams();
                }

            param_type(pybind11::dict v)
                {
                epsilon = v["epsilon"].cast<Scalar>();
                sigma = v["sigma"].cast<Scalar>();
                n = v["n"].cast<Scalar>();
                m = v["m"].cast<Scalar>();

                computeParams();
                }

            pybind11::dict asDict()
                {
                pybind11::dict v;
                v["epsilon"] = epsilon;
                v["sigma"] = sigma;
                v["n"] = n;
                v["m"] = m;

                return v;
                }

            private:
                // compute params relevant for evaluating potential from the
                // ones given by the user
                void computeParams()
                    {
                    m1 = epsilon * pow(sigma, n) * (n/(n-m)) * pow(n/m, m/(n-m));
                    m2 = epsilon * pow(sigma, m) * (n/(n-m)) * pow(n/m, m/(n-m));
                    m3 = n;
                    m4 = m;
                    }
            #endif
            }
            #ifdef SINGLE_PRECISION
            __attribute__((aligned(8)));
            #else
            __attribute__((aligned(16)));
            #endif

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance between the particles
            \param _rcutsq Squared distance at which the potential goes to 0
            \param _n First, larger exponent that captures hard-core repulsion
            \param -m Second, smaller exponent that captures attraction
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairMie(Scalar _rsq, Scalar _rcutsq,  const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), mie1(_params.m1), mie2(_params.m2), mie3(_params.m3), mie4(_params.m4)
            {
            }

        //! Mie doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! Mie doesn't use charge
        DEVICE static bool needsCharge() { return false; }
        //! Accept the optional diameter values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }

        //! Evaluate the force and energy
        /*! \param force_divr Output parameter to write the computed force divided by r.
            \param pair_eng Output parameter to write the computed pair energy
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff
            \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed
                  in PotentialPair.

            \return True if they are evaluated or false if they are not because we are beyond the cutoff
        */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
            {
            // compute the force divided by r in force_divr
            if (rsq < rcutsq && mie1 != 0)
                {
                Scalar r2inv = Scalar(1.0)/rsq;
                Scalar rninv = pow(r2inv,mie3/Scalar(2.0));
                Scalar rminv = pow(r2inv,mie4/Scalar(2.0));
                force_divr= r2inv * (mie3 * mie1 * rninv - mie4 * mie2 * rminv);

                pair_eng = mie1 * rninv - mie2 * rminv;

                if (energy_shift)
                    {
                    Scalar rcutninv = Scalar(1.0)/pow(rcutsq,mie3/Scalar(2.0));
                    Scalar rcutminv = Scalar(1.0)/pow(rcutsq,mie4/Scalar(2.0));
                    pair_eng -= mie1 * rcutninv - mie2* rcutminv;
                    }
                return true;
                }
            else
                return false;
            }

        #ifndef __HIPCC__
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("mie");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar mie1;     //!< mie1 parameter extracted from the params passed to the constructor
        Scalar mie2;     //!< mie2 parameter extracted from the params passed to the constructor
        Scalar mie3;     //!< mie3 parameter extracted from the params passed to the constructor
        Scalar mie4;     //!< mie4 parameter extracted from the params passed to the constructor
    };


#endif // __PAIR_EVALUATOR_MIE_H__
