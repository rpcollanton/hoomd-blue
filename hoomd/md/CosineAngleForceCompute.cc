// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "CosineAngleForceCompute.h"

#include <iostream>
#include <math.h>
#include <sstream>
#include <stdexcept>

using namespace std;

// SMALL a relatively small number
#define SMALL Scalar(1E-5)

/*! \file CosineAngleForceCompute.cc
    \brief Contains code for the CosineAngleForceCompute class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
CosineAngleForceCompute::CosineAngleForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef), m_k(NULL), m_t_0(NULL)
    {
    m_exec_conf->msg->notice(5) << "Constructing CosineAngleForceCompute" << endl;

    // access the angle data for later use
    m_angle_data = m_sysdef->getAngleData();

    // check for some silly errors a user could make
    if (m_angle_data->getNTypes() == 0)
        {
        throw runtime_error("No angle types in system.");
        }

    // allocate the parameters -- same as for harmonic
    m_k = new Scalar[m_angle_data->getNTypes()];
    m_t_0 = new Scalar[m_angle_data->getNTypes()];
    }

CosineAngleForceCompute::~CosineAngleForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying CosineAngleForceCompute" << endl;

    delete[] m_k;
    delete[] m_t_0;
    m_k = NULL;
    m_t_0 = NULL;
    }

/*! \param type Type of the angle to set parameters for
    \param k
    \param t_0 

    Sets parameters for the potential of a particular angle type
*/
void CosineAngleForceCompute::setParams(unsigned int type, Scalar k, Scalar t_0)
    {
    // make sure the type is valid
    if (type >= m_angle_data->getNTypes())
        {
        throw runtime_error("Invalid angle type.");
        }

    m_k[type] = k;
    m_t_0[type] = t_0;

    }

void CosineAngleForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    auto typ = m_angle_data->getTypeByName(type);
    auto _params = cosine_params(params);
    setParams(typ, _params.k, _params.t_0);
    }

pybind11::dict CosineAngleForceCompute::getParams(std::string type)
    {
    auto typ = m_angle_data->getTypeByName(type);
    if (typ >= m_angle_data->getNTypes())
        {
        throw runtime_error("Invalid angle type.");
        }

    pybind11::dict params;
    params["k"] = m_k[typ];
    params["t_0"] = m_t_0[typ];
    return params;
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void CosineAngleForceCompute::computeForces(uint64_t timestep)
    {
    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    size_t virial_pitch = m_virial.getPitch();

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);

    // Zero data for force calculation.
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getGlobalBox();

    // for each of the angles
    const unsigned int size = (unsigned int)m_angle_data->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the angle
        const AngleData::members_t& angle = m_angle_data->getMembersByIndex(i);
        assert(angle.tag[0] <= m_pdata->getMaximumTag());
        assert(angle.tag[1] <= m_pdata->getMaximumTag());
        assert(angle.tag[2] <= m_pdata->getMaximumTag());

        // transform a, b, and c into indices into the particle data arrays
        // MEM TRANSFER: 6 ints
        unsigned int idx_a = h_rtag.data[angle.tag[0]];
        unsigned int idx_b = h_rtag.data[angle.tag[1]];
        unsigned int idx_c = h_rtag.data[angle.tag[2]];

        // throw an error if this angle is incomplete
        if (idx_a == NOT_LOCAL || idx_b == NOT_LOCAL || idx_c == NOT_LOCAL)
            {
            this->m_exec_conf->msg->error()
                << "angle.cosine: angle " << angle.tag[0] << " " << angle.tag[1] << " "
                << angle.tag[2] << " incomplete." << endl
                << endl;
            throw std::runtime_error("Error in angle calculation");
            }

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());

        // calculate d\vec{r}
        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

        Scalar3 dcb;
        dcb.x = h_pos.data[idx_c].x - h_pos.data[idx_b].x;
        dcb.y = h_pos.data[idx_c].y - h_pos.data[idx_b].y;
        dcb.z = h_pos.data[idx_c].z - h_pos.data[idx_b].z;

        Scalar3 dac;
        dac.x = h_pos.data[idx_a].x - h_pos.data[idx_c].x; // used for the 1-3 JL interaction
        dac.y = h_pos.data[idx_a].y - h_pos.data[idx_c].y;
        dac.z = h_pos.data[idx_a].z - h_pos.data[idx_c].z;

        // apply minimum image conventions to all 3 vectors
        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        dac = box.minImage(dac);

        // this is where cosine differs from harmonic
        // FLOPS: 14 / MEM TRANSFER: 2 Scalars

        // FLOPS: 42 / MEM TRANSFER: 6 Scalars
        Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z; // squared magnitude of r_ab
        Scalar rab = sqrt(rsqab);                                     // magnitude of r_ab
        Scalar rsqcb = dcb.x * dcb.x + dcb.y * dcb.y + dcb.z * dcb.z; // squared magnitude of r_cb
        Scalar rcb = sqrt(rsqcb);                                     // magnitude of r_cb

        Scalar cos_abbc = dab.x * dcb.x + dab.y * dcb.y + dab.z * dcb.z; // = ab dot bc
        cos_abbc /= rab * rcb;                                           // cos(t)
        Scalar sin_abbc = fast::sqrt(1.0 - cos_abbc * cos_abbc);

        // calculate the force
        unsigned int angle_type = m_angle_data->getTypeByIndex(i);
        
        // get sine and cosine of preferred angle
        Scalar sin_t0, cos_t0;
        fast::sincos(m_t_0[angle_type], sin_t0, cos_t0);

        // get gradients wrt bond vectors
        Scalar r1r2inv = 1/(rab*rcb);
        Scalar3 dcosdrab = dcb*r1r2inv - dab/rsqab * cos_abbc;
        Scalar3 dcosdrcb = dab*r1r2inv - dcb/rsqcb * cos_abbc;

        // check sine of preferred angle. if preferred angle 0 or pi, avoid approximation:
        Scalar dudcos;
        if (sin_t0 < 1E-6)
            {
            dudcos = -m_k[angle_type] * cos_t0;
            }
        else
            {
            if (sin_abbc < SMALL)
                sin_abbc = SMALL;
            dudcos = -m_k[angle_type]*(cos_t0 - sin_t0 * cos_abbc/sin_abbc);
            }

        Scalar fab[3], fcb[3];

        fab[0] = -dudcos * dcosdrab.x;
        fab[1] = -dudcos * dcosdrab.y;
        fab[2] = -dudcos * dcosdrab.z;

        fcb[0] = -dudcos * dcosdrcb.x;
        fcb[1] = -dudcos * dcosdrcb.y;
        fcb[2] = -dudcos * dcosdrcb.z;

        // the rest of the computation should stay the same
        // compute 1/3 of the energy, 1/3 for each atom in the angle
        Scalar angle_eng = Scalar(1. / 3.) * m_k[angle_type] * (1 - cos_abbc*cos_t0 - sin_abbc*sin_t0);

        // compute 1/3 of the virial, 1/3 for each atom in the angle
        // upper triangular version of virial tensor
        Scalar angle_virial[6];
        angle_virial[0] = Scalar(1. / 3.) * (dab.x * fab[0] + dcb.x * fcb[0]);
        angle_virial[1] = Scalar(1. / 3.) * (dab.y * fab[0] + dcb.y * fcb[0]);
        angle_virial[2] = Scalar(1. / 3.) * (dab.z * fab[0] + dcb.z * fcb[0]);
        angle_virial[3] = Scalar(1. / 3.) * (dab.y * fab[1] + dcb.y * fcb[1]);
        angle_virial[4] = Scalar(1. / 3.) * (dab.z * fab[1] + dcb.z * fcb[1]);
        angle_virial[5] = Scalar(1. / 3.) * (dab.z * fab[2] + dcb.z * fcb[2]);

        // Now, apply the force to each individual atom a,b,c, and accumulate the energy/virial
        // do not update ghost particles
        if (idx_a < m_pdata->getN())
            {
            h_force.data[idx_a].x += fab[0];
            h_force.data[idx_a].y += fab[1];
            h_force.data[idx_a].z += fab[2];
            h_force.data[idx_a].w += angle_eng;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_a] += angle_virial[j];
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x -= fab[0] + fcb[0];
            h_force.data[idx_b].y -= fab[1] + fcb[1];
            h_force.data[idx_b].z -= fab[2] + fcb[2];
            h_force.data[idx_b].w += angle_eng;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_b] += angle_virial[j];
            }

        if (idx_c < m_pdata->getN())
            {
            h_force.data[idx_c].x += fcb[0];
            h_force.data[idx_c].y += fcb[1];
            h_force.data[idx_c].z += fcb[2];
            h_force.data[idx_c].w += angle_eng;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_c] += angle_virial[j];
            }
        }
    }

namespace detail
    {
void export_CosineAngleForceCompute(pybind11::module& m)
    {
    pybind11::class_<CosineAngleForceCompute,
                     ForceCompute,
                     std::shared_ptr<CosineAngleForceCompute>>(m, "CosineAngleForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("getParams", &CosineAngleForceCompute::getParams)
        .def("setParams", &CosineAngleForceCompute::setParamsPython);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
