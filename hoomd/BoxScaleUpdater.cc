// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BoxScaleUpdater.cc
    \brief Defines the BoxScaleUpdater class
*/

#include "BoxScaleUpdater.h"

#include <iostream>
#include <math.h>
#include <stdexcept>

using namespace std;

namespace hoomd
    {
/*! \param sysdef System definition containing the particle data to set the box size on
    \param Lx length of the x dimension over time
    \param Ly length of the y dimension over time
    \param Lz length of the z dimension over time

    The default setting is to scale particle positions along with the box.
*/

BoxScaleUpdater::BoxScaleUpdater(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<Trigger> trigger,
                                   std::shared_ptr<BoxDim> box,
                                   std::shared_ptr<Variant> variant_x,
                                   std::shared_ptr<Variant> variant_y,
                                   std::shared_ptr<Variant> variant_z,
                                   std::shared_ptr<Variant> variant_xy,
                                   std::shared_ptr<Variant> variant_xz,
                                   std::shared_ptr<Variant> variant_yz,
                                   std::shared_ptr<ParticleGroup> group)
    : Updater(sysdef, trigger), m_box(box), m_variant_x(variant_x), m_variant_y(variant_y), 
      m_variant_z(variant_z), m_variant_xy(variant_xy), m_variant_xz(variant_xz), 
      m_variant_yz(variant_yz), m_group(group)
    {
    assert(m_pdata);
    assert(m_variant_x);
    assert(m_variant_y);
    assert(m_variant_z);
    assert(m_variant_xy);
    assert(m_variant_xz);
    assert(m_variant_yz);
    m_exec_conf->msg->notice(5) << "Constructing BoxScaleUpdater" << endl;
    }

BoxScaleUpdater::~BoxScaleUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying BoxScaleUpdater" << endl;
    }

/// Get box
std::shared_ptr<BoxDim> BoxScaleUpdater::getBox()
    {
    return m_box;
    }

/// Set a new box
void BoxScaleUpdater::setBox(std::shared_ptr<BoxDim> box)
    {
    m_box = box;
    }

/// Get the current box based on the timestep
BoxDim BoxScaleUpdater::getCurrentBox(uint64_t timestep)
    {
    Scalar init_value_x = (*m_variant_x)(0)
    Scalar init_value_y = (*m_variant_y)(0)
    Scalar init_value_z = (*m_variant_z)(0)
    Scalar init_value_xy = (*m_variant_xy)(0)
    Scalar init_value_xz = (*m_variant_xz)(0)
    Scalar init_value_yz = (*m_variant_yz)(0)
    Scalar cur_value_x = (*m_variant_x)(timestep)
    Scalar cur_value_y = (*m_variant_y)(timestep)
    Scalar cur_value_z = (*m_variant_z)(timestep)
    Scalar cur_value_xy = (*m_variant_xy)(timestep)
    Scalar cur_value_xz = (*m_variant_xz)(timestep)
    Scalar cur_value_yz = (*m_variant_yz)(timestep)
    
    Scalar3 scale_L = make_scalar3(1,1,1);
    Scalar scale_xy = 1;
    Scalar scale_xz = 1;
    Scalar scale_yz = 1;
    if (timestep > 0)
        {
        scale_L = make_scalar3(cur_value_x/init_value_x, 
                               cur_value_y/init_value_y, 
                               cur_value_z/init_value_z);
        scale_xy = cur_value_xy/init_value_xy;
        scale_xz = cur_value_xz/init_value_xz;
        scale_yz = cur_value_yz/init_value_yz;
        }
    
    const auto& box = *m_box;
    Scalar3 new_L = box.getL() * scale_L
    Scalar xy = box.getTiltFactorXY() * scale_xy;
    Scalar xz = box.getTiltFactorXZ() * scale_xz;
    Scalar yz = box.getTiltFactorYZ() * scale_yz;

    BoxDim new_box = BoxDim(new_L);
    new_box.setTiltFactors(xy, xz, yz);
    return new_box;
    }

/** Perform the needed calculations to scale the box size
    \param timestep Current time step of the simulation
*/
void BoxScaleUpdater::update(uint64_t timestep)
    {
    Updater::update(timestep);
    m_exec_conf->msg->notice(10) << "Box resize update" << endl;

    // first, compute the new box
    BoxDim new_box = getCurrentBox(timestep);

    // check if the current box size is the same
    BoxDim cur_box = m_pdata->getGlobalBox();

    // only change the box if there is a change in the box dimensions
    if (new_box != cur_box)
        {
        // set the new box
        m_pdata->setGlobalBox(new_box);

        // scale the particle positions (if we have been asked to)
        // move the particles to be inside the new box
        scaleAndWrapParticles(cur_box, new_box);

        // scale the origin
        Scalar3 old_origin = m_pdata->getOrigin();
        Scalar3 fractional_old_origin = cur_box.makeFraction(old_origin);
        Scalar3 new_origin = new_box.makeCoordinates(fractional_old_origin);
        m_pdata->translateOrigin(new_origin - old_origin);
        }
    }

/// Scale particles to the new box and wrap any others back into the box
void BoxScaleUpdater::scaleAndWrapParticles(const BoxDim& cur_box, const BoxDim& new_box)
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < m_group->getNumMembers(); group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        // obtain scaled coordinates in the old global box
        Scalar3 fractional_pos
            = cur_box.makeFraction(make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z));

        // intentionally scale both rigid body and free particles, this
        // may waste a few cycles but it enables the debug inBox checks
        // to be left as is (otherwise, setRV cannot fixup rigid body
        // positions without failing the check)
        Scalar3 scaled_pos = new_box.makeCoordinates(fractional_pos);
        h_pos.data[j].x = scaled_pos.x;
        h_pos.data[j].y = scaled_pos.y;
        h_pos.data[j].z = scaled_pos.z;
        }

    // ensure that the particles are still in their
    // local boxes by wrapping them if they are not
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    const BoxDim& local_box = m_pdata->getBox();

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        // need to update the image if we move particles from one side
        // of the box to the other
        local_box.wrap(h_pos.data[i], h_image.data[i]);
        }
    }

namespace detail
    {
void export_BoxScaleUpdater(pybind11::module& m)
    {
    pybind11::class_<BoxScaleUpdater, Updater, std::shared_ptr<BoxScaleUpdater>>(
        m,
        "BoxScaleUpdater")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<BoxDim>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<ParticleGroup>>())
        .def_property("box", &BoxScaleUpdater::getBox, &BoxScaleUpdater::setBox)
        .def_property("variant_x", &BoxScaleUpdater::getVariantX, &BoxScaleUpdater::setVariantX)
        .def_property("variant_y", &BoxScaleUpdater::getVariantY, &BoxScaleUpdater::setVariantY)
        .def_property("variant_z", &BoxScaleUpdater::getVariantZ, &BoxScaleUpdater::setVariantZ)
        .def_property("variant_xy", &BoxScaleUpdater::getVariantXY, &BoxScaleUpdater::setVariantXY)
        .def_property("variant_xz", &BoxScaleUpdater::getVariantXZ, &BoxScaleUpdater::setVariantXZ)
        .def_property("variant_yz", &BoxScaleUpdater::getVariantYZ, &BoxScaleUpdater::setVariantYZ)
        .def_property_readonly("filter",
                               [](const std::shared_ptr<BoxScaleUpdater> method)
                               { return method->getGroup()->getFilter(); })
        .def("get_current_box", &BoxScaleUpdater::getCurrentBox);
    }

    } // end namespace detail

    } // end namespace hoomd
