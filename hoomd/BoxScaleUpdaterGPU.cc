// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BoxScaleUpdaterGPU.cc
    \brief Defines the BoxScaleUpdaterGPU class
*/

#include "BoxScaleUpdaterGPU.h"
#include "BoxResizeUpdaterGPU.cuh"

namespace hoomd
    {
/*! \param sysdef System definition containing the particle data to set the box size on
    \param Lx length of the x dimension over time
    \param Ly length of the y dimension over time
    \param Lz length of the z dimension over time

    The default setting is to scale particle positions along with the box.
*/

BoxScaleUpdaterGPU::BoxScaleUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                                         std::shared_ptr<Trigger> trigger,
                                         std::shared_ptr<BoxDim> box,
                                         std::shared_ptr<Variant> variant_x,
                                         std::shared_ptr<Variant> variant_y,
                                         std::shared_ptr<Variant> variant_z,
                                         std::shared_ptr<Variant> variant_xy,
                                         std::shared_ptr<Variant> variant_xz,
                                         std::shared_ptr<Variant> variant_yz,
                                         std::shared_ptr<ParticleGroup> group)
    : BoxScaleUpdater(sysdef, trigger, box, variant_x, variant_y, variant_z, 
                        variant_xy, variant_xz, variant_yz, group)
    {
    // only one GPU is supported
    if (!m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error("Cannot initialize BoxScaleUpdaterGPU on a CPU device.");
        }

    m_tuner_scale.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                         m_exec_conf,
                                         "box_resize_scale"));
    m_tuner_wrap.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                        m_exec_conf,
                                        "box_resize_wrap"));
    }

BoxScaleUpdaterGPU::~BoxScaleUpdaterGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying BoxScaleUpdater" << std::endl;
    }

/// Scale particles to the new box and wrap any others back into the box
void BoxScaleUpdaterGPU::scaleAndWrapParticles(const BoxDim& cur_box, const BoxDim& new_box)
    {
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                               access_location::device,
                               access_mode::readwrite);

    ArrayHandle<int3> d_image(m_pdata->getImages(),
                              access_location::device,
                              access_mode::readwrite);

    unsigned int group_size = m_group->getNumMembers();
    ArrayHandle<unsigned int> d_group_members(m_group->getIndexArray(),
                                              access_location::device,
                                              access_mode::read);
    m_tuner_scale->begin();
    kernel::gpu_box_resize_scale(d_pos.data,
                                 cur_box,
                                 new_box,
                                 d_group_members.data,
                                 group_size,
                                 m_tuner_scale->getParam()[0]);
    m_tuner_scale->end();

    m_tuner_wrap->begin();
    kernel::gpu_box_resize_wrap(m_pdata->getN(),
                                d_pos.data,
                                d_image.data,
                                new_box,
                                m_tuner_wrap->getParam()[0]);
    m_tuner_wrap->end();
    }

namespace detail
    {
void export_BoxScaleUpdaterGPU(pybind11::module& m)
    {
    pybind11::class_<BoxScaleUpdaterGPU, BoxScaleUpdater, std::shared_ptr<BoxScaleUpdaterGPU>>(
        m,
        "BoxScaleUpdaterGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<BoxDim>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<ParticleGroup>>());
    }

    } // end namespace detail
    } // end namespace hoomd
