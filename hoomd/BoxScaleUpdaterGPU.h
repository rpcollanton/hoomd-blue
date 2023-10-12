// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BoxScaleUpdater.h
    \brief Declares an updater that resizes the simulation box of the system
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "BoxScaleUpdater.h"

#ifndef __BOX_SCALE_UPDATER_GPU_H__
#define __BOX_SCALE_UPDATER_GPU_H__

namespace hoomd
    {
/// Updates the simulation box over time using the GPU
/** This simple updater gets the box lengths from specified variants and sets
 * those box sizes over time. As an option, particles can be rescaled with the
 * box lengths or left where they are. Note: rescaling particles does not work
 * properly in MPI simulations with HPMC.
 * \ingroup updaters
 */
class PYBIND11_EXPORT BoxScaleUpdaterGPU : public BoxScaleUpdater
    {
    public:
    /// Constructor
    BoxScaleUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<Trigger> trigger,
                       std::shared_ptr<BoxDim> box,
                       std::shared_ptr<Variant> variant_x,
                       std::shared_ptr<Variant> variant_y,
                       std::shared_ptr<Variant> variant_z,
                       std::shared_ptr<Variant> variant_xy,
                       std::shared_ptr<Variant> variant_xz,
                       std::shared_ptr<Variant> variant_yz,
                       std::shared_ptr<ParticleGroup> m_group);

    /// Destructor
    virtual ~BoxScaleUpdaterGPU();

    /// Scale particles to the new box and wrap any others back into the box
    virtual void scaleAndWrapParticles(const BoxDim& cur_box, const BoxDim& new_box);

    private:
    /// Autotuner for block size (scale kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_scale;
    /// Autotuner for block size (wrap kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_wrap;
    };

namespace detail
    {
/// Export the BoxScaleUpdaterGPU to python
void export_BoxScaleUpdaterGPU(pybind11::module& m);
    }  // end namespace detail
    }  // end namespace hoomd
#endif // __BOX_SCALE_UPDATER_GPU_H__
