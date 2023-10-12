// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BoxScaleUpdater.h
    \brief Declares an updater that resizes the simulation box of the system
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "BoxDim.h"
#include "ParticleGroup.h"
#include "Updater.h"
#include "Variant.h"

#include <memory>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>

#ifndef __BOXSCALEUPDATER_H__
#define __BOXSCALEUPDATER_H__

namespace hoomd
    {
/// Updates the simulation box over time
/** This simple updater gets the box lengths from specified variants and sets
 * those box sizes over time. As an option, particles can be rescaled with the
 * box lengths or left where they are. Note: rescaling particles does not work
 * properly in MPI simulations.
 * \ingroup updaters
 */
class PYBIND11_EXPORT BoxScaleUpdater : public Updater
    {
    public:
    /// Constructor
    BoxScaleUpdater(std::shared_ptr<SystemDefinition> sysdef,
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
    virtual ~BoxScaleUpdater();

    /// Get the current m_box
    std::shared_ptr<BoxDim> getBox();

    /// Set a new m_box
    void setBox(std::shared_ptr<BoxDim> box);

    /// Gets particle scaling filter
    std::shared_ptr<ParticleGroup> getGroup()
        {
        return m_group;
        }

    /// Set the variant for interpolation
    void setVariantX(std::shared_ptr<Variant> variant)  { m_variant_x = variant; }
    void setVariantY(std::shared_ptr<Variant> variant)  { m_variant_y = variant; }
    void setVariantZ(std::shared_ptr<Variant> variant)  { m_variant_z = variant; }
    void setVariantXY(std::shared_ptr<Variant> variant) { m_variant_xy = variant; }
    void setVariantXZ(std::shared_ptr<Variant> variant) { m_variant_xz = variant; }
    void setVariantYZ(std::shared_ptr<Variant> variant) { m_variant_yz = variant; }

    /// Get the variant for interpolation
    std::shared_ptr<Variant> getVariantX() { return m_variant_x; }
    std::shared_ptr<Variant> getVariantY() { return m_variant_y; }
    std::shared_ptr<Variant> getVariantZ() { return m_variant_z; }
    std::shared_ptr<Variant> getVariantXY() { return m_variant_xy; }
    std::shared_ptr<Variant> getVariantXZ() { return m_variant_xz; }
    std::shared_ptr<Variant> getVariantYZ() { return m_variant_yz; }

    /// Get the current box for the given timestep
    BoxDim getCurrentBox(uint64_t timestep);

    /// Update box interpolation based on provided timestep
    virtual void update(uint64_t timestep);

    /// Scale particles to the new box and wrap any back into the box
    virtual void scaleAndWrapParticles(const BoxDim& cur_box, const BoxDim& new_box);

    protected:
    std::shared_ptr<BoxDim> m_box;         ///< C++ box assoc with min
    std::shared_ptr<Variant> m_variant_x;     //!< Variant that scales box in x direction
    std::shared_ptr<Variant> m_variant_y;     //!< Variant that scales box in y direction
    std::shared_ptr<Variant> m_variant_z;     //!< Variant that scales box in z direction
    std::shared_ptr<Variant> m_variant_xy;     //!< Variant that scales box xy tilt 
    std::shared_ptr<Variant> m_variant_xz;     //!< Variant that scales box xz tilt 
    std::shared_ptr<Variant> m_variant_yz;     //!< Variant that scales box yz tilt 
    std::shared_ptr<ParticleGroup> m_group; //!< Selected particles to scale when resizing the box.
    };

namespace detail
    {
/// Export the BoxScaleUpdater to python
void export_BoxScaleUpdater(pybind11::module& m);
    } // end namespace detail
    } // end namespace hoomd
#endif
