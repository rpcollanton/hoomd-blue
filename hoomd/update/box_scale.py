# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement BoxScale."""

import hoomd
from hoomd.operation import Updater
from hoomd.box import Box
from hoomd.data.parameterdicts import ParameterDict
from hoomd.variant import Variant, Constant
from hoomd import _hoomd
from hoomd.filter import ParticleFilter, All
from hoomd.trigger import Periodic


class BoxScale(Updater):
    """Scales box parameters independently.

    `BoxScale` rescales the box parameters independently according to a set of
    variants corresponding with each box parameter. The variants are rescaled 
    such that their value at :math:`t=0` is unity, giving the starting box.

    .. math::

        \\begin{align*}
        L_{x}' &= \\lambda_x L_{x,0} \\\\
        L_{y}' &= \\lambda_y L_{y,0} \\\\
        L_{z}' &= \\lambda_z L_{z,0} \\\\
        xy' &= \\lambda_{xy} xy_{0} \\\\
        xz' &= \\lambda_{xz} xz_{0} \\\\
        yz' &= \\lambda_{yz} yz_{0} \\\\
        \\end{align*}

    Where `box` is :math:`(L_{x,0}, L_{y,0}, L_{z,0}, xy_0, xz_0, yz_0)`,
    :math:`\\lambda_i = \\frac{f_i(t)}{f_i(0)}`, :math:`t`
    is the timestep, and :math:`f_i(t)` is given by `variants[i]`, and 
    :math:`\\min f > 0`.

    For each particle :math:`i` matched by `filter`, `BoxScale` scales the
    particle to fit in the new box: 

    .. math::

        \\vec{r}_i \\leftarrow s_x \\vec{a}_1' + s_y \\vec{a}_2' +
                               s_z \\vec{a}_3' -
                    \\frac{\\vec{a}_1' + \\vec{a}_2' + \\vec{a}_3'}{2}

    where :math:`\\vec{a}_k'` are the new box vectors determined by
    :math:`(L_x', L_y', L_z', xy', xz', yz')` and the scale factors are
    determined by the current particle position :math:`\\vec{r}_i` and the old
    box vectors :math:`\\vec{a}_k`:

    .. math::

        \\vec{r}_i = s_x \\vec{a}_1 + s_y \\vec{a}_2 + s_z \\vec{a}_3 -
                    \\frac{\\vec{a}_1 + \\vec{a}_2 + \\vec{a}_3}{2}

    After scaling particles that match the filter, `BoxResize` wraps all
    particles :math:`j` back into the new box:

    .. math::

        \\vec{r_j} \\leftarrow \\mathrm{minimum\\_image}_{\\vec{a}_k}'
                               (\\vec{r}_j)

    Important:
        Each `Variant` in the passed list must be positive and non-zero.

    Warning:
        Rescaling particles fails in HPMC simulations with more than one MPI
        rank.

    Note:
        When using rigid bodies, ensure that the `BoxScale` updater is last in
        the operations updater list. Immediately after the `BoxScale` updater
        triggers, rigid bodies (`hoomd.md.constrain.Rigid`) will be temporarily
        deformed. `hoomd.md.Integrator` will run after the last updater and
        resets the constituent particle positions before computing forces.

    Args:
        trigger (hoomd.trigger.trigger_like): The trigger to activate this
            updater.
        box (hoomd.box.box_like): The initial simulation box.
        variants (List[hoomd.variant.Variant]): A list of variants used to 
            scale each box parameter.
        filter (hoomd.filter.filter_like): The subset of particle positions
            to update.

    Attributes:
        box1 (hoomd.Box): The initial simulation box.
        variants (List[hoomd.variant.Variant]): A list of variants used to 
            scale each box parameter.
        trigger (hoomd.trigger.Trigger): The trigger to activate this updater.
        filter (hoomd.filter.filter_like): The subset of particles to
            update.
    """

    def __init__(self, trigger, box, variants, filter=All()):
        params = ParameterDict(box=Box,
                               filter=ParticleFilter)
        
        # make variant list
        if isinstance(variants, Variant):
            variants = [variants]
        if len(variants)==1:
            variants = [variants[0], variants[0], variants[0], 
                        Constant(1), Constant(1), Constant(1)]
        elif len(variants)==3:
            for i in range(3):
                variants.append(Constant(1))
        elif len(variants!=6):
            raise ValueError("Can not determine appropriate list of variants.")
        
        params['box'] = box
        params['variants'] = variants
        params['trigger'] = trigger
        params['filter'] = filter
        self._param_dict.update(params)
        super().__init__(trigger)

    def _attach_hook(self):
        group = self._simulation.state._get_group(self.filter)
        if isinstance(self._simulation.device, hoomd.device.CPU):
            self._cpp_obj = _hoomd.BoxScaleUpdater(
                self._simulation.state._cpp_sys_def, self.trigger,
                self.box._cpp_obj, self.variants[0], self.variants[1],
                self.variants[2], self.variants[3], self.variants[4],
                self.variants[5], group)
        else:
            self._cpp_obj = _hoomd.BoxScaleUpdaterGPU(
                self._simulation.state._cpp_sys_def, self.trigger,
                self.box._cpp_obj, self.variants[0], self.variants[1],
                self.variants[2], self.variants[3], self.variants[4],
                self.variants[5], group)

    def get_box(self, timestep):
        """Get the box for a given timestep.

        Args:
            timestep (int): The timestep to use for determining the resized
                box.

        Returns:
            Box: The box used at the given timestep.
            `None` before the first call to `Simulation.run`.
        """
        if self._attached:
            timestep = int(timestep)
            if timestep < 0:
                raise ValueError("Timestep must be a non-negative integer.")
            return Box._from_cpp(self._cpp_obj.get_current_box(timestep))
        else:
            return None
