# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Classes for sampling grasps.
Author: Jeff Mahler
"""

from abc import ABCMeta, abstractmethod
import copy
import IPython
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import random
import sys
import time

USE_OPENRAVE = True
try:
    import openravepy as rave
except:
    logging.warning('Failed to import OpenRAVE')
    USE_OPENRAVE = False

import scipy.stats as stats

from dexnet.grasping import Contact3D, ParallelJawPtGrasp3D, PointGraspMetrics3D, GraspableObject3D


class GraspSampler:
    """ Base class for various methods to sample a number of grasps on an object.
    Should not be instantiated directly.

    Attributes
    ----------
    gripper : :obj:`RobotGripper`
        the gripper to compute grasps for
    config : :obj:`YamlConfig`
        configuration for the grasp sampler
    """
    __metaclass__ = ABCMeta

    def __init__(self, gripper, config):
        self.gripper = gripper
        self._configure(config)

    def _configure(self, config):
        """ Configures the grasp generator."""
        self.friction_coef = config['sampling_friction_coef']
        self.num_cone_faces = config['num_cone_faces']
        self.num_samples = config['grasp_samples_per_surface_point']
        self.target_num_grasps = config['target_num_grasps']
        if self.target_num_grasps is None:
            self.target_num_grasps = config['min_num_grasps']

        self.min_contact_dist = config['min_contact_dist']
        self.num_grasp_rots = config['num_grasp_rots']
        if 'max_num_surface_points' in config.keys():
            self.max_num_surface_points_ = config['max_num_surface_points']
        else:
            self.max_num_surface_points_ = 100
        if 'grasp_dist_thresh' in config.keys():
            self.grasp_dist_thresh_ = config['grasp_dist_thresh']
        else:
            self.grasp_dist_thresh_ = 0

    @abstractmethod
    def sample_grasps(self, graspable, num_grasp_generate=100, vis=False, quality_fn=None):
        """
        Create a list of candidate grasps for a given object.
        Must be implemented for all grasp sampler classes.

        Parameters
        ---------
        graspable : :obj:`GraspableObject3D`
            object to sample grasps on
        num_grasp_generate: :int:
            number of grasps to generate
        vis: :bool:
            visualising the process (for debugging purposes)
        quality_fn: :function:
            function to calculate grasp qualities with.
        """
        pass

    def generate_grasps_stable_poses(self, graspable, stable_poses, target_num_grasps=None, grasp_gen_mult=5, max_iter=3,
                        sample_approach_angles=False, vis=False, **kwargs):
        """Samples a set of grasps for an object, aligning the approach angles to the object stable poses.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        stable_poses : :obj:`list` of :obj:`meshpy.StablePose`
            list of stable poses for the object with ids read from the database
        target_num_grasps : int
            number of grasps to return, defualts to self.target_num_grasps
        grasp_gen_mult : int
            number of additional grasps to generate
        max_iter : int
            number of attempts to return an exact number of grasps before giving up
        sample_approach_angles : bool
            whether or not to sample approach angles

        Return
        ------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            list of generated grasps
        """
        # sample dense grasps
        unaligned_grasps = self.generate_grasps(graspable, target_num_grasps=target_num_grasps,
                                                grasp_gen_mult=grasp_gen_mult,
                                                max_iter=max_iter, vis=vis)
        
        # align for each stable pose
        grasps = {}
        for stable_pose in stable_poses:
            grasps[stable_pose.id] = []
            for grasp in unaligned_grasps:
                aligned_grasp = grasp.perpendicular_table(grasp)
                grasps[stable_pose.id].append(copy.deepcopy(aligned_grasp))
        return grasps
        
    def generate_grasps(self, graspable, target_num_grasps=None, grasp_gen_mult=5, max_iter=3,
                        sample_approach_angles=False, vis=False, prune_grasps=False,
                        quality_fn=None, truncating=False, **kwargs):
        """Samples a set of grasps for an object.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        target_num_grasps : int
            number of grasps to return, defualts to self.target_num_grasps
        grasp_gen_mult : int
            number of additional grasps to generate
        quality_fn : function
            If given, it will be used to select the best grasp for a given contact
        max_iter : int
            number of attempts to return an exact number of grasps before giving up

        Return
        ------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            list of generated grasps
        """
        # get num grasps 
        if target_num_grasps is None:
            target_num_grasps = self.target_num_grasps
        num_grasps_remaining = target_num_grasps

        grasps = []
        qualities = []
        k = 1
        while num_grasps_remaining > 0 and k <= max_iter:
            # SAMPLING: generate more than we need
            num_grasps_generate = grasp_gen_mult * num_grasps_remaining
            new_grasps, new_qualities = self.sample_grasps(graspable, num_grasps_generate,
                                               vis, quality_fn=quality_fn, **kwargs)

            # COVERAGE REJECTION: prune grasps by distance
            if prune_grasps:
                pruned_grasps = []
                pruned_qualities = []
                for cnt, grasp in enumerate(new_grasps):
                    min_dist = np.inf
                    for cur_grasp in grasps:
                        dist = ParallelJawPtGrasp3D.distance(cur_grasp, grasp)
                        if dist < min_dist:
                            min_dist = dist
                    for cur_grasp in pruned_grasps:
                        dist = ParallelJawPtGrasp3D.distance(cur_grasp, grasp)
                        if dist < min_dist:
                            min_dist = dist
                    if min_dist >= self.grasp_dist_thresh_:
                        pruned_grasps.append(grasp)
                        if new_qualities:
                            pruned_qualities.append(new_qualities[cnt])
                candidate_grasps = pruned_grasps
                candidate_qualities = pruned_qualities
            else:
                candidate_grasps = new_grasps
                candidate_qualities = new_qualities

            # add to the current grasp set
            grasps += candidate_grasps
            qualities += candidate_qualities
            logging.info('%d/%d grasps found after iteration %d.',
                         len(grasps), target_num_grasps, k)

            grasp_gen_mult *= 2
            num_grasps_remaining = target_num_grasps - len(grasps)
            k += 1

        # shuffle computed grasps (and qualities)
        if qualities:
            temp = list(zip(grasps, qualities))
            random.shuffle(temp)
            grasps, qualities = zip(*temp)
        else:
            random.shuffle(grasps)
        if truncating and len(grasps) > target_num_grasps:
            logging.info('Truncating %d grasps to %d.',
                         len(grasps), target_num_grasps)
            grasps = grasps[:target_num_grasps]
            if qualities:
                qualities = qualities[:target_num_grasps]
        logging.info('Found %d grasps.', len(grasps))

        return grasps, qualities


class AntipodalGraspSampler(GraspSampler):
    """ Samples antipodal pairs using rejection sampling.
    The proposal sampling ditribution is to choose a random point on
    the object surface, then sample random directions within the friction cone, then form a grasp axis along the direction,
    close the fingers, and keep the grasp if the other contact point is also in the friction cone.
    """
    def sample_from_cone(self, n, tx, ty, num_samples=1):
        """ Samples directoins from within the friction cone using uniform sampling.
        
        Parameters
        ----------
        n : 3x1 normalized :obj:`numpy.ndarray`
            surface normal
        tx : 3x1 normalized :obj:`numpy.ndarray`
            tangent x vector
        ty : 3x1 normalized :obj:`numpy.ndarray`
            tangent y vector
        num_samples : int
            number of directions to sample

        Returns
        -------
        v_samples : :obj:`list` of 3x1 :obj:`numpy.ndarray`
            sampled directions in the friction cone
       """
        v_samples = []
        for i in range(num_samples):
            theta = 2 * np.pi * np.random.rand()
            r = self.friction_coef * np.random.rand()
            v = n + r * np.cos(theta) * tx + r * np.sin(theta) * ty
            v = -v / np.linalg.norm(v)
            v_samples.append(v)
        return v_samples

    def within_cone(self, cone, n, v):
        """
        Checks whether or not a direction is in the friction cone.
        This is equivalent to whether a grasp will slip using a point contact model.

        Parameters
        ----------
        cone : 3xN :obj:`numpy.ndarray`
            supporting vectors of the friction cone
        n : 3x1 :obj:`numpy.ndarray`
            outward pointing surface normal vector at c1
        v : 3x1 :obj:`numpy.ndarray`
            direction vector

        Returns
        -------
        in_cone : bool
            True if alpha is within the cone
        alpha : float
            the angle between the normal and v
        """
        if (v.dot(cone) < 0).any(): # v should point in same direction as cone
            v = -v # don't worry about sign, we don't know it anyway...
        f = -n / np.linalg.norm(n)
        alpha = np.arccos(f.T.dot(v) / np.linalg.norm(v))
        return alpha <= np.arctan(self.friction_coef), alpha

    def perturb_point(self, x, scale):
        """ Uniform random perturbations to a point """
        x_samp = x + (scale / 2.0) * (np.random.rand(3) - 0.5)
        return x_samp

    def sample_grasps(self, graspable, num_grasps=100, vis=False, quality_fn=None):
        """Returns a list of candidate grasps for graspable object.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            number of grasps to sample
        vis : bool
            whether or not to visualize progress, for debugging

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            the sampled grasps
        """
        # get surface points
        grasps = []
        qualities = []
        surface_points, _, surface_points_grid = graspable.sdf.surface_points(grid_basis=False)
        np.random.shuffle(surface_points)
        indices = np.random.choice(np.arange(len(surface_points)),
                                   size=min(self.max_num_surface_points_, len(surface_points)),
                                   replace=False)
        shuffled_surface_points = surface_points[indices]
        # shuffled_surface_points_grid = surface_points_grid[indices]
        logging.info('Num surface: %d' % (len(surface_points)))

        for k, x_surf in enumerate(shuffled_surface_points):
            start_time = time.clock()

            # perturb grasp for num samples
            for i in range(1):  # Se could set this to a different value than 1 and get slightly different contacts
                # perturb contact (TODO: sample in tangent plane to surface)
                x1 = self.perturb_point(x_surf, graspable.sdf.resolution)

                # compute friction cone faces
                c1 = Contact3D(graspable, x1, in_direction=None)
                _, tx1, ty1 = c1.tangents()
                cone_succeeded, cone1, n1 = c1.friction_cone(self.num_cone_faces, self.friction_coef)
                if not cone_succeeded:
                    continue
                cone_time = time.clock()

                # sample grasp axes from friction cone
                v_samples = self.sample_from_cone(n1, tx1, ty1, num_samples=self.num_samples)
                sample_time = time.clock()
                grasp_possibilities = []
                grasp_qualities = []

                for v in v_samples:
                    # random axis flips since we don't have guarantees on surface normal directions
                    if random.random() > 0.5:
                        v = -v

                    # start searching for contacts
                    grasp, c1, c2 = ParallelJawPtGrasp3D.grasp_from_contact_and_axis_on_grid(graspable, x1, v, 0.12,
                                                                                             min_grasp_width_world=self.gripper.min_width,
                                                                                             vis=vis, backup=0.2)

                    if grasp is None or c2 is None:
                        continue

                    # get true contacts (previous is subject to variation)
                    success, c = grasp.close_fingers(graspable, check_approach=False)
                    if not success:
                        continue
                    c1 = c[0]
                    c2 = c[1]

                    # make sure grasp is wide enough
                    x2 = c2.point
                    if np.linalg.norm(x1 - x2) < self.min_contact_dist:
                        continue

                    v_true = grasp.axis
                    # compute friction cone for contact 2
                    cone_succeeded, cone2, n2 = c2.friction_cone(self.num_cone_faces, self.friction_coef)
                    if not cone_succeeded:
                        continue

                    # check friction cone
                    if PointGraspMetrics3D.force_closure(c1, c2, self.friction_coef):
                        if quality_fn is not None:
                            grasp_qualities.append(quality_fn(grasp).quality)
                            grasp_possibilities.append(grasp)
                        else:
                            grasps.append(grasp)
                if quality_fn is not None and grasp_qualities:
                    max_quality = max(grasp_qualities)
                    grasps.append(grasp_possibilities[grasp_qualities.index(max_quality)])
                    qualities.append(max_quality)

        # randomly sample max num grasps from total list
        # random.shuffle(grasps)
        return grasps, qualities









