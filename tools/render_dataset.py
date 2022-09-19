import numpy as np
import gc
import os
import logging

from meshpy import RenderMode, SceneObject
from autolab_core import RigidTransform, Point
from dexnet.visualization import DexNetVisualizer3D as vis
from dexnet.grasping import GraspCollisionChecker, ParallelJawPtGrasp3D
from dataset_generator import GenerateDataset

"""Script to generate a 4DOF grasp quality dataset from a DexNet mesh dataset."""

DATA_DIR = '/data'
DATASET_DIR = DATA_DIR + '/test/'
CONFIG_DIR = './cfg/tools/render_dataset.yaml'


class ContactPoint:
    def __init__(self, cords=(None, None), width=None, x_axis=None, z_axis=None, quaternion=None):
        self.image_cords = cords  # image coordinates of contact
        self.width = width  # Distance to second contact point
        self.x_axis = x_axis  # Direction of x-axis (to second contact)
        self.z_axis = z_axis  # Direction of z-axis, how to grasp is approached
        self.quaternion = quaternion


class ContactXYQ(GenerateDataset):
    def __init__(self, output_dir, config_dir):
        GenerateDataset.__init__(self, output_dir, config_dir)

        self.num_contact_grasps = 0
        self.num_centre_grasps = 0
        self.num_objects = 0

        self.gripper_name = 'pal_gripper'

        self.metric_name = 'robust_force_closure'
        self.threshold = 0.5
        self._table_mesh_filenames = ['data/meshes/table.obj', 'data/meshes/table_1m.obj',
                                      'data/meshes/table_05m.obj', 'data/meshes/table_05m_offcentre.obj',
                                      'data/meshes/table_1m_2m.obj', 'data/meshes/table_1m_2m_offcentre.obj',
                                      'data/meshes/table_1m_2m_offcentre_2.obj',
                                      'data/meshes/table_1m_2m_offcentre_2_rotated.obj',
                                      'data/meshes/table_1m_offcentre.obj',
                                      'data/meshes/table_offcentre.obj',
                                      'data/meshes/table_orig.obj']

        self.table_meshes = [self._set_table_mesh(table) for table in self._table_mesh_filenames]

        self.image_dir = output_dir + '/tensors/'
        if not os.path.exists(self.image_dir):
            os.mkdir(self.image_dir)

    def _set_tensor_config(self):
        """ Sets the tensor config based on the used config file.

            Returns
            -------
            tensor_config (dict): tensor config that can be used to initiate a tensor dataset.
        """
        tensor_config = self.config['tensors']
        tensor_config['fields']['robust_ferrari_canny'] = {}
        tensor_config['fields']['robust_ferrari_canny']['dtype'] = 'float32'
        return tensor_config

    @staticmethod
    def get_hand_pose(contact_point, grasp_2d, rotation_range, quality, collision_free):
        """ Organises numpy array for hand_pose tensor.

            Parameters
            ----------
            contact_point (ContactPoint)
            grasp_2d (Grasp2D)
            rotation_range (float)
            quality (float)
            collision_free (Boolean)

            Returns
            -------
            hand_pose (np.array): Hand_pose tensor including the image coordinates of the visible contact point,
                                  image coordinates and distance to the camera of the TCP,
                                  width of the grasp, orientation in the camera frame in form of a quaternion,
                                  the maximum collision-free rotation range in radians, collisions of the grasp
                                  and the grasp quality (robust force closure)
        """

        return np.r_[grasp_2d.center.x,  # Grasp center point x image coordinate
                     grasp_2d.center.y,  # Grasp center point y image coordinate
                     grasp_2d.depth,     # Grasp center point depth
                     contact_point.image_cords[0],  # Contact point x
                     contact_point.image_cords[1],  # Contact point y
                     contact_point.width,  # Contact point width
                     contact_point.quaternion,   # w x y z layout
                     rotation_range,  # rotation range in radians
                     quality,  # quality of the grasp [float]
                     collision_free]  # is the grasp collision-free?

    def save_samples(self, sample, aligned_grasps, T_obj_stp,
                     grasp_metrics, stable_pose, contacts):
        T_stp_camera = sample.camera.object_to_camera_pose
        self.T_obj_camera = T_stp_camera * T_obj_stp.as_frames('obj', T_stp_camera.from_frame)

        depth_im_table = sample.renders[RenderMode.DEPTH_SCENE].image
        segmask = sample.renders[RenderMode.SEGMASK].image

        final_camera_intr = sample.camera.camera_intr.crop(self.image_height,
                                                           self.image_width,
                                                           depth_im_table.center[0],
                                                           depth_im_table.center[1])
        camera_intrinsics = np.r_[final_camera_intr.fx,
                                  final_camera_intr.fy,
                                  final_camera_intr.cx,
                                  final_camera_intr.cy]
        self.save_image(depth_im_table, segmask=segmask)

        np.save('{}/camera_intrs_{:07d}.npy'.format(self.image_dir, self.cur_image_label), camera_intrinsics)
        np.save('{}/camera_poses_{:07d}.npy'.format(self.image_dir, self.cur_image_label), self.get_camera_pose(sample))
        np.save('{}/obj_camera_transform_{:07d}.npy'.format(self.image_dir,
                                                            self.cur_image_label),
                self.T_obj_camera.matrix)
        np.save('{}/labels_{:07d}.npy'.format(self.image_dir, self.cur_image_label), np.array((self.cur_pose_label,
                                                                                               self.cur_obj_label,
                                                                                               self.obj.key,
                                                                                               stable_pose.id)))
        contact_grasps = []
        contact_metrics = []
        all_contacts = []

        centre_grasps = []
        centre_metrics = []
        all_centre_grasps = []

        for cnt, grasp in enumerate(aligned_grasps):
            if type(grasp) == list:
                # We have multiple regions and have to choose one of them!
                z_axis = []
                for grasp_option in grasp:
                    T_grasp_camera = self.T_obj_camera * grasp_option[0].gripper_pose(self.gripper)
                    z_axis.append(T_grasp_camera.z_axis[2])
                # Choose the grasp coming in from the closest to the camera
                grasp = grasp[np.argmax(z_axis)]

            cf = False if grasp[1] == -1 else True
            rotation_range = grasp[1]
            chosen_grasp = grasp[0]
            T_grasp_camera = self.T_obj_camera * chosen_grasp.gripper_pose(self.gripper)

            # Project contacts into image
            _c1 = np.matmul(self.T_obj_camera.matrix, np.concatenate([contacts[cnt][0], [1]]))[:3]  # To camera frame
            p_grasp_camera = Point(_c1, frame=final_camera_intr.frame)
            c1_u = final_camera_intr.project(p_grasp_camera)  # Project

            _c2 = np.matmul(self.T_obj_camera.matrix, np.concatenate([contacts[cnt][1], [1]]))[:3]
            p_grasp_camera = Point(_c2, frame=final_camera_intr.frame)
            c2_u = final_camera_intr.project(p_grasp_camera)

            if np.any(c1_u.data < 0) or c1_u[1] >= self.image_height or c1_u[0] >= self.image_width or\
                    np.any(c2_u.data < 0) or c2_u[0] >= self.image_width or c2_u[1] >= self.image_height:
                # Contacts outside of image - can't see full grasp/object -- exclude from dataset
                continue

            # Project grasp centre into image
            grasp_2d = chosen_grasp.project_camera(self.T_obj_camera, final_camera_intr)

            cp = ContactPoint()
            self.num_centre_grasps += 1

            # We need to check if the contacts are occluded or visible by the camera
            # Note we go for c1_u[1], c1_u[0] in that order, since we're changing from image coords (x, y) to array
            # coords (row, column) == (y, x)
            c1_dist = depth_im_table[c1_u[1], c1_u[0]] - _c1[2]  # Subtract contact depth from depth at contact cords
            c2_dist = depth_im_table[c2_u[1], c2_u[0]] - _c2[2]

            c1_visible = False
            c2_visible = False

            if c1_dist >= 0.0 and c2_dist >= 0.0:
                # Both contacts seem to be in front of the object (this should rarely happen, I think)
                if c1_dist < c2_dist:
                    # We decide which one to take according to which contact is closer to the pixel depth we see
                    c1_visible = True
                else:
                    c2_visible = True
            elif c1_dist >= 0.0:
                # Contact one is visible
                c1_visible = True
            elif c2_dist >= 0.0:
                # Contact two is visible
                c2_visible = True

            # Store our contact point details according to which contact we deemed visible
            # We store our contact points in image coords (u, v), rather than numpy coords!
            if c1_visible:
                cp.image_cords = (c1_u[0], c1_u[1])
                c_dir = _c2 - _c1
                self.num_contact_grasps += 1
            elif c2_visible:
                cp.image_cords = (c2_u[0], c2_u[1])
                c_dir = _c1 - _c2
                # We are rotating the gripper by 180 degree to always have the same gripper side at the contact
                # Otherwise, the gripper orientation would be ambigious for the contact-point grasp representation
                T_grasp_camera.rotation = np.matmul(T_grasp_camera.rotation, T_grasp_camera.z_axis_rotation(np.pi))

                self.num_contact_grasps += 1
            else:
                # No contacts are visible
                cp.image_cords = (None, None)
                c_dir = _c1 - _c2

            # If we already stored a grasp for that image coordinate, we don't want to add another one
            current_metric = grasp_metrics[chosen_grasp.id][self.metric_name] * cf

            store_contact = True
            if cp.image_cords[0] is None:
                # We don't want to store it if we can't see the contact points!
                store_contact = False
            elif cp.image_cords in all_contacts:
                ind = all_contacts.index(cp.image_cords)
                old_metric = contact_metrics[ind]
                if current_metric > old_metric:
                    # The new one seems to be a better grasp - use this one!
                    all_contacts.pop(ind)
                    contact_grasps.pop(ind)
                    contact_metrics.pop(ind)
                else:
                    # We don't want to add it if we have a better one stored already
                    store_contact = False

            store_centre = True
            centre_pixel = (grasp_2d.center.x, grasp_2d.center.y)
            if centre_pixel in all_centre_grasps:
                ind = all_centre_grasps.index(centre_pixel)
                old_metric = centre_metrics[ind]
                if current_metric > old_metric:
                    # The new one seems to be a better grasp - use this one!
                    all_centre_grasps.pop(ind)
                    centre_grasps.pop(ind)
                    centre_metrics.pop(ind)
                else:
                    # We don't want to add it if we have a better one stored already
                    store_centre = False

            cp.width = np.sqrt(c_dir[0] ** 2 + c_dir[1] ** 2 + c_dir[2] ** 2)
            cp.x_axis = T_grasp_camera.x_axis
            cp.z_axis = T_grasp_camera.z_axis
            cp.quaternion = T_grasp_camera.quaternion

            grasp_pose = self.get_hand_pose(cp, grasp_2d, rotation_range, current_metric, cf)
            if store_contact:
                all_contacts.append(cp.image_cords)
                contact_grasps.append(grasp_pose)
                contact_metrics.append(current_metric)
            if store_centre:
                all_centre_grasps.append(centre_pixel)
                centre_grasps.append(grasp_pose)
                centre_metrics.append(current_metric)

        np.savez_compressed('{}/contact_grasps_{:07d}.npz'.format(self.image_dir, self.cur_image_label),
                            np.array(contact_grasps).astype(np.float32))
        np.savez_compressed('{}/centre_grasps_{:07d}.npz'.format(self.image_dir, self.cur_image_label),
                            np.array(centre_grasps).astype(np.float32))

        self.cur_image_label += 1

    def get_contacts(self, grasp):
        grasp_point = grasp.close_fingers(self.obj, check_approach=False, vis=False)
        if grasp_point[0]:
            # Contacts in object coordinates
            return [grasp_point[1][0].point, grasp_point[1][1].point]
        else:
            # Could not find contacts
            return False

    def _get_collision_vector(self, grasp, collision_checker, stable_pose, steps_in_deg=15):
        phi_offsets = np.deg2rad(np.arange(0, 360, steps_in_deg))
        steps_in_rad = np.deg2rad(steps_in_deg)

        grasp_collisions = []
        for phi_offset in phi_offsets:
            rotated_grasp = grasp.grasp_y_axis_offset(phi_offset)
            collides = collision_checker.collides_along_approach(rotated_grasp, self._approach_dist,
                                                                 self._delta_approach)
            grasp_collisions.append(collides)
        collisions = np.array(grasp_collisions).astype(bool)
        if not collisions[0] and not collisions[-1] and np.any(collisions):
            # We need to have our collision free regions combined, so we roll our array if they are not
            while collisions[-1] == False:
                collisions = np.roll(collisions, 1)
        if np.all(collisions):
            # We don't have a grasp orientation in here that doesn't collide with anything
            # Return the grasp most aligned with the table normal
            return grasp.perpendicular_table(stable_pose), -1
        elif np.all(~collisions):
            # We don't collide at all - I can't think of a situation where this would occur, so let's raise an error
            vis.figure()
            T_obj_world = vis.mesh_stable_pose(self.obj.mesh.trimesh,
                                               stable_pose.T_obj_world, style='surface', dim=1.0)
            vis.gripper(self.gripper, grasp, T_obj_world, color=(0.3, 0.3, 0.3))
            vis.show()
            raise UserWarning("No collision when rotating 360deg around the grasp axis - don't think this is viable")
        else:
            # We have a viable grasp orientation in here
            # Collapse consecutive identical values
            collision_free = np.where(collisions == False)[0]
            regions = self._consecutive_numbers(collision_free)
            if len(regions) == 1:
                return self._get_grasp_and_rotation_flexibility(grasp, regions[0], steps_in_rad)
            else:
                grasps = []
                for region in regions:
                    # We probably shouldn't need to do this more than twice, normally?
                    grasps.append(self._get_grasp_and_rotation_flexibility(grasp, region, steps_in_rad))
                return grasps

    @staticmethod
    def _get_grasp_and_rotation_flexibility(grasp, collision_free, rotation_steps):
        middle = len(collision_free) // 2  # What's the centre grasp in our rotation region?
        approach_angle = collision_free[middle] * rotation_steps
        return grasp.grasp_y_axis_offset(approach_angle), middle * rotation_steps

    @staticmethod
    def _consecutive_numbers(data):
        return np.split(data, np.where(np.diff(data) != 1)[0] + 1)

    def render_data(self):
        logging.basicConfig(level=logging.WARNING)
        all_grasps = 0
        all_good_grasps = 0
        for dataset in self.datasets:
            logging.info('Generating data for dataset %s' % dataset.name)
            object_keys = dataset.object_keys

            for obj_key in object_keys:
                self.cur_obj_label += 1
                if self.cur_obj_label % 5 == 0:
                    logging.info("Object number: {} of {}".format(self.cur_obj_label, len(object_keys)))

                self.obj = dataset[obj_key]

                grasps = dataset.grasps(self.obj.key, gripper=self.gripper_name)

                if len(grasps) == 0:
                    print("No grasps found. Continue")
                    continue

                # Load grasp metrics
                grasp_metrics = dataset.grasp_metrics(self.obj.key,
                                                      grasps,
                                                      gripper=self.gripper_name)

                metrics = [grasp_metrics[grasp.id][self.metric_name] for grasp in grasps]
                success = np.where(np.array(metrics) >= self.threshold, True, False)

                good_grasps = sum(success)
                all_good_grasps += good_grasps

                all_grasps += len(grasps)
                # setup collision checker
                collision_checker = GraspCollisionChecker(self.gripper)
                collision_checker.set_graspable_object(self.obj)

                # read in the stable poses of the mesh
                try:
                    stable_poses = dataset.stable_poses(self.obj.key)
                except KeyError:
                    print("No stable poses associated with {}.".format(self.obj.key))
                    continue

                # Iterate through stable poses
                for stable_pose in stable_poses:
                    if not stable_pose.p > self._stable_pose_min_p:
                        continue

                    # setup table in collision checker
                    T_obj_stp = stable_pose.T_obj_table.as_frames('obj', 'stp')
                    T_obj_table = self.obj.mesh.get_T_surface_obj(T_obj_stp,
                                                                  delta=self._table_offset).as_frames('obj', 'table')
                    T_table_obj = T_obj_table.inverse()
                    T_obj_stp = self.obj.mesh.get_T_surface_obj(T_obj_stp)

                    # Use different tables to include all kind of tables in our dataset
                    table_choice = np.random.randint(0, len(self._table_mesh_filenames))
                    table_mesh_filename = self._table_mesh_filenames[table_choice]
                    table_mesh = self.table_meshes[table_choice]

                    collision_checker.set_table(table_mesh_filename, T_table_obj)

                    # sample images from random variable
                    T_table_obj = RigidTransform(from_frame='table', to_frame='obj')
                    scene_objs = {'table': SceneObject(table_mesh, T_table_obj)}

                    grasps_and_collisions = []
                    contacts = []

                    for grasp, quality in zip(grasps, success):
                        contact_points = self.get_contacts(grasp)
                        if not contact_points:
                            continue
                        contacts.append(contact_points)
                        if quality:
                            # Check collision-free areas and how to best orient our grasp.
                            grasps_and_collisions.append(self._get_collision_vector(grasp,
                                                                                    collision_checker,
                                                                                    stable_pose))
                        else:
                            # When our grasp is bad, we don't care about the grasp pose or the collision range
                            grasps_and_collisions.append((grasp, -1))

                    # # Set up image renderer
                    samples = self.render_images(scene_objs,
                                                 stable_pose,
                                                 self.config['images_per_stable_pose'],
                                                 camera_config=self._camera_configs())
                    if self.config['images_per_stable_pose'] == 1:
                        self.save_samples(samples, grasps_and_collisions, T_obj_stp,
                                          grasp_metrics, stable_pose, contacts)
                    else:
                        for sample in samples:
                            self.save_samples(sample, grasps_and_collisions, T_obj_stp,
                                              grasp_metrics, stable_pose, contacts)
                    self.cur_pose_label += 1
                    gc.collect()
                    # next stable pose
                # next object
                self.num_objects += 1
        # Save dataset
        with open('{}/../dataset_stats.txt'.format(self.image_dir), 'w') as f:
            lines = ['Number of centre grasps: {}'.format(self.num_centre_grasps),
                     'Number of contact grasps: {}'.format(self.num_contact_grasps),
                     'Number of objects: {}'.format(self.num_objects),
                     'Number of stable poses: {}'.format(self.cur_pose_label),
                     'Number of images: {}'.format(self.cur_image_label)]
            f.writelines('\n'.join(lines))
        print("Good grasps: {}; percentage: {:.1f}".format(all_good_grasps, 100 * all_good_grasps / all_grasps))


if __name__ == "__main__":
    Generator = ContactXYQ(DATASET_DIR, CONFIG_DIR)
    Generator.render_data()
