import numpy as np
import os
import json

from meshpy import UniformPlanarWorksurfaceImageRandomVariable, ObjFile, RenderMode
from autolab_core import Point, YamlConfig
from dexnet.learning import TensorDataset
from PIL import Image, ImageDraw
from dexnet.constants import READ_ONLY_ACCESS
from dexnet.grasping import RobotGripper
from dexnet.database import Hdf5Database

"""
Script to render dexnet_2.0 dataset.
"""


class GenerateDataset:
    def __init__(self, output_dir, config_dir):
        self.config = YamlConfig(config_dir)
        self.general_config = YamlConfig('cfg/tools/config.yaml')

        self.datasets, self.target_object_keys = self._load_datasets()

        if os.path.exists(output_dir):
            in_choice = raw_input("Dataset already exists. Overwrite? ")
            if in_choice != 'y':
                raise AttributeError("Dataset already exists and user opted not to overwrite. Abort.")
        else:
            os.mkdir(output_dir)
            os.mkdir('{}/tensors'.format(output_dir))

        json.dump(self._set_general_config(), open(output_dir + '/tensors/config.json', 'w'))
        self.output_dir = output_dir
        self.image_dir = output_dir + '/images/'
        self.gripper = self._set_gripper()

        self.cur_pose_label = 0
        self.cur_obj_label = -1
        self.cur_image_label = 0

        self.image_height = self.config['env_rv_params']['im_height']
        self.image_width = self.config['env_rv_params']['im_width']

        self.obj = None

    def _camera_configs(self):
        return self.config['env_rv_params'].copy()

    @property
    def _render_modes(self):
        return [RenderMode.DEPTH_SCENE, RenderMode.SEGMASK]

    @property
    def _approach_dist(self):
        return self.config['collision_checking']['approach_dist']

    @property
    def _delta_approach(self):
        return self.config['collision_checking']['delta_approach']

    @property
    def _table_offset(self):
        return self.config['collision_checking']['table_offset']

    @property
    def _stable_pose_min_p(self):
        return self.config['stable_pose_min_p']

    def _set_table_mesh_filename(self):
        table_mesh_filename = self.config['collision_checking']['table_mesh_filename']
        if not os.path.isabs(table_mesh_filename):
            return os.path.join(DATA_DIR, table_mesh_filename)
        return table_mesh_filename

    def _set_table_mesh(self, name=None):
        if name is not None:
            return ObjFile(name).read()
        return ObjFile(self._table_mesh_filename).read()

    def _set_gripper(self, gripper=None):
        if gripper is None:
            gripper = self.general_config['gripper_name']
        return RobotGripper.load(gripper)

    def _load_datasets(self):
        database = Hdf5Database('/data/{}.hdf5'.format(self.general_config['dataset_name']),
                                access_level=READ_ONLY_ACCESS,
                                cache_dir='.{}'.format(self.general_config['dataset_name']))
        target_object_keys = self.config['target_objects']
        dataset_names = target_object_keys.keys()
        datasets = [database.dataset(dn) for dn in dataset_names]
        return datasets, target_object_keys

    def _set_tensor_config(self):
        pass

    def _set_general_config(self):
        gen_config = dict()
        gen_config['env_rv_params'] = self.config['env_rv_params']
        gen_config['collision_checking'] = self.config['collision_checking']
        gen_config['images_per_stable_pose'] = self.config['images_per_stable_pose']
        gen_config['gripper'] = self.general_config['gripper_name']
        gen_config['database_name'] = self.general_config['dataset_name']
        gen_config['target_objects'] = self.config['target_objects']
        gen_config['stable_pose_min_p'] = self.config['stable_pose_min_p']
        return gen_config

    def render_images(self, scene_objs, stable_pose, num_images, camera_config=None):
        """ Renders depth and binary images from self.obj at the given stable pose. The camera
            sampling occurs within urv.rvs.

            Parameters
            ----------
            scene_objs (dict): Objects occuring in the scene, mostly includes the table mesh.
            stable_pose (StablePose): Stable pose of the object
            num_images (int): Numbers of images to render
            camera_config (dict): Camera sampling parameters with minimum/maximum values for radius, polar angle, ...

            Returns
            -------
            render_samples (list): list of rendered images including sampled camera positions
        """
        if camera_config is None:
            camera_config = self.config['env_rv_params']

        urv = UniformPlanarWorksurfaceImageRandomVariable(self.obj.mesh,
                                                          self._render_modes,
                                                          'camera',
                                                          camera_config,
                                                          scene_objs=scene_objs,
                                                          stable_pose=stable_pose)
        # Render images
        render_samples = urv.rvs(size=num_images)
        return render_samples

    def get_camera_pose(self, sample):
        return np.r_[sample.camera.radius,
                     sample.camera.elev,
                     sample.camera.az,
                     sample.camera.roll,
                     sample.camera.focal,
                     sample.camera.tx,
                     sample.camera.ty]

    def save_image(self, depth, segmask=None, size=None):
        np.savez_compressed(self.image_dir + "/depth_im_{:07d}.npz".format(self.cur_image_label),
                            np.asarray(depth.data))
        if segmask is not None:
            np.savez_compressed(self.image_dir + "/binary_{:07d}.npz".format(self.cur_image_label),
                                np.asarray(segmask.data))

