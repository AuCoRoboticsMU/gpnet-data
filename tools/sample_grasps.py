import numpy as np
from dexnet.database import Hdf5Database
from autolab_core import YamlConfig
from dexnet.constants import READ_WRITE_ACCESS
from dexnet.grasping import RobotGripper
from dexnet.visualization import DexNetVisualizer3D as vis
from dexnet.grasping import grasp_sampler, GraspQualityConfigFactory, GraspQualityFunctionFactory, GraspableObject3D
import dexnet.database.mesh_processor as mp
import shutil
import os
import tempfile
import time

"""
This python script can read in raw object meshes, rescale them, sample grasps over the mesh surface and calculate
their robust_force_closure metric.
"""

general_config = YamlConfig('cfg/tools/config.yaml')
gripper = RobotGripper.load(general_config['gripper_name'])
metric_name = 'robust_force_closure'
mesh_path = general_config['raw_mesh_path']
config = YamlConfig('cfg/tools/grasp_sampling.yaml')


database = Hdf5Database('/data/{}.hdf5'.format(general_config['dataset_name']),
                        access_level=READ_WRITE_ACCESS,
                        cache_dir='.{}'.format(general_config['dataset_name']))

if 'ferrari' in metric_name:
    threshold = 0.002
else:
    threshold = 0.5


sampler = grasp_sampler.AntipodalGraspSampler(gripper, config)
metric_config = config['metrics'][metric_name]
quality_config = GraspQualityConfigFactory.create_config(config['metrics'][metric_name])

obj_cnt = 1
try:
    for dset_name in os.listdir(mesh_path):
        dset_path = '{}{}'.format(mesh_path, dset_name)
        database.create_dataset(dset_name)
        dset = database.dataset(dset_name)
        all_meshes = os.listdir(dset_path)

        for mesh_file in all_meshes:
            mp_cache = tempfile.mkdtemp()
            file_name = '{}/{}'.format(dset_path, mesh_file)
            config['obj_target_scale'] = np.random.uniform(0.06, 0.1)
            mesh_processor = mp.MeshProcessor(file_name, mp_cache)
            try:
                mesh_processor.generate_graspable(config)
            except IndexError:
                obj_cnt += 1
                continue
            if len(mesh_processor.stable_poses) == 0:
                # We don't want to sample grasps that we can't use anyway, it's a waste of time!
                obj_cnt += 1
                print("We didn't find any stable poses. Don't include this object in our dataset.")
                continue
            shutil.rmtree(mp_cache)

            key = mesh_processor.key
            dset.create_graspable(key=key,
                                  mesh=mesh_processor.mesh,
                                  sdf=mesh_processor.sdf,
                                  stable_poses=mesh_processor.stable_poses,
                                  mass=1.0)
            graspable = dset.graspable(key)
            quality_fn = GraspQualityFunctionFactory.create_quality_function(graspable, quality_config)
            sample_time = time.clock()
            grasps, grasp_quality_metric = sampler.generate_grasps(graspable,
                                                                   max_iter=config['max_grasp_sampling_iters'],
                                                                   quality_fn=quality_fn)  # sample grasps
            post_sampling_time = time.clock()
            print("Sampling took {}".format(post_sampling_time - sample_time))

            pos_cnt = 0

            metrics = {}
            for cnt in range(len(grasps)):
                metrics[cnt] = {metric_name: grasp_quality_metric[cnt]}
                if grasp_quality_metric[cnt] > threshold:
                    pos_cnt += 1
            if len(grasps) > 0:
                print("Object #{} of {}; grasp positivity rate {:.02f}".format(obj_cnt,
                                                                               len(all_meshes),
                                                                               100.0 * pos_cnt / len(grasps)))
            dset.store_grasps(key, grasps, gripper=gripper.name)
            dset.store_grasp_metrics(key, metrics, gripper=gripper.name)
            obj_cnt += 1
            break

finally:
    print("Done. Close hdf5 database.")
    database.close()
