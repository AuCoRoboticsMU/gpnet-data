import numpy as np
import cv2
import os
import multiprocessing
from autolab_core import YamlConfig

general_config = YamlConfig('cfg/tools/config.yaml')
root = '/data/{}'.format(general_config['dataset_name'])
USE_MULTIPROCESSING = False

def add_gaussian_shifts(depth, std=1 / 2.0):
    rows, cols = depth.shape
    gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates
    xx = np.linspace(0, cols - 1, cols)
    yy = np.linspace(0, rows - 1, rows)

    # get xpixels and ypixels
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

    return depth_interp

def filterDisp(disp, dot_pattern_, invalid_disp_):
    size_filt_ = 9

    xx = np.linspace(0, size_filt_ - 1, size_filt_)
    yy = np.linspace(0, size_filt_ - 1, size_filt_)

    xf, yf = np.meshgrid(xx, yy)

    xf = xf - int(size_filt_ / 2.0)
    yf = yf - int(size_filt_ / 2.0)

    sqr_radius = (xf ** 2 + yf ** 2)
    vals = sqr_radius * 1.2 ** 2

    vals[vals == 0] = 1
    weights_ = 1 / vals

    fill_weights = 1 / (1 + sqr_radius)
    fill_weights[sqr_radius > 9] = -1.0

    disp_rows, disp_cols = disp.shape
    dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape

    lim_rows = np.minimum(disp_rows - size_filt_, dot_pattern_rows - size_filt_)
    lim_cols = np.minimum(disp_cols - size_filt_, dot_pattern_cols - size_filt_)

    center = int(size_filt_ / 2.0)

    window_inlier_distance_ = 0.1

    out_disp = np.ones_like(disp) * invalid_disp_

    interpolation_map = np.zeros_like(disp)

    r = np.arange(lim_rows)
    c = np.arange(lim_cols)

    r_all = np.repeat(r, len(c))
    c_all = np.tile(c, len(r))

    mask = dot_pattern_[r_all + center, c_all + center] > 0

    c_all = c_all[mask]
    r_all = r_all[mask]

    windows = []
    dot_windows = []
    for i in range(size_filt_):
        for k in range(size_filt_):
            windows.append(disp[r_all + i, c_all + k])
            dot_windows.append(dot_pattern_[r_all + i, c_all + k])

    windows = np.array(windows).T.reshape(len(c_all), 9, 9)
    dot_windows = np.array(dot_windows).T.reshape(len(c_all), 9, 9)

    valid_dots = np.where(windows < invalid_disp_, dot_windows, 0)

    all_n_valids = np.sum(np.sum(valid_dots, axis=1), axis=1) / 255.0
    all_n_thresh = np.sum(np.sum(dot_windows, axis=1), axis=1) / 255.0

    mask = np.where(all_n_valids > all_n_thresh / 1.2)

    filtered_windows = windows[mask]
    filtered_dot_windows = dot_windows[mask]
    filtered_n_thresh = all_n_thresh[mask]
    r_all = r_all[mask]
    c_all = c_all[mask]

    all_mean = np.nanmean(np.where(filtered_windows < invalid_disp_, filtered_windows, np.nan), axis=(1, 2))

    all_diffs = np.abs(np.subtract(filtered_windows, np.repeat(all_mean, 81).reshape(len(all_mean), 9, 9)))
    all_diffs = np.multiply(all_diffs, weights_)

    all_cur_valid_dots = np.multiply(np.where(filtered_windows < invalid_disp_, filtered_dot_windows, 0),
                                     np.where(all_diffs < window_inlier_distance_, 1, 0))

    all_n_valids = np.sum(all_cur_valid_dots, axis=(1, 2)) / 255.0

    mask = np.where(all_n_valids > filtered_n_thresh / 1.2)

    filtered_windows = filtered_windows[mask]
    r_all = r_all[mask]
    c_all = c_all[mask]

    accu = filtered_windows[:, center, center]

    for i in range(len(c_all)):
        r = r_all[i]
        c = c_all[i]
        out_disp[r + center, c + center] = np.round(accu[i] * 8.0) / 8.0

        interpolation_window = interpolation_map[r:r + size_filt_, c:c + size_filt_]
        disp_data_window = out_disp[r:r + size_filt_, c:c + size_filt_]

        substitutes = np.where(interpolation_window < fill_weights, 1, 0)
        interpolation_window[substitutes == 1] = fill_weights[substitutes == 1]

        disp_data_window[substitutes == 1] = out_disp[r + center, c + center]


    return out_disp

def calculate_noise(start, end, worker_num):
    scale_factor = 100  # converting depth from m to cm
    focal_length = 480.0  # focal length of the camera used
    baseline_m = 0.075  # baseline in m
    invalid_disp_ = 99999999.9
    dot_pattern_ = cv2.imread("data/kinect-pattern_3x3.png", 0)
    k = 0
    step_size = (end - start) / 100

    for image_label in range(start, end):
        if USE_MULTIPROCESSING and (k % step_size) == 0:
            print("Worker #{}: {}%".format(worker_num, k // step_size))
        image = np.load('{}/tensors/depth_im_{:07d}.npz'.format(root, image_label))['arr_0'].squeeze()

        depth = image.copy()
        h, w = depth.shape

        depth_interp = add_gaussian_shifts(depth)

        disp_ = focal_length * baseline_m / (depth_interp + 1e-10)
        depth_f = np.round(disp_ * 8.0)/8.0

        out_disp = filterDisp(depth_f, dot_pattern_, invalid_disp_)

        depth = focal_length * baseline_m / out_disp
        depth[out_disp == invalid_disp_] = 0

        # The depth here needs to converted to cms so scale factor is introduced
        # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects
        noisy_depth = (35130/np.round((35130/np.round(depth*scale_factor)) +
                                      np.random.normal(size=(h, w))*(1.0/6.0) + 0.5)) / scale_factor

        np.savez_compressed("{}/noise/depth_im_{:07d}.npz".format(root, image_label), noisy_depth)
        k += 1


if __name__ == '__main__':
    worker_count = 4
    if not os.path.exists('{}/noise'.format(root)):
        os.mkdir('{}/noise'.format(root))
    worker_pool = []
    start = 0
    # Count files generated
    with open('{}/dataset_stats.txt'.format(root), 'read') as f:
        all = f.readlines()
        images_num = int(all[-1].split(':')[-1])
    steps = (images_num - start) / worker_count
    if USE_MULTIPROCESSING:
        for i in range(worker_count):
            new_start = int(start + i * steps)
            new_end = int(start + (1+i) * steps)
            p = multiprocessing.Process(target=calculate_noise, args=(new_start, new_end, i))
            p.start()
            worker_pool.append(p)
        for p in worker_pool:
            p.join()  # Wait for all of the workers to finish.
    else:
        calculate_noise(start, images_num, 1)
    print("Finished")
