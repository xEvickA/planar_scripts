import datetime
import os
import pickle
import time

import numpy as np
import pyrobustac


def get_pose_diff(pose_src, pose_dst):
    T_src = np.concatenate([pose_src, np.array([[0, 0, 0, 1.0]])], axis=0)
    T_dst = np.concatenate([pose_dst, np.array([[0, 0, 0, 1.0]])], axis=0)
    T_diff = np.linalg.inv(T_src) @ T_dst

    return T_src, T_dst, T_diff


def export_pkl(path, sequence_name, save_every=10, start_from_last=True):
    poses_file = os.path.join(path, 'poses', '{}.txt'.format(sequence_name))
    poses = np.loadtxt(poses_file)
    poses = np.reshape(poses, [-1, 3, 4])

    img_dir = os.path.join(path, 'sequences', sequence_name, 'image_0')
    img_filenames = os.listdir(img_dir)

    pkl_path = os.path.join(path, 'extracted_features', '{}.pkl'.format(sequence))

    if len(img_filenames) != poses.shape[0]:
        print("Something went wrong! Pose entries and images do not match!")
        return
    else:
        print("Loaded seqence {}".format(sequence))

    if start_from_last:
        try:
            with open(pkl_path, 'rb') as f:
                dict_list = pickle.load(f)
                start=len(dict_list)
        except Exception:
            dict_list = []
            start = 0
    else:
        dict_list = []
        start = 0


    start_time = time.time()

    for i in range(start, len(img_filenames) - 1):
        pose_src = poses[i]
        pose_dst = poses[i + 1]

        T_src, T_dst, T_diff = get_pose_diff(pose_src, pose_dst)

        img_filename_src = os.path.join(img_dir, img_filenames[i])
        img_filename_dst = img_filenames[i + 1]

        src_pts, dst_pts, affines = pyrobustac.extractACs(os.path.join(img_dir, img_filename_src), os.path.join(img_dir, img_filename_dst))

        d = {'src_pts': src_pts, 'dst_pts': dst_pts, 'affines': affines,
             'img_filename_src': img_filename_src, 'img_filename_dst': img_filename_dst,
             'T_src': T_src, 'T_dst': T_dst, 'T_diff': T_diff, 'R_diff': T_diff[:3, :3], 't_diff': T_diff[:3, 3]}

        total_time = time.time() - start_time
        per_pair_time = total_time / (i + 1 - start)
        print("Average time per image pair: {}".format(per_pair_time))
        print("Estimated time left: {}".format(datetime.timedelta(seconds=per_pair_time * (len(img_filenames) - 1))))

        dict_list.append(d)

        if i % save_every == 0:
            print("Saving for sequence {} at index {}".format(sequence, i))
            with open(pkl_path, 'wb') as handle:
                pickle.dump(dict_list, handle)

    print("Saving sequence {} final".format(sequence))
    with open(pkl_path, 'wb') as handle:
        pickle.dump(dict_list, handle)


if __name__ == '__main__':
    odometry_path = 'D:/Skola/PhD/data/KITTI/odometry/dataset/'
    sequences = ["{:02d}".format(x) for x in range(0, 11)]

    for sequence in sequences:
        export_pkl(odometry_path, sequence)
