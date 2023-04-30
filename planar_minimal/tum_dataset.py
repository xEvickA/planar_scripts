import os

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from planar_minimal.utils import angle


def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.

    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.

    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < 1e-8:
        return np.array([
            [1.0, 0.0, 0.0, t[0]]
            [0.0, 1.0, 0.0, t[1]]
            [0.0, 0.0, 1.0, t[2]]
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], t[0]],
        [q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], t[1]],
        [q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], t[2]],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)


def load_data(dir):
    # loads TUM dataset
    # output columns are: timestamp, tx, ty, tz, qx, qy, qz, qw
    gt_filename = os.path.join(dir, 'groundtruth.txt')

    with open(gt_filename) as f:
        content = f.readlines()
    gt_data = [x.strip().split(" ") for x in content[3:]]
    gt_data = [[float(x) for x in y] for y in gt_data]
    gt_data = np.array(gt_data)

    print("Loaded {} with {} entries".format(gt_filename, len(gt_data)))

    img_list = os.listdir(os.path.join(dir, 'rgb'))

    img_dict = {}
    img_stamps = []
    for img_name in img_list:
        timestamp = float(img_name.split(".png")[0])
        img_stamps.append(timestamp)
        img_filename = os.path.join(dir, 'rgb', img_name)
        img_dict[timestamp] = img_filename

    img_stamps = np.array(sorted(img_stamps))

    return gt_data, img_stamps, img_dict


def find_closest(timestamp, stamp_array, return_idx=False):
    idx = (np.abs(stamp_array - timestamp)).argmin()
    if return_idx:
        return idx
    return stamp_array[idx]


def extract_planar_dir(dir, tol=0.01):
    gt_data,  img_stamps, img_dict = load_data(dir)

    Ts = [transform44(l) for l in gt_data]

    out_path = os.path.join(dir, 'planar.txt')

    i_start = find_closest(img_stamps[0], gt_data[:, 0], return_idx=True)
    i_end = find_closest(img_stamps[-1], gt_data[:, 0], return_idx=True)

    for i in range(i_start, i_end):
        T_i = Ts[i]

        # t_i = gt_data[i, 1:4]
        # q_i = gt_data[i, 4:]
        # r_i = Rotation.from_quat(q_1)

        for j in range(i + 1, i_end):
            T_j = Ts[j]

            T_d = np.linalg.inv(T_j) @ T_i

            R_d = T_d[:3, :3]
            t_d = T_d[:3, 3]

            r_d = Rotation.from_matrix(R_d)

            a = angle(t_d, r_d.as_rotvec())
            # a = 0

            if 90.0 - tol < a < 90.0 + tol:
                gt_timestamp_i = gt_data[i, 0]
                gt_timestamp_j = gt_data[j, 0]

                img_timestamp_i = find_closest(gt_timestamp_i, img_stamps)
                img_timestamp_j = find_closest(gt_timestamp_j, img_stamps)

                if np.abs(img_timestamp_i - gt_timestamp_i) > 0.01 or np.abs(img_timestamp_j - gt_timestamp_j) > 0.01:
                    continue

                print("Found planar with angle: ", a)
                print("1st gt frame at {} img frame at {}".format(gt_timestamp_i, img_timestamp_i))
                print("2nd gt frame at {} img frame at {}".format(gt_timestamp_j, img_timestamp_j))

                img_i = cv2.imread(img_dict[img_timestamp_i])
                img_j = cv2.imread(img_dict[img_timestamp_j])

                r_quat = r_d.as_quat()
                line = ('{},' * 10 + '{} \n').format(img_dict[img_timestamp_i], img_dict[img_timestamp_j], gt_timestamp_i, gt_timestamp_j, *t_d, *r_quat)
                print(r_d.as_rotvec(), t_d)

                with open(out_path, 'a') as f:
                    f.write(line)

                print(line)

                cv2.imshow("img_i", img_i)
                cv2.imshow("img_j", img_j)

                cv2.waitKey(1)

    print(gt_data)

if __name__ == '__main__':
    dir = os.path.join('dataset', 'rgbd_dataset_freiburg1_desk')
    extract_planar_dir(dir)