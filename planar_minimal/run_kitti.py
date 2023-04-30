import os
import pickle

import numpy as np
from estimators_fundamental import *
from utils import display_errors

KITTI_K_GT = np.array([[7.188560000000e+02, 0.0, 6.071928000000e+02],
                 [0.0, 7.188560000000e+02, 1.852157000000e+02],
                 [0.0, 0.0, 1.0]])

KITTI_H = 376
KITTI_W = 1241

def evaluate_sequence(pkl_path, estimators):
    t_errs = [[] for _ in estimators]
    R_errs = [[] for _ in estimators]
    inliers = [[] for _ in estimators]
    times = [[] for _ in estimators]

    with open(pkl_path, 'rb') as f:
        entries = pickle.load(f)

    for entry in entries:
        src_pts = entry['src_pts']
        dst_pts = entry['dst_pts']
        affines = entry['affines']
        t_gt = entry['t_diff']
        R_gt = entry['R_diff']

        for i, estimator in enumerate(estimators):
            F, mask, time = estimator.estimate_timed(src_pts, dst_pts, A=affines, h1=KITTI_H, w1=KITTI_W, h2=KITTI_H, w2=KITTI_W, threshold=1.0, conf=0.99)
            t_err, R_err = estimator.get_errs_gt(F, KITTI_K_GT, t_gt, R_gt)
            t_errs[i].append(t_err)
            R_errs[i].append(R_err)
            inliers[i].append(np.sum(mask) / len(mask))
            times[i].append(time)

    return t_errs, R_errs, inliers, times

def evaluate_kitti(path):
    sequences = ["{:02d}".format(x) for x in range(2, 11)]
    pkl_paths = [os.path.join(path, 'extracted_features', '{}.pkl'.format(s)) for s in sequences]
    estimators = [PlanarSixPointGC(), GeneralSevenPointGC(), PlanarTwoAffineGC(), GeneralThreeAffineGC(), Planar8ptEstimator(), OpenCVGeneral8ptEstimator(), OpenCVGeneral7ptEstimator()]

    t_errs = [[] for _ in estimators]
    R_errs = [[] for _ in estimators]
    inliers = [[] for _ in estimators]
    times = [[] for _ in estimators]

    for pkl_path in pkl_paths:
        seq_t_errs, seq_R_errs, seq_inliers, seq_times = evaluate_sequence(pkl_path, estimators)

        print("Errors for {}".format(pkl_path))
        display_errors(estimators, seq_t_errs, seq_R_errs, seq_inliers, seq_times)

if __name__ == '__main__':
    odometry_path = 'D:/Skola/PhD/data/KITTI/odometry/dataset/'
    evaluate_kitti(odometry_path)