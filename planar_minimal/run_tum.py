import os

import cv2
import matlab
import numpy as np
import pyrobustac
from estimators_fundamental import Planar6ptEstimator, General7ptEstimator, PlanarHomography5ptEstimator, \
    Planar2ACEstimator, PlanarGC, GeneralGC, Planar8ptEstimator, GeneralGCAffine, PlanarGCAffine
from scipy.spatial.transform import Rotation
from synth_point_generation import generate_synth_pcs
from ransac import ransac
from utils import angle_matrix, angle
from matplotlib import pyplot as plt


tum_fx = 525.0  # focal length x
tum_fy = 525.0  # focal length y
tum_cx = 319.5  # optical center x
tum_cy = 239.5  # optical center y
tum_K = np.array([[tum_fx, 0, tum_cx], [0, tum_fy, tum_cy], [0, 0, 1]])


def tum_generator(dir):
    planar_filename = os.path.join(dir, 'planar.txt')

    with open(planar_filename) as f:
        content = f.readlines()

    for line in content:
        d = line.split(",")
        img_path_i = d[0]
        img_path_j = d[1]
        t_gt = np.array([float(x) for x in d[4:7]])
        q_gt = np.array([float(x) for x in d[7:]])
        r_gt = Rotation.from_quat(q_gt)
        R_gt = r_gt.as_dcm()

        p1, p2, A = pyrobustac.extractACs(img_path_i, img_path_j)

        # img_j = cv2.imread(img_path_j)
        # for p in p2:
        #     img_j = cv2.circle(img_j, (int(p[0]), int(p[1])), 1, (0, 0, 255), thickness=-1)
        # cv2.imshow("keypoints", img_j)
        # cv2.waitKey(0)

        A = np.reshape(A, [len(A), 2, 2])

        yield p1, p2, A, tum_K, t_gt, R_gt


def evaluate_for_dir(dir, estimators):
    t_errs = [[] for _ in estimators]
    R_errs = [[] for _ in estimators]
    inliers = [[] for _ in estimators]

    for p1, p2, A, K_gt, t_gt, R_gt in tum_generator(dir):
        if len(p1) < 50:
            continue
        Ms_estimated = [ransac(estimator, p1, p2, A=A, K_gt=K_gt, inlier_threshold=0.5, num_iters=50) for estimator in estimators]
        for i, M in enumerate(Ms_estimated):
            t_err, R_err, inliers_single = estimators[i].get_errs(M, K_gt, t_gt, R_gt, p1, p2)
            t_errs[i].append(t_err)
            R_errs[i].append(R_err)
            inliers[i].append(inliers_single)

    return t_errs, R_errs, inliers


def evaluate_tum():
    dirs = ['rgbd_dataset_freiburg1_desk']
    dirs = ['rgbd_dataset_freiburg1_xyz']
    dirs = [os.path.join('dataset', dir) for dir in dirs]

    # eng = matlab.engine.start_matlab()
    # eng.cd(r'./solvers')
    # estimators = [Planar6ptEstimator(eng), Planar2ACEstimator(eng), PlanarHomography5ptEstimator(), General7ptEstimator()]

    # estimators = [PlanarHomography5ptEstimator(), General7ptEstimator()]
    estimators = [PlanarGC(), GeneralGC(), Planar8ptEstimator(), General7ptEstimator()]
    estimators = [PlanarGCAffine(), GeneralGCAffine()]

    t_means = np.empty([len(estimators), len(dirs)])
    R_means = np.empty([len(estimators), len(dirs)])
    i_means = np.empty([len(estimators), len(dirs)])

    t_medians = np.empty([len(estimators), len(dirs)])
    R_medians = np.empty([len(estimators), len(dirs)])
    i_medians = np.empty([len(estimators), len(dirs)])

    nans = np.empty([len(estimators), len(dirs)])

    for j, dir in enumerate(dirs):
        t_errs, R_errs, inliers = evaluate_for_dir(dir, estimators)
        for i in range(len(estimators)):
            t_means[i, j] = np.nanmean(t_errs[i])
            R_means[i, j] = np.nanmean(t_errs[i])
            i_means[i, j] = np.nanmean(inliers[i])

            t_medians[i, j] = np.median(t_errs[i])
            R_medians[i, j] = np.median(R_errs[i])
            i_medians[i, j] = np.median(inliers[i])

            nans[i, j] = np.sum([~np.isnan(x) for x in inliers[i]])/len(t_errs[i])

    for i, estimator in enumerate(estimators):
        print("For estimator: {} Mean errrors t: {}, errors R: {}, inliers: {}".format(estimator.__class__.__name__, t_means[i], R_means[i], i_means[i]))
        print("For estimator: {} Median errrors t: {}, errors R: {}, inliers: {}".format(estimator.__class__.__name__, t_medians[i], R_medians[i], i_medians[i]))
        print("For estimator: {} Failed runs: {}".format(estimator.__class__.__name__, nans[i]))


if __name__ == '__main__':
    evaluate_tum()




