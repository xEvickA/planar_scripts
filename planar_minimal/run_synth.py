import matlab
import numpy as np
from estimators_essential import PlanarRel4ptEstimator, PlanarRel1pt1ACEstimator
from estimators_fundamental import *
from estimators_semi import PlanarSemi5ptReducedEstimator, PlanarSemi5ptEstimator
from synth_point_generation import generate_synth_pcs
from ransac import ransac
from matplotlib import pyplot as plt
from utils import display_errors


def evaluate_for_sigma(estimators, sigma=1.0, repeats=10):
    t_errs = [[] for _ in estimators]
    R_errs = [[] for _ in estimators]
    inliers = [[] for _ in estimators]
    times = [[] for _ in estimators]

    for i in range(repeats):
        p1, p2, A, F_gt, K_gt, t_gt, R_gt = generate_synth_pcs(300, shuffle=0.0, major_plane=0.0, num_planes=10, sigma_pix=sigma)
        for i, estimator in enumerate(estimators):
            M, mask, time = estimator.estimate_timed(p1, p2, A=A, K_gt=K_gt)
            t_err, R_err = estimator.get_errs_gt(M, K_gt, t_gt, R_gt)
            t_errs[i].append(t_err)
            R_errs[i].append(R_err)
            inliers[i].append(np.sum(mask) / len(mask))
            times[i].append(time)
    return t_errs, R_errs, inliers, times


def evaluate_synth():
    # planar_6pt_estimator = Planar6ptEstimator()
    # general_7pt_estimator = General7ptEstimator()

    estimators = [PlanarSixPointGC(), GeneralSevenPointGC(), PlanarTwoAffineGC(), GeneralThreeAffineGC(), Planar8ptEstimator(), OpenCVGeneral8ptEstimator(), OpenCVGeneral7ptEstimator()]

    sigmas = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0])
    # sigmas = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    # sigmas = np.array([0.0, 0.0])

    t_means = np.empty([len(estimators), len(sigmas)])
    R_means = np.empty([len(estimators), len(sigmas)])
    i_means = np.empty([len(estimators), len(sigmas)])
    time_means = np.empty([len(estimators), len(sigmas)])
    nans = np.empty([len(estimators), len(sigmas)])

    for s, sigma in enumerate(sigmas):
        t_errs, R_errs, inliers, times = evaluate_for_sigma(estimators, sigma=sigma, repeats=10)

        display_errors(estimators, t_errs, R_errs, inliers, times)

        for i in range(len(estimators)):
            t_means[i, s] = np.nanmean(t_errs[i])
            R_means[i, s] = np.nanmean(t_errs[i])
            i_means[i, s] = np.nanmean(inliers[i])
            time_means[i, s] = np.nanmean(times[i])
            nans[i, s] = np.sum([~np.isnan(x) for x in inliers[i]])/len(t_errs[i])


    for i, estimator in enumerate(estimators):
        plt.plot(sigmas, t_means[i, :],  label=estimator.__class__.__name__)
    plt.legend(loc='best')
    plt.show()

    for i, estimator in enumerate(estimators):
        plt.plot(sigmas, R_means[i, :],  label=estimator.__class__.__name__)
    plt.legend(loc='best')
    plt.show()

    for i, estimator in enumerate(estimators):
        plt.plot(sigmas, i_means[i, :],  label=estimator.__class__.__name__)
    plt.legend(loc='best')
    plt.show()

    for i, estimator in enumerate(estimators):
        plt.plot(sigmas, time_means[i, :],  label=estimator.__class__.__name__)
    plt.legend(loc='best')
    plt.show()

    for i, estimator in enumerate(estimators):
        plt.plot(sigmas, nans[i, :],  label=estimator.__class__.__name__)
    plt.legend(loc='best')
    plt.show()



if __name__ == '__main__':
    evaluate_synth()




