import numpy as np


def ransac(estimator, p1, p2, inlier_threshold=0.1, num_iters=1000, **kwargs):
    best_inliers = 0
    best_model = None

    for M in estimator.estimate_samples(p1, p2, num_iters, **kwargs):
        if M is None:
            continue
        if best_model is None:
            best_model = M
        # print(estimator.__class__.__name__)
        inliers = estimator.calc_inliers(M, p1, p2, t=inlier_threshold, **kwargs)
        if inliers > best_inliers:
            best_inliers = inliers
            best_model = M
    # print(best_inliers)
    return best_model
