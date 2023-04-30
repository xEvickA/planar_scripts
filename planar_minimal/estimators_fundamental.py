import time
import warnings

import cv2
import numpy as np

import pygcransac

import pyrobustac
from utils import skew_symmetric, angle, angle_matrix, focal_from_fundamental, prod_matrix_l


class FundamentalEstimator():
    def get_errs_gt(self, F, K_gt, t_gt, R_gt, **kwargs):
        if F is None:
            return np.NaN, np.NaN, np.NaN
        E = K_gt.T @ F @ K_gt
        R1, R2, tt = cv2.decomposeEssentialMat(E)
        t_err = min(angle(t_gt, tt), angle(t_gt, -tt))
        R_err = min(angle_matrix(R_gt, R1), angle_matrix(R_gt, R2))

        return t_err, R_err

    def calc_inliers(self, F, p1, p2, t=0.1, **kwargs):
        if p1.shape[1] == 2:
            p1 = np.concatenate([p1, np.ones([len(p1), 1])], axis=1)
            p2 = np.concatenate([p2, np.ones([len(p2), 1])], axis=1)
        lx1 = F.T @ p2.T
        lx2 = F @ p1.T

        # sampson distance
        d = np.sum(p2 * lx2.T, axis=1) / np.sqrt(lx1[0] ** 2 + lx1[1] ** 2 + lx2[0] ** 2 + lx2[1] ** 2)

        return np.abs(d) < t

    def estimate_timed(self, *args, **kwargs):
        start = time.time()
        F, mask = self.estimate(*args, **kwargs)
        end = time.time()
        return F, mask, end - start


class PlanarSixPointGC(FundamentalEstimator):
    def estimate(self, p1, p2, A=None, h1=360, w1=480, h2=360, w2=480, threshold=1.0, conf=0.99, spatial_coherence_weight=0.975, max_iters=100000, **kwargs):
        F, mask = pygcransac.findPlanarFundamentalMatrix(np.ascontiguousarray(p1), np.ascontiguousarray(p2), h1, w1, h2, w2, threshold, conf, spatial_coherence_weight, max_iters)
        return F, mask


class GeneralSevenPointGC(FundamentalEstimator):
    def estimate(self, p1, p2, A=None, h1=360, w1=480, h2=360, w2=480, threshold=1.0, conf=0.99, spatial_coherence_weight=0.975, max_iters=100000, **kwargs):
        F, mask = pygcransac.findFundamentalMatrix(np.ascontiguousarray(p1), np.ascontiguousarray(p2), h1, w1, h2, w2, threshold, conf, spatial_coherence_weight, max_iters)
        return F, mask


class PlanarTwoAffineGC(FundamentalEstimator):
    def estimate(self, p1, p2, A=None, h1=360, w1=480, h2=360, w2=480, threshold=1.0, conf=0.99, spatial_coherence_weight=0.975, max_iters=100000, **kwargs):
        if A is None:
            return None, None
        if len(A.shape) == 3:
            A = np.reshape(A, [A.shape[0], 4])
        F, mask = pyrobustac.findPlanarFundamentalMat(np.ascontiguousarray(p1), np.ascontiguousarray(p2), np.ascontiguousarray(A), h1, w1, h2, w2, threshold, conf, spatial_coherence_weight, max_iters)
        return F, mask


class GeneralThreeAffineGC(FundamentalEstimator):
    def estimate(self, p1, p2, A=None, h1=360, w1=480, h2=360, w2=480, threshold=1.0, conf=0.99, spatial_coherence_weight=0.975, max_iters=100000, **kwargs):
        if A is None:
            return None, None
        if len(A.shape) == 3:
            A = np.reshape(A, [A.shape[0], 4])
        F, mask = pyrobustac.findFundamentalMat(np.ascontiguousarray(p1), np.ascontiguousarray(p2), np.ascontiguousarray(A), h1, w1, h2, w2, threshold, conf, spatial_coherence_weight, max_iters)
        return F, mask


class Planar8ptEstimator(FundamentalEstimator):
    def estimate(self, p1, p2, threshold=1.0, **kwargs):
        v1 = p1[:, 1]
        u1 = p1[:, 0]
        v2 = p2[:, 1]
        u2 = p2[:, 0]

        A = np.column_stack([u1 * u2, v1 * u2, u2, u1 * v2, v1 * v2, v2, u1, v1, np.ones(len(p1))])

        U, S, V = np.linalg.svd(A)
        F = np.reshape(V[-1, :], [3,3])

        U, S, V = np.linalg.svd(F)
        ep = U[:, 2]
        e = V[2, :]

        M_l = prod_matrix_l(e, ep)
        P = np.linalg.pinv(M_l)
        l = P @ np.reshape(F, [9, 1])[:, 0]

        FF = skew_symmetric(ep) @ skew_symmetric(l) @ skew_symmetric(e)

        mask = self.calc_inliers(F, p1, p2, threshold)

        return FF, mask


class OpenCVGeneral8ptEstimator(FundamentalEstimator):
    def estimate(self, p1, p2, threshold=1.0, conf=0.99, max_iters=10000, **kwargs):
        F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_8POINT, threshold, conf, max_iters)
        return F, mask


class OpenCVGeneral7ptEstimator(FundamentalEstimator):
    def estimate(self, p1, p2, threshold=1.0, conf=0.99, max_iters=10000, **kwargs):
        F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, threshold, conf, max_iters)
        return F, mask