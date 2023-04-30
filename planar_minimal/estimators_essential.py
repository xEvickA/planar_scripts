import cv2
import numpy as np

import matlab.engine
from utils import skew_symmetric, angle, angle_matrix


class RelEstimator():
    def get_errs(self, E, K, t, R, p1, p2, **kwargs):
        R1, R2, tt = cv2.decomposeEssentialMat(E)
        t_err = min(angle(t, tt), angle(t, -tt))
        R_err = min(angle_matrix(R, R1), angle_matrix(R, R2))
        inliers = self.calc_inliers(E, p1, p2, K_gt=K, t=1.0) / len(p1)

        return t_err, R_err, inliers

    def calc_inliers(self, E, p1, p2, K_gt, t=0.1, **kwargs):
        K_inv = np.linalg.inv(K_gt)
        F = K_inv.T @ E @ K_inv
        if p1.shape[1] == 2:
            p1 = np.concatenate([p1, np.ones([len(p1), 1])], axis=1)
            p2 = np.concatenate([p2, np.ones([len(p2), 1])], axis=1)

        lx1 = F.T @ p2.T
        lx2 = F @ p1.T

        # sampson distance
        d = np.sum(p2 * lx2.T, axis=1) / np.sqrt(lx1[0] ** 2 + lx1[1] ** 2 + lx2[0] ** 2 + lx2[1] ** 2)

        return np.sum(np.abs(d) < t)


class PlanarRel4ptEstimator(RelEstimator):
    def __init__(self, eng=None):
        if eng is None:
            self.eng = matlab.engine.start_matlab()
            self.eng.cd(r'./solvers')
        else:
            self.eng = eng

    def estimate_samples(self, p1, p2, num_iters, K_gt=None, **kwargs):
        if p1.shape[1] == 2:
            p1 = np.concatenate([p1, np.ones([len(p1), 1])], axis=1)
            p2 = np.concatenate([p2, np.ones([len(p2), 1])], axis=1)

        if K_gt is None:
            K_inv = np.eye(3)
        else:
            K_inv = np.linalg.inv(K_gt)

        p1n = (K_inv @ p1.T).T
        p2n = (K_inv @ p2.T).T

        p1n = p1n[:, :2] / p1n[:, 2, np.newaxis]
        p2n = p2n[:, :2] / p2n[:, 2, np.newaxis]

        idxs = np.random.choice(len(p1), size=(num_iters, 4))

        for idx in idxs:
            Es = self.estimate(p1n[idx], p2n[idx])
            for E in Es:
                yield E

    def estimate(self, p1, p2):
        # A = np.column_stack([p2[:, 0] * p1[:, 0], p2[:, 0] * p1[:, 1], p2[:, 0] * p1[:, 2],
        #                      p2[:, 1] * p1[:, 0], p2[:, 1] * p1[:, 1], p2[:, 1] * p1[:, 2],
        #                      p2[:, 2] * p1[:, 0], p2[:, 2] * p1[:, 1], p2[:, 2] * p1[:, 2]])

        v1 = p1[:, 1]
        u1 = p1[:, 0]
        v2 = p2[:, 1]
        u2 = p2[:, 0]

        A = np.column_stack([u1 * u2, v1 * u2, u2, u1 * v2, v1 * v2, v2, u1, v1, np.ones(len(p1))])

        U, S, V = np.linalg.svd(A)

        E1 = V[-1]
        E2 = V[-2]
        E3 = V[-3]
        E4 = V[-4]
        E5 = V[-5]

        E1m = matlab.double(E1.tolist())
        E2m = matlab.double(E2.tolist())
        E3m = matlab.double(E3.tolist())
        E4m = matlab.double(E4.tolist())
        E5m = matlab.double(E5.tolist())

        a, b, c, d = self.eng.solver_planar_rel_4pt(E1m, E2m, E3m, E4m, E5m, nargout=4)

        a = np.asarray(a)
        b = np.asarray(b)
        c = np.asarray(c)
        d = np.asarray(d)

        if len(a.shape) == 0:
            E = a * E1 + b * E2 + c * E3 + d * E4 + E5
            E = np.reshape(E, [3, 3])
            return [E]
        else:
            Es = []
            for i in range(a.shape[1]):
                E = a[0, i]*E1 + b[0, i] * E2 + c[0, i] * E3 + d[0, i] * E4 + E5
                E = np.reshape(E, [3, 3])
                Es.append(E)
            return Es


class PlanarRel1pt1ACEstimator(RelEstimator):
    def __init__(self, eng=None):
        if eng is None:
            self.eng = matlab.engine.start_matlab()
            self.eng.cd(r'./solvers')
        else:
            self.eng = eng

    def estimate_samples(self, p1, p2, num_iters, A, K_gt=None, **kwargs):
        if p1.shape[1] == 2:
            p1 = np.concatenate([p1, np.ones([len(p1), 1])], axis=1)
            p2 = np.concatenate([p2, np.ones([len(p2), 1])], axis=1)

        if K_gt is None:
            K_inv = np.eye(3)
        else:
            K_inv = np.linalg.inv(K_gt)

        p1n = (K_inv @ p1.T).T
        p2n = (K_inv @ p2.T).T

        p1n = p1n[:, :2] / p1n[:, 2, np.newaxis]
        p2n = p2n[:, :2] / p2n[:, 2, np.newaxis]

        idxs = np.random.choice(len(p1), size=(num_iters, 2))

        for idx in idxs:
            Es = self.estimate(p1n[idx], p2n[idx], A[idx])
            for E in Es:
                yield E

    def estimate(self, p1, p2, A):
        # A = np.column_stack([p2[:, 0] * p1[:, 0], p2[:, 0] * p1[:, 1], p2[:, 0] * p1[:, 2],
        #                      p2[:, 1] * p1[:, 0], p2[:, 1] * p1[:, 1], p2[:, 1] * p1[:, 2],
        #                      p2[:, 2] * p1[:, 0], p2[:, 2] * p1[:, 1], p2[:, 2] * p1[:, 2]])

        v1 = p1[:, 1]
        u1 = p1[:, 0]
        v2 = p2[:, 1]
        u2 = p2[:, 0]
        a1 = A[:, 0, 0]
        a2 = A[:, 0, 1]
        a3 = A[:, 1, 0]
        a4 = A[:, 1, 1]

        PG = np.column_stack([u1 * u2, v1 * u2, u2, u1 * v2, v1 * v2, v2, u1, v1, np.ones(len(p1))])
        AG1 = np.column_stack([u2 + a1 * u1, a1 * v1, a1, v2 + a3 * u1, a3 * v1, a3, np.ones(len(p1)), np.zeros(len(p1)), np.zeros(len(p1))])
        AG2 = np.column_stack([a2 * u1, u2 + a2 * v1, a2, a4 * u1, v2 + a4 * v1, a4, np.zeros(len(p1)), np.ones(len(p1)), np.zeros(len(p1))])

        G = np.row_stack([PG, AG1[0, :], AG2[0, :]])

        U, S, V = np.linalg.svd(G)

        E1 = V[-1]
        E2 = V[-2]
        E3 = V[-3]
        E4 = V[-4]
        E5 = V[-5]

        E1m = matlab.double(E1.tolist())
        E2m = matlab.double(E2.tolist())
        E3m = matlab.double(E3.tolist())
        E4m = matlab.double(E4.tolist())
        E5m = matlab.double(E5.tolist())

        a, b, c, d = self.eng.solver_planar_rel_4pt(E1m, E2m, E3m, E4m, E5m, nargout=4)

        a = np.asarray(a)
        b = np.asarray(b)
        c = np.asarray(c)
        d = np.asarray(d)

        if len(a.shape) == 0:
            E = a * E1 + b * E2 + c * E3 + d * E4 + E5
            E = np.reshape(E, [3, 3])
            return [E]
        else:
            Es = []
            for i in range(a.shape[1]):
                E = a[0, i] * E1 + b[0, i] * E2 + c[0, i] * E3 + d[0, i] * E4 + E5
                E = np.reshape(E, [3, 3])
                Es.append(E)
            return Es
