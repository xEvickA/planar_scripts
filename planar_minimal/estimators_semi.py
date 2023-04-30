import cv2
import matlab
import numpy as np
from utils import angle_matrix, angle, focal_from_fundamental


class FundamentalSemiEstimator():
    def get_errs(self, F, K_gt, t, R, p1, p2, **kwargs):
        f = focal_from_fundamental(F, K_gt)
        if f is None:
            return np.NaN, np.NaN, np.NaN
        K = K_gt.astype(np.float)
        K[0, 0] = f
        K[1, 1] = f

        E = K.T @ F @ K
        R1, R2, tt = cv2.decomposeEssentialMat(E)
        t_err = min(angle(t, tt), angle(t, -tt))
        R_err = min(angle_matrix(R, R1), angle_matrix(R, R2))
        inliers = self.calc_inliers(F, p1, p2, t=1.0) / len(p1)

        return t_err, R_err, inliers

    def calc_inliers(self, F, p1, p2, t=0.1, **kwargs):
        if p1.shape[1] == 2:
            p1 = np.concatenate([p1, np.ones([len(p1), 1])], axis=1)
            p2 = np.concatenate([p2, np.ones([len(p2), 1])], axis=1)

        lx1 = F.T @ p2.T
        lx2 = F @ p1.T

        # sampson distance
        d = np.sum(p2 * lx2.T, axis=1) / np.sqrt(lx1[0] ** 2 + lx1[1] ** 2 + lx2[0] ** 2 + lx2[1] ** 2)

        return np.sum(np.abs(d) < t)


class PlanarSemi5ptEstimator(FundamentalSemiEstimator):
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

        px = K_gt[0, 2]
        py = K_gt[1, 2]

        T = np.array([[1, 0, -px], [0, 1, -py], [0, 0, 1]])

        p1 = (T @ p1.T).T
        p2 = (T @ p2.T).T

        idxs = np.random.choice(len(p1), size=(num_iters, 5))

        for idx in idxs:
            Fs = self.estimate(p1[idx], p2[idx])
            for F in Fs:
                yield T.T @ F @ T

    def estimate(self, p1, p2):
        A = np.column_stack([p2[:, 0] * p1[:, 0], p2[:, 0] * p1[:, 1], p2[:, 0] * p1[:, 2],
                             p2[:, 1] * p1[:, 0], p2[:, 1] * p1[:, 1], p2[:, 1] * p1[:, 2],
                             p2[:, 2] * p1[:, 0], p2[:, 2] * p1[:, 1], p2[:, 2] * p1[:, 2]])

        # v1 = p1[:, 1]
        # u1 = p1[:, 0]
        # v2 = p2[:, 1]
        # u2 = p2[:, 0]
        #
        # A = np.column_stack([u1 * u2, v1 * u2, u2, u1 * v2, v1 * v2, v2, u1, v1, np.ones(len(p1))])

        U, S, V = np.linalg.svd(A)

        F1 = V[-1]
        F2 = V[-2]
        F3 = V[-3]
        F4 = V[-4]

        Fm = []
        Fm.extend(F1.tolist())
        Fm.extend(F2.tolist())
        Fm.extend(F3.tolist())
        Fm.extend(F4.tolist())

        Fm = matlab.double(Fm)

        sol = self.eng.solver_semi_5pt_planar(Fm, nargout=1)

        sol = np.array(sol)

        if len(sol.shape) == 1:
            if np.isreal(sol).all():
                F = sol[0] * F1 + sol[1] * F2 + sol[2] * F3 + F4
                F = np.real(np.reshape(F, [3, 3]))
                return [F]
            else:
                return []
        else:
            Fs = []
            for i in range(sol.shape[1]):
                F = sol[0, i] * F1 + sol[1, i] * F2 + sol[2, i] * F3 + F4
                F = np.real(np.reshape(F, [3, 3]))
                Fs.append(F)
            return Fs


class PlanarSemi5ptReducedEstimator(FundamentalSemiEstimator):
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

        px = K_gt[0, 2]
        py = K_gt[1, 2]

        T = np.array([[1e-6, 0, - 1e-6 * px], [0, 1e-6, - 1e-6 * py], [0, 0, 1]])

        p1 = (T @ p1.T).T
        p2 = (T @ p2.T).T

        idxs = np.random.choice(len(p1), size=(num_iters, 5))

        for idx in idxs:
            Fs = self.estimate(p1[idx], p2[idx])
            for F in Fs:
                yield T.T @ F @ T

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

        F1 = V[-1]
        F2 = V[-2]
        F3 = V[-3]
        F4 = V[-4]

        Fm = []
        Fm.extend(F1.tolist())
        Fm.extend(F2.tolist())
        Fm.extend(F3.tolist())
        Fm.extend(F4.tolist())

        Fm = matlab.double(Fm)

        sol = self.eng.solver_semi_5pt_planar_red(Fm, nargout=1)

        sol = np.array(sol)

        if len(sol.shape) == 1:
            if np.isreal(sol).all():
                F = sol[0] * F1 + sol[1] * F2 + sol[2] * F3 + F4
                F = np.real(np.reshape(F, [3, 3]))
                return [F]
            else:
                return []
        else:
            Fs = []
            for i in range(sol.shape[1]):
                F = sol[0, i] * F1 + sol[1, i] * F2 + sol[2, i] * F3 + F4
                F = np.real(np.reshape(F, [3, 3]))
                Fs.append(F)
            return Fs
