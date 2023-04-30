import numpy as np


def angle_matrix(R1, R2):
    # cos = (np.trace(R1.T @ R2) - 1) / 2
    # cos = max(cos, -1.0)
    # cos = min(cos, 1.0)
    # return np.rad2deg(np.arccos(cos))
    return np.rad2deg(np.real(np.arccos((np.trace(R1.T @ R2) - 1) / 2 + 0.0j)))

def angle(t1, t2):
    return np.rad2deg(np.real(np.arccos(np.dot(t1, t2) / (np.linalg.norm(t1) * np.linalg.norm(t2)) + 0j)))

def skew_symmetric(vec):
    return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])


def display_errors(estimators, t_errs, R_errs, inliers, times):
    for i in range(len(estimators)):
        print('*'*40)
        print(estimators[i].__class__.__name__)
        print("Mean errrors t: {}, errors R: {}, inliers: {}, time: {}"
              .format(np.nanmean(t_errs[i]), np.nanmean(R_errs[i]), np.nanmean(inliers[i]), np.nanmean(times[i])))
        print("Median errrors t: {}, errors R: {}, inliers: {}, time: {}"
              .format(np.nanmedian(t_errs[i]), np.nanmedian(R_errs[i]),np.nanmedian(inliers[i]), np.nanmedian(times[i])))
        print("Failed runs: {}".format(np.sum([np.isnan(x) for x in inliers[i]]) / len(t_errs[i]), np.nanmean([times]) ))

def prod_matrix_l(e, ep):
    return np.array([[-e[1] * ep[2] + e[2] * ep[1], 0, 0],
        [e[0] * ep[2], e[2] * ep[1], e[2] * ep[2]],
        [-e[0] * ep[1], -e[1] * ep[1], -e[1] * ep[2]],
        [-e[2] * ep[0], -e[1] * ep[2], -e[2] * ep[2]],
        [0, e[0] * ep[2] - e[2] * ep[0], 0],
        [e[0] * ep[0], e[1] * ep[0], e[0] * ep[2]],
        [e[1] * ep[0], e[1] * ep[1], e[2] * ep[1]],
        [-e[0] * ep[0], -e[0] * ep[1], -e[2] * ep[0]],
        [0, 0, -e[0] * ep[1] + e[1] * ep[0]]])


def focal_from_fundamental(F, K_gt):
    pp = K_gt[:, 2, np.newaxis]

    U, S, V = np.linalg.svd(F)
    e = V[2, :]
    ex = skew_symmetric(e)
    II = np.diag([1.0, 1.0, 0])
    f2 = -(pp.T @ ex @ II @ F @ pp) * (pp.T @ F.T @ pp) / (pp.T @ ex @ II @ F.T @ II @ F @ pp)

    # f2 = F[2,2] * (F[0, 0] * F[1, 2] * F[2, 0] + F[0, 1] * F[1, 2] * F[2 ,1]
    #                - F[0, 2] * F[1, 0] * F[2, 0] - F[0, 2] * F[1, 1] * F[2, 1])/\
    #      (F[0, 0]**2 * F[0, 2] * F[1, 2] - F[0, 0] * F[0, 2]**2 * F[1, 0]
    #       + F[0, 0] * F[1, 0] * F[1, 2]**2 + F[0, 1]**2 * F[0, 2] * F[1, 2]
    #       - F[0, 1] * F[0, 2]**2 * F[1, 1] + F[0, 1] * F[1, 1] * F[1, 2]**2
    #       - F[0, 2] * F[1, 0]**2 * F[1, 2] - F[0, 2] * F[1, 1]**2 * F[1, 2])

    # T = np.array([[1, 0, pp[0]], [0, 1, pp[1]], [0, 0, 1]], dtype=np.float)
    # G = T.T @ F @ T
    #
    # F0 = np.diag([1.0, 1.0, 1.0])
    # G = F0 @ G @ F0
    # G = G / np.linalg.norm(G, ord='fro')
    #
    # U, S, V = np.linalg.svd(G)
    #
    # a = S[0]
    # b = S[1]
    #
    # u13 = U[2, 0]
    # u23 = U[2, 1]
    #
    # v13 = V[2, 0]
    # v23 = V[2, 1]
    #
    # la1 = a * u13 * u23 * (1 - v13 ** 2) + b * v13 * v23 * (1 - u23 ** 2)
    # lb1 = u23 * v13 * (a * u13 * v13 + b * u23 * v23)
    #
    # la2 = a * v13 * v23 * (1 - u13 ** 2) + b * u13 * u23 * (1 - v23 ** 2)
    # lb2 = u13 * v23 * (a * u13 * v13 + b * u23 * v23)
    #
    # # a * f**4  + b * f**2 + c = 0
    #
    # qa = a ** 2 * (1 - u13 ** 2) * (1 - v13 ** 2) \
    #      - b ** 2 * (1 - u23 ** 2) * (1 - v23 ** 2)
    #
    # qb = a ** 2 * (u13 ** 2 + v13 ** 2 - 2 * u13 ** 2 * v13 ** 2) \
    #      - b ** 2 * (u23 ** 2 + v23 ** 2 - 2 * u23 ** 2 * v23 ** 2)
    #
    # qc = a ** 2 * u13 ** 2 * v13 ** 2 \
    #      - b ** 2 * u23 ** 2 * v23 ** 2
    #
    # # f1 = 2 * qc / (-qb + np.sqrt(qb**2 - 4*qa*qc))
    # # f2 = 2 * qc / (-qb - np.sqrt(qb**2 - 4*qa*qc))
    #
    # f1, f2 = np.sqrt(np.abs(np.roots([qa, qb, qc])))
    #
    # # print("focal 1: {}".format(f1 * f0))
    # # print(qa * f1**4 + qb * f1**2 + qc)
    # # print(la1 * f1**2 + lb1)
    # # print(la2 * f1**2 + lb2)
    # #
    # # print("focal 2: {}".format(f2 * f0))
    # # print(qa * f2**4  + qb * f2**2 + qc)
    # # print(la1 * f2**2 + lb1)
    # # print(la2 * f2**2 + lb2)


    if f2 < 0.0:
        return None
    else:
        return np.sqrt(f2)



