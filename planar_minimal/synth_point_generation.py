import numpy as np
from scipy.spatial.transform import Rotation
from planar_minimal.utils import skew_symmetric


def get_random_n():
    n = np.random.rand(3) - 0.5
    n /= np.linalg.norm(n)
    np1 = np.array([n[1], -n[0], 0])
    np1 /= np.linalg.norm(np1)
    np2 = np.cross(n, np1)
    return n, np1, np2


def get_single_points(num_pts, width, height, focal):
    X = np.random.rand(num_pts, 3)
    X[:, 0] *= 2 * width
    X[:, 1] *= 2 * height
    X[:, 2] *= 3 * focal
    X[:, 2] += 2 * focal

    return X


def compute_homography(X, n, K, R, t):
    d = -np.dot(X, n)
    H = R - t[:, np.newaxis] @ n[np.newaxis, :] / d
    H = K @ H @ np.linalg.inv(K)

    return H

def get_plane_points(num_planes, num_pts, major_plane, K, R, t, width, height, focal):
    Xs = []
    Hs = []
    X_seeds = get_single_points(num_planes, width, height, focal)

    if major_plane > 0.0:
        pvals = np.empty(num_planes)
        pvals[0] = major_plane
        pvals[1:] = np.ones(num_planes - 1) * (1 - major_plane)/(num_planes - 1)
        pts_per_seed = np.random.multinomial(num_pts - num_planes, pvals, size=1)[0]
    else:
        pts_per_seed = np.random.multinomial(num_pts - num_planes, np.ones(num_planes) / num_planes, size=1)[0]
    for i in range(num_planes):
        n, np1, np2 = get_random_n()
        H = compute_homography(X_seeds[i], n, K, R, t)
        Xs.append(X_seeds[i])
        # Hs.append(H)
        X_n = X_seeds[i, :] + focal * 3 * np.random.randn(pts_per_seed[i], 1) * np1[np.newaxis, :] + focal * 3 * np.random.randn(pts_per_seed[i], 1) * np2[np.newaxis, :]
        # print(np.abs(np.linalg.det(np.vstack([X_n[-4:].T, np.ones((1, 4))]))))
        Xs.extend(X_n)
        Hs.extend([H for _ in range(pts_per_seed[i] + 1)])

    return np.array(Xs), np.array(Hs)


def get_affine_correspondences(p1, p2, H):
    if H is None:
        return None

    A = np.empty([len(p1), 2, 2])

    p1 = np.concatenate([p1, np.ones([len(p1), 1])], axis=-1)

    for i in range(len(p1)):
        s = np.dot(H[i, 2, :], p1[i])
        A[i, 0, :] = (H[i, 0, :2] - H[i, 2, :2] * p2[i, 0]) / s
        A[i, 1, :] = (H[i, 1, :2] - H[i, 2, :2] * p2[i, 1]) / s

    return A


def get_random_t_R(theta, focal):
    u = np.random.rand(3) - 0.5
    u /= np.linalg.norm(u)
    R = Rotation.from_rotvec(u * 10 * np.random.randn()).as_matrix()

    p1 = np.array([u[1], -u[0], 0])
    p1 /= np.linalg.norm(p1)
    p2 = np.cross(u, p1)

    t = focal / 2 * np.random.randn() * p1 + focal / 2 * np.random.randn() * p2

    pt = np.cross(t, u)
    pt = np.deg2rad(theta) * pt / np.linalg.norm(pt)
    R_angle = Rotation.from_rotvec(pt).as_matrix()

    t = R_angle @ t

    return R, t


def generate_synth_pcs(num_pts, theta=90.0, num_planes=10, shuffle=0.0, major_plane=0.0, sigma_pix=0.5, focal=300, width=640, height=480, px=320, py=240):
    # init random intrinsics if they are not set
    if focal is None:
        focal = 100 + np.random.rand() * 10
    if width is None:
        width = np.random.rand() * 2000 + 200
    if height is None:
        height = np.random.rand() * 1000 + 200
    if px is None:
        px = width / 2
    if py is None:
        py = height / 2


    motion_n, motion_np1, motion_np2 = get_random_n()

    R, t = get_random_t_R(theta, focal)
    rotation = Rotation.from_matrix(R)

    # print("zeros")
    # print((skew_symmetric(t) @ R - R.T @ skew_symmetric(t)) @ (t - R.T @ t))
    # print((R @ skew_symmetric(R.T @ t) @ t + R.T @ skew_symmetric(t) @ R.T @ t))
    # print((R @ skew_symmetric(R.T @ t) - R.T @ skew_symmetric(t)) @ (t - R.T @ t))
    # print((R.T - R) @ (np.cross(t, R.T @ t)))

    K = np.array([[focal, 0 , px], [0, focal, py], [0, 0, 1]])
    Kinv = np.linalg.inv(K)
    F = Kinv.T @ skew_symmetric(t) @ R @ Kinv


    if num_planes == 0:
        X1 = get_single_points(num_pts, width, height, focal)
        H = None
    else:
        X1, H = get_plane_points(num_planes,num_pts, major_plane, K, R, t, width, height, focal)

    # h = np.linalg.inv(K).T @ rot_vec
    # print("Corresponding horizon: ", h/h[2])
    # hh = H[0].T @ h
    # print("H.T @ h", hh/hh[2])
    # v, w = np.linalg.eig(H[0].T)
    # for vec in w.T:
    #     print(vec / vec[2])

    X2 = rotation.apply(X1) + t

    p1 = (K @ X1.T).T
    p2 = (K @ X2.T).T

    # pp = (H[0] @ p1.T).T
    # print(pp[:, :2] / pp[:, 2, np.newaxis] - p2[:, :2] / p2[:, 2, np.newaxis])

    p1 = p1[:, :2] / p1[:, 2, np.newaxis]
    p2 = p2[:, :2] / p2[:, 2, np.newaxis]

    p1 += np.random.randn(num_pts, 2) * sigma_pix
    p2 += np.random.randn(num_pts, 2) * sigma_pix


    A = get_affine_correspondences(p1, p2, H)

    if shuffle > 0.0:
        num_shuffle = int(np.round(num_pts * shuffle))

        idxs_shuffle = np.random.choice(num_pts, num_shuffle, replace=False)

        p2[idxs_shuffle] = np.random.rand(num_shuffle, 2) * np.array([[width, height]])
        #A[idxs_shuffle] = np.random.rand(num_shuffle, 2, 2)

    return p1, p2, A, F, K, t, R
