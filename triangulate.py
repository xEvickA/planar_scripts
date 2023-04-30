import numpy as np
import cv2

def triangulate(K, p, q):
  # params:
  # K - 3x3 matrix with intrinsic camera parameters
  # p - Nx2 array with image points from the first camera
  # q - Nx2 array with image points from the second cameras
  # returns:
  # R - 3x3 rotation matrix as numpy array
  # t - translation vector of shape (3,)
  # X - Nx3 array with each row representing a 3D reconstructed point
  # p - Nx2 array with image points from the first camera which have an associated 3D point
  # q - Nx2 array with image points from the second camera which have an associated 3D point
  
  E, mask = cv2.findEssentialMat(p, q)
  mask = np.squeeze(mask)
  good, R, t, maskk = cv2.recoverPose(E, p[mask == 1], q[mask == 1], K)
  
  maskk = np.squeeze(maskk)
  
  p = p[mask == 1]
  q = q[mask == 1]
  P1 = K @ np.concatenate([R, t], axis=1)
  P2 = K @ np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)

  X = cv2.triangulatePoints(P2, P1, p[maskk == 255].T, q[maskk == 255].T).T

  X = X[:, :3] / X[:, 3, np.newaxis]
  
  # X = X.T
  # X /= X[:,3, None]

  p = p[maskk==255]
  q = q[maskk==255]

  return R, t.ravel(), X, p, q

# p = kp0[matches != -1]
# q = kp1[matches[matches != -1]]
# R, t, X, p, q = triangulate(gt_K, p, q)  

# colors = [img1[int(y), int(x), ::-1] for x, y in p]
# plot_interactive_pointcloud(X, colors)