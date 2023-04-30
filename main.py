import numpy as np
import time
import poselib
import math
from matplotlib import pyplot as plt
import cv2
from IPython.display import display
from IPython.display import set_matplotlib_formats
import torch
from SuperGluePretrainedNetwork.models.matching import Matching
from triangulate import triangulate
import planar_minimal.utils as utils
from scipy.spatial.transform import Rotation

def get_pictures(x):
    with open("dataset/planar.txt") as f:
        lines = f.readlines()
    lines[x] = lines[x].strip()
    return lines[x].split(",")[0], lines[x].split(",")[1], lines[x].split(",")[2:4], lines[x].split(",")[4:7], lines[x].split(",")[7::]

def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

def find_matches(x):
    
  im0, im1, timespamps, t, quat = get_pictures(x)
  R = Rotation.from_quat(quat)

  matching = Matching().eval()

  image0 = cv2.imread(im0, cv2.IMREAD_GRAYSCALE)
  inp0 = frame2tensor(image0, 'cpu')

  image1 = cv2.imread(im1, cv2.IMREAD_GRAYSCALE)
  inp1 = frame2tensor(image1, 'cpu')

  pred = matching({'image0': inp0, 'image1': inp1})
  pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
  kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
  matches, conf = pred['matches0'], pred['matching_scores0']
  #timer.update('matcher')
  out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                  'matches': matches, 'match_confidence': conf}

#   mconf = conf[matches != -1]
  p = kpts0[matches != -1]
  q = kpts1[matches[matches != -1]]
  return p, q, t, R

def get_num_of_lines(file_name):
  with open(file_name) as f:
        lines = f.readlines()
        return len(lines)

def write_to_file_cam_F(file_name, cam, F, mask1, mask2, i, start, end, end1):
  with open(file_name, 'a') as f:
   f.write(str(i + 1) + ".")
   f.write("\n6pt:\n")
   
   f.write(('{},' * 0 + '{} \n').format(cam.Rt))
   f.write("num inliers: " + str(mask1) + "\n")
   f.write(str(end - start) + "\n")

   f.write("7pt:\n")
   f.write(('{},' * 0 + '{} \n').format(F))
   f.write("num inliers: " + str(mask2) + "\n")
   f.write(str(end1 - end) + "\n")

def Rt_from_fundamental(F):
   E = K.T @ F @ K
   u, s, v = np.linalg.svd(E)
   W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
   if(np.linalg.det(u @ W @ v) < 0):
      W = -W
   R = u @ W @ v
   t = u[:, 2]
   return R, t

def angle_rowwise(A, B):
    p1 = np.einsum('ij,ij->i',A,B)
    p2 = np.linalg.norm(A,axis=1)
    p3 = np.linalg.norm(B,axis=1)
    p4 = p1 / (p2*p3)
    return np.arccos(np.clip(p4,-1.0,1.0))

def angle_vector(v, u):
   return np.arccos(np.clip(np.dot(v, u),-1.0, 1.0))

def angle_matrix(A, B):
   return np.arccos(np.sum(A * B) / (math.sqrt(np.sum(A * A)) * math.sqrt(np.sum(B * B))))

def write_error(file_name, R, t, Rpt, tpt):
   with open(file_name, 'a') as err:
      line = ('{},' * 1 + '{} \n').format(angle_matrix(R, Rpt), angle_vector(t, tpt))
      err.write(line)


fx = 525.0  # focal length x
fy = 525.0  # focal length y
cx = 319.5  # optical center x
cy = 239.5  # optical center y
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
C1 = {'model': 'PINHOLE', 'params': [fx, fy, cx, cy]}
C2 = {'model': 'PINHOLE', 'params': [fx, fy, cx, cy]}
ransac = dict() #{"max_iterations":1000}
count = 0

num_of_lines = get_num_of_lines("dataset/planar.txt")
count = 0

# tu odporúčam zmenit počet, ak nechceš aby zbehli všetky dvojice
for i in range(num_of_lines):
   p, q, t, R = find_matches(i)
   t = np.array(t, dtype=float).T
   R = np.array(R.as_matrix())
   cam = None
   F = None

   start = time.time()
   cam, mask1 = poselib.estimate_relative_planar_pose_6pt(p, q, C1, C2, ransac, dict())
   end = time.time()
   F, mask2 = poselib.estimate_fundamental(p, q, ransac, dict())
   end1 = time.time()

   R7pt, t7pt = Rt_from_fundamental(F)

   write_to_file_cam_F("results6ptvs7pt.txt", cam, F, mask1['inliers'].count(True), mask2['inliers'].count(True), i, start, end, end1)
   write_error("error6pt.txt", R, t, cam.R, cam.t)
   write_error("error7pt.txt", R, t, R7pt, t7pt)

   if (end - start < end1 - end): count += 1

# zapis kolko 6pt bolo rýchlejsích ako 7pt
with open("time.txt", 'a') as time:
  time.write("In " + str(count) + "/" + str(i + 1) + " cases was 6pt faster than 7pt. \n")

