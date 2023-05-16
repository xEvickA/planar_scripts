import numpy as np
import time
import poselib
from matplotlib import pyplot as plt
import cv2
from IPython.display import display
from IPython.display import set_matplotlib_formats
import torch
from SuperGluePretrainedNetwork.models.matching import Matching
from triangulate import triangulate
import planar_minimal.utils as utils
from scipy.spatial.transform import Rotation
from planar_minimal.utils import angle, angle_matrix
from os.path import exists
from planar_minimal import tum_dataset

def get_pictures(file, x):
    with open(file) as f:
        lines = f.readlines()
    lines[x] = lines[x].strip()
    return lines[x].split(",")[0], lines[x].split(",")[1], lines[x].split(",")[2:4], lines[x].split(",")[4:7], lines[x].split(",")[7::]

def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

def find_matches(x, file):
    
  im0, im1, timespamps, t, quat = get_pictures(file, x)
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

def write_to_file_cam_F(file_name, F1, F, mask1, mask2, i, start, end, end1):
  with open(file_name, 'a') as f:
   f.write(str(i + 1) + ".")
   f.write("\n6pt:\n")
   
   f.write(('{},' * 0 + '{} \n').format(F1))
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

def write_error(file_name, R, t, Rpt, tpt, time, inliers):
   with open(file_name, 'a') as err:
      alpha = angle_matrix(R, Rpt)
      beta = angle(t, tpt)
      if alpha > 90: alpha = 180 - alpha
      if beta > 90: beta = 180 - beta
      line = ('{},' * 3 + '{} \n').format(alpha, beta, time, inliers)
      err.write(line)


fx = 525.0  # focal length x
fy = 525.0  # focal length y
cx = 319.5  # optical center x
cy = 239.5  # optical center y
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
C1 = {'model': 'PINHOLE', 'params': [fx, fy, cx, cy]}
C2 = {'model': 'PINHOLE', 'params': [fx, fy, cx, cy]}
ransac = dict() 
count = 0


def calcul_for_dataset(diff):
   if diff == 0:
      diff = ''
   num_of_lines = get_num_of_lines(f"dataset/planar{diff}.txt")
   matches = []
   for i in range(num_of_lines):
      if(len(matches) != num_of_lines):
         p, q, t, R = find_matches(i, f"dataset/planar{diff}.txt")
         matches.append([p, q, t, R])
      else:
         p, q, t, R = matches[i]
      if(len(p) < 100):
         with open(f"Results6ptvs7pt{diff}.txt", 'a') as f:
            f.write(str(i + 1) + ". Low number of matches!\n")
         continue
      t = np.array(t, dtype=float).T
      R = np.array(R.as_matrix())

      start = time.time()
      F1, mask1 = poselib.estimate_planar_fundamental_6pt(p, q, ransac, dict(), True)
      end = time.time()
      F2, mask2 = poselib.estimate_planar_fundamental_6pt(p, q, ransac, dict(), False)
      end1 = time.time()
      F3, mask3 = poselib.estimate_fundamental(p, q, ransac, dict(), True)
      end2 = time.time()
      F4, mask4 = poselib.estimate_fundamental(p, q, ransac, dict(), False)
      end3 = time.time()
      time1 = end - start
      time2 = end1 - end
      time3 = end2 - end1
      time4 = end3 - end2
      try:
         R6ptT, t6ptT = Rt_from_fundamental(F1)
         R6ptF, t6ptF = Rt_from_fundamental(F2)
         R7ptT, t7ptT = Rt_from_fundamental(F3)
         R7ptF, t7ptF = Rt_from_fundamental(F4)
         write_error(f"error6ptT{diff}.txt", R, t, R6ptT, t6ptT, time1, mask1['inliers'].count(True))
         write_error(f"error6ptF{diff}.txt", R, t, R6ptF, t6ptF, time2, mask2['inliers'].count(True))
         write_error(f"error7ptT{diff}.txt", R, t, R7ptT, t7ptT, time3, mask3['inliers'].count(True))
         write_error(f"error7ptF{diff}.txt", R, t, R7ptF, t7ptF, time4, mask4['inliers'].count(True))
      except:
         continue

if __name__ == '__main__':
   if not exists("dataset/planar0.txt"):
      tum_dataset.extract_planar_dir("dataset", 0.01, 0)
   calcul_for_dataset(0)
