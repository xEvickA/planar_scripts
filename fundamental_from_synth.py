import time
import poselib
import planar_minimal.utils as utils
from planar_minimal.synth_point_generation import generate_synth_pcs
from fundamental_from_dataset import Rt_from_fundamental


def write_error(file_name, R, t, Rpt, tpt, time, inliers, outliers):
   with open(file_name, 'a') as err:
      alpha = utils.angle_matrix(R, Rpt)
      beta = utils.angle(t, tpt)
      if alpha > 90: alpha = 180 - alpha
      if beta > 90: beta = 180 - beta
      line = ('{},' * 4 + '{} \n').format(alpha, beta, time, inliers, outliers)
      err.write(line)

def synth(points, theta):
    ransac = dict()
    p1, p2, A, F, K, t, R = generate_synth_pcs(points, theta)

    start = time.time()
    F1, mask1 = poselib.estimate_planar_fundamental_6pt(p1, p2, ransac, dict(), True)
    end = time.time()
    F2, mask2 = poselib.estimate_planar_fundamental_6pt(p1, p2, ransac, dict(), False)
    end1 = time.time()
    F3, mask3 = poselib.estimate_fundamental(p1, p2, ransac, dict(), True)
    end2 = time.time()
    F4, mask4 = poselib.estimate_fundamental(p1, p2, ransac, dict(), False)
    end3 = time.time()
    time1 = end - start
    time2 = end1 - end
    time3 = end2 - end1
    time4 = end3 - end2
    R6ptT, t6ptT = Rt_from_fundamental(F1)
    R6ptF, t6ptF = Rt_from_fundamental(F2)
    R7ptT, t7ptT = Rt_from_fundamental(F3)
    R7ptF, t7ptF = Rt_from_fundamental(F4)
    write_error(f"synth_error6ptT{theta}.txt", R, t, R6ptT, t6ptT, time1, mask1['inliers'].count(True), points)
    write_error(f"synth_error6ptF{theta}.txt", R, t, R6ptF, t6ptF, time2, mask2['inliers'].count(True), points)
    write_error(f"synth_error7ptT{theta}.txt", R, t, R7ptT, t7ptT, time3, mask3['inliers'].count(True), points)
    write_error(f"synth_error7ptF{theta}.txt", R, t, R7ptF, t7ptF, time4, mask4['inliers'].count(True), points)


if __name__ == '__main__':
    for j in range(0, 36, 5):
        for i in range(300):    
            synth(500, j)
