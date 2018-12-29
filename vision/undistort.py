import sys
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob

# You should replace these 3 lines with the output in calibration step
DIM=(640, 480)
K=np.array([[528.8725348422289, 0.0, 309.32147655398825], [0.0, 529.2444960604813, 239.89341657922114], [0.0, 0.0, 1.0]])
D=np.array([[0.3200057978906918], [1.6776306679257464], [-19.357255163837433], [66.83205502847353]])
def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("original image", img)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)

