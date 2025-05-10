import cv2
import json
import numpy as np
import pandas as pd
import glob
import argparse
import os
import time
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from numpy.core.records import array
from numpy.distutils.system_info import x11_info
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from traits.trait_types import self
from pathlib import Path
from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer, make_matching_plot_fast, frame2tensor)
from scipy.interpolate import splprep, splev

torch.set_grad_enabled(False)
mpl.use('TkAgg')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def cart2hom(arr):
    """ Convert catesian to homogenous points by appending a row of 1s
    :param arr: array of shape (num_dimension x num_points)
    :returns: array of shape ((num_dimension+1) x num_points) 
    """
    if arr.ndim == 1:
        return np.hstack([arr, 1])
    return np.asarray(np.vstack([arr, np.ones(arr.shape[1])]))


def compute_P_from_essential(E):
    """ Compute the second camera matrix (assuming P1 = [I 0])
        from an essential matrix. E = [t]R
    :returns: list of 4 possible camera matrices.
    """
    U, S, V = np.linalg.svd(E)

    # Ensure rotation matrix are right-handed with positive determinant
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    # create 4 possible camera matrices (Hartley p 258)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
           np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
           np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
           np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    return P2s


def correspondence_matrix(p1, p2):
    p1x, p1y = p1[:2]
    p2x, p2y = p2[:2]

    return np.array([
        p1x * p2x, p1x * p2y, p1x,
        p1y * p2x, p1y * p2y, p1y,
        p2x, p2y, np.ones(len(p1x))
    ]).T

    return np.array([
        p2x * p1x, p2x * p1y, p2x,
        p2y * p1x, p2y * p1y, p2y,
        p1x, p1y, np.ones(len(p1x))
    ]).T


def scale_and_translate_points(points):
    """ Scale and translate image points so that centroid of the points
        are at the origin and avg distance to the origin is equal to sqrt(2).
    :param points: array of homogenous point (3 x n)
    :returns: array of same input shape and its normalization matrix
    """
    x = points[0]
    y = points[1]
    center = points.mean(axis=1)  # mean of each row
    cx = x - center[0]  # center the points
    cy = y - center[1]
    dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
    scale = np.sqrt(2) / dist.mean()
    norm3d = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0, 0, 1]
    ])

    return np.dot(norm3d, points), norm3d


def compute_image_to_image_matrix(x1, x2, compute_essential=False):
    """ Compute the fundamental or essential matrix from corresponding points
        (x1, x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
    """
    A = correspondence_matrix(x1, x2)
    # compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F. Make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    if compute_essential:
        S = [1, 1, 0]  # Force rank 2 and equal eigenvalues
    F = np.dot(U, np.dot(np.diag(S), V))

    return F


def compute_normalized_image_to_image_matrix(p1, p2, compute_essential=False):
    """ Computes the fundamental or essential matrix from corresponding points
        using the normalized 8 point algorithm.
    :input p1, p2: corresponding points with shape 3 x n
    :returns: fundamental or essential matrix with shape 3 x 3
    """
    n = p1.shape[1]
    if p2.shape[1] != n:
        raise ValueError('Number of points do not match.')

    # preprocess image coordinates
    p1n, T1 = scale_and_translate_points(p1)
    p2n, T2 = scale_and_translate_points(p2)

    # compute F or E with the coordinates
    F = compute_image_to_image_matrix(p1n, p2n, compute_essential)

    # reverse preprocessing of coordinates
    # We know that P1' E P2 = 0
    F = np.dot(T1.T, np.dot(F, T2))

    return F / F[2, 2]


def compute_fundamental_normalized(p1, p2):
    return compute_normalized_image_to_image_matrix(p1, p2)


def compute_essential_normalized(p1, p2):
    return compute_normalized_image_to_image_matrix(p1, p2, compute_essential=True)


def skew(x):
    """ Create a skew symmetric matrix *A* from a 3d vector *x*.
        Property: np.cross(A, v) == np.dot(x, v)
    :param x: 3d vector
    :returns: 3 x 3 skew symmetric matrix from *x*
    """
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])


def reconstruct_one_point(pt1, pt2, m1, m2):
    """
        pt1 and m1 * X are parallel and cross product = 0
        pt1 x m1 * X  =  pt2 x m2 * X  =  0
    """
    A = np.vstack([
        np.dot(skew(pt1), m1),
        np.dot(skew(pt2), m2)
    ])
    U, S, V = np.linalg.svd(A)
    P = np.ravel(V[-1, :4])

    return P / P[3]


def linear_triangulation(p1, p2, m1, m2):
    """
    Linear triangulation (Hartley ch 12.2 pg 312) to find the 3D point X
    where p1 = m1 * X and p2 = m2 * X. Solve AX = 0.
    :param p1, p2: 2D points in homo. or catesian coordinates. Shape (3 x n)
    :param m1, m2: Camera matrices associated with p1 and p2. Shape (3 x 4)
    :returns: 4 x n homogenous 3d triangulated points
    """
    num_points = p1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        A = np.asarray([
            (p1[0, i] * m1[2, :] - m1[0, :]),
            (p1[1, i] * m1[2, :] - m1[1, :]),
            (p2[0, i] * m2[2, :] - m2[0, :]),
            (p2[1, i] * m2[2, :] - m2[1, :])
        ])

        _, _, V = np.linalg.svd(A)
        X = V[-1, :4]
        res[:, i] = X / X[3]

    return res


def writetofile(dict, path):
    for index, item in enumerate(dict):
        dict[item] = np.array(dict[item])
        dict[item] = dict[item].tolist()
    js = json.dumps(dict)
    with open(path, 'w') as f:
        f.write(js)
        print("参数已成功保存到文件")


def readfromfile(path):
    with open(path, 'r') as f:
        js = f.read()
        mydict = json.loads(js)
    print("参数读取成功")
    return mydict


def camera_calibration(SaveParamPath, ImagePath):
    # 循环中断
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 棋盘格尺寸
    row = 8
    column = 5
    objpoint = np.zeros((row * column, 3), np.float32)
    objpoint[:, :2] = np.mgrid[0:row, 0:column].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    batch_images = glob.glob(ImagePath + '/*.png')
    for i, fname in enumerate(batch_images):
        img = cv2.imread(batch_images[i])
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find chess board corners
        ret, corners = cv2.findChessboardCorners(imgGray, (row, column), None)
        # ret, corners = cv2.goodFeaturesToTrack(imgGray, 5000, 0.01, 10)
        # if found, add object points, image points (after refining them)
        # if ret:
        #     objpoints.append(objpoint)
        #     corners2 = cv2.cornerSubPix(imgGray, corners, (11, 11), (-1, -1), criteria)
        #     imgpoints.append(corners2)
        #     # Draw and display the corners
        #     img = cv2.drawChessboardCorners(img, (row, column), corners2, ret)
        #     cv2.imshow('FoundCorners', img)
        #     cv2.waitKey(500)
        #     cv2.imwrite('Checkerboard_Image0/Temp_JPG/Temp_' + str(i) + '.png', img)
        if ret:
            # cv2.cornerSubPix(gray_img, cp_img, (11,11), (-1,-1), criteria)
            objpoints.append(objpoint)
            imgpoints.append(corners)
            # view the corners
            cv2.drawChessboardCorners(img, (row, column), corners, ret)
            # cv2.imshow('FoundCorners', img)
            # cv2.waitKey(500)

    print("成功提取:", len(batch_images), "张图片角点！")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgGray.shape[::-1], None, None)
    dict = {'ret': ret, 'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}
    writetofile(dict, SaveParamPath)

    meanError = 0
    total_error = 0
    for i in range(len(objpoints)):
        #     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        #     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        #     meanError += error
        # print("total error: ", meanError / len(objpoints))

        #    修正
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        total_error += error * error
    print("rms error: {}".format(np.sqrt(total_error / (len(objpoints) * len(imgpoints2)))))


def showpoint(img, ptx):
    for i in range(ptx.shape[1]):
        x = int(round(ptx[0, i]))
        y = int(round(ptx[1, i]))
        # if x>20 and y>20 and x<640 and y <450:
        #   None
        cv2.circle(img, (x, y), 3, (255, 255, 255))
    return img


def epipolar_geometric(Images_Path, K):
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description='SuperPoints+SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default=Images_Path,  # 图像存储路径
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default='SImage_result',  # 图像结果路径
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[512, 512],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    # This class helps load input images from different sources.读图 读下一张图
    vs = VideoStreamer(opt.input, opt.resize, opt.skip, opt.image_glob, opt.max_length)
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'

    frame_tensor = frame2tensor(frame, device)
    last_data = matching.superpoint({'image': frame_tensor})
    # print('last_data = ', last_data)
    last_data = {k + '0': last_data[k] for k in keys}
    # print('last_data = ', last_data)
    last_data['image0'] = frame_tensor
    last_frame = frame
    last_image_id = 0

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')

    timer = AverageTimer()

    while True:
        frame, ret = vs.next_frame()
        if not ret:
            print('Finished demo_superglue.py')
            break
        timer.update('data')
        stem0, stem1 = last_image_id, vs.i - 1

        frame_tensor = frame2tensor(frame, device)
        pred = matching({**last_data, 'image1': frame_tensor})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        timer.update('forward')

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        color = cm.jet(confidence[valid])
        text = [
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]
        out = make_matching_plot_fast(
            last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=opt.show_keypoints, small_text=small_text)
        # print('kpts0 = ', kpts0)
        # print('kpts1 = ', kpts1)
        # print('mkpts0 = ', mkpts0)
        # print('mkpts1 = ', mkpts1)

        # 特征点图
        pts1 = cart2hom(kpts0.T)
        pts2 = cart2hom(kpts1.T)
        plt.figure(figsize=(6, 6))
        plt.subplot(221), plt.axis('off'), plt.title("Contour1")
        plt.imshow(last_frame, 'Blues')
        imgx = last_frame.copy()
        img11 = showpoint(imgx, pts1)
        plt.subplot(222), plt.axis('off'), plt.title("SuperPoint_KeyPoints1：66")
        plt.imshow(img11, 'CMRmap_r')
        plt.subplot(223), plt.axis('off'), plt.title("Contour2")
        plt.imshow(frame, 'Blues')
        imgxx = frame.copy()
        img22 = showpoint(imgxx, pts2)
        plt.subplot(224), plt.axis('off'), plt.title("SuperPoint_KeyPoints2：64")
        plt.imshow(img22, 'CMRmap_r')
        plt.tight_layout()
        plt.show()

        # 匹配图
        # cv2.imshow('SuperGlue matches', out)
        plt.figure(figsize=(9, 5))
        plt.subplot(111), plt.axis('off'), plt.title("SuperGlue_Output", fontweight='bold', fontsize=12)
        plt.imshow(out, 'CMRmap_r')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0, 0)
        plt.tight_layout()
        plt.show()

    points1 = cart2hom(mkpts0.T)
    points2 = cart2hom(mkpts1.T)
    # print('points1.T = ', points1.T)
    # print('points1 = ', points1)
    # print('points1[0] = ', points1[0])
    # print('points1[1] = ', points1[1])
    # print('points1[2] = ', points1[2])
    # print('points1.shape = ', points1.shape[1])

    points1n = np.dot(np.linalg.inv(K), points1)
    # print('points1n = ', points1n)
    points2n = np.dot(np.linalg.inv(K), points2)
    E = compute_essential_normalized(points1n, points2n)
    print('Computed essential matrix:', (-E / E[0][1]))

    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2s = compute_P_from_essential(E)

    ind = -1
    for i, P2 in enumerate(P2s):
        # Find the correct camera parameters
        d1 = reconstruct_one_point(points1n[:, 0], points2n[:, 0], P1, P2)
        # Convert P2 from camera view to world view
        P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
        d2 = np.dot(P2_homogenous[:3, :4], d1)
        if d1[2] > 0 and d2[2] > 0:
            ind = i

    P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
    Points3D = linear_triangulation(points1n, points2n, P1, P2)
    print('Points3D = ', Points3D)
    # print('Points3D[0] = ', Points3D[0])
    print('Points3D.shape = ', Points3D.shape[1])

    return Points3D


def main():
    CameraParam_Path = 'CameraParam0.txt'
    CheckerboardImage_Path = 'Checkerboard_Image0'
    # Images_Path = 'SubstitutionCalibration_Image0/*.jpg'
    # Images_Path = 'SubstitutionCalibration_Image9'
    Images_Path = 'SubstitutionCalibration_Image10'

    # 计算相机参数
    camera_calibration(CameraParam_Path, CheckerboardImage_Path)
    # 读取相机参数
    config = readfromfile(CameraParam_Path)
    K = np.array(config['mtx'])
    # 计算3D点
    Points3D = epipolar_geometric(Images_Path, K)
    # #  保存3D点
    # data1 = pd.DataFrame(Points3D)
    # print("data1的type: {}".format(type(data1)))
    # print("data1的shape: {}".format(data1.shape))
    # print(data1)
    # data1.to_csv('64.csv', index=False)
    # data = pd.read_csv('64.csv')
    # print(data)
    # print(type(Points3D[0]), Points3D[1], Points3D[2])
    # 重建3D点
    fig = plt.figure()
    # fig.suptitle('3D Unordered Feature Points', fontweight='bold', fontsize=12)
    ax = Axes3D(fig)
    fig.add_axes(ax)
    ax.scatter(Points3D[0], Points3D[1], Points3D[2], label='UFP', c="red")
    # ax.plot(Points3D[0], Points3D[1], 'b.')
    # 标号
    x = Points3D[0]
    y = Points3D[1]
    z = Points3D[2]
    for i in range(len(x)):
        ax.text(x[i], y[i], z[i], i)
    mpl.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'无法显示的问题
    ax.set_xlabel('X', fontsize=16)
    # ax.set_xlim(0.0750, 0.0950)
    ax.set_ylabel('Y', fontsize=16)
    # ax.set_ylim(0.1375, 0.1575)
    ax.set_zlabel('Z', fontsize=16)
    # ax.set_zlim(-0.5025, -0.4825)
    # ax.view_init(elev=-150, azim=60)
    ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,
              ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0, 0)
    # plt.savefig('Reconstruction.png', transparent=False, pad_inches=0.0, dpi=1000)
    plt.show()


if __name__ == '__main__':
    main()
