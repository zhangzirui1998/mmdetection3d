# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""

from __future__ import division
import os
import numpy as np
import mayavi.mlab as mlab


# 过滤指定范围之外的点和目标框
def get_filtered_lidar(lidar, boxes3d=None):
    xrange = (0, 70.4)
    yrange = (-40, 40)
    zrange = (-3, 1)
    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    filter_x = np.where((pxs >= xrange[0]) & (pxs < xrange[1]))[0]
    filter_y = np.where((pys >= yrange[0]) & (pys < yrange[1]))[0]
    filter_z = np.where((pzs >= zrange[0]) & (pzs < zrange[1]))[0]
    filter_xy = np.intersect1d(filter_x, filter_y)
    filter_xyz = np.intersect1d(filter_xy, filter_z)
    if boxes3d is not None:
        box_x = (boxes3d[:, :, 0] >= xrange[0]) & (boxes3d[:, :, 0] < xrange[1])
        box_y = (boxes3d[:, :, 1] >= yrange[0]) & (boxes3d[:, :, 1] < yrange[1])
        box_z = (boxes3d[:, :, 2] >= zrange[0]) & (boxes3d[:, :, 2] < zrange[1])
        box_xyz = np.sum(box_x & box_y & box_z, axis=1)

        return lidar[filter_xyz], boxes3d[box_xyz > 0]

    return lidar[filter_xyz]


def draw_lidar(lidar, is_grid=False, is_axis=True, is_top_region=True, fig=None):
    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    prs = lidar[:, 3]
    if fig is None: fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
    mlab.points3d(
        pxs, pys, pzs, prs,
        mode='point',  # 'point'  'sphere'
        colormap='gnuplot',  # 'bone',  #'spectral',  #'copper',
        scale_factor=1,
        figure=fig)
    # draw grid
    if is_grid:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
        for y in np.arange(-50, 50, 1):
            x1, y1, z1 = -50, y, 0
            x2, y2, z2 = 50, y, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(0.5, 0.5, 0.5), tube_radius=None, line_width=1, figure=fig)
        for x in np.arange(-50, 50, 1):
            x1, y1, z1 = x, -50, 0
            x2, y2, z2 = x, 50, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(0.5, 0.5, 0.5), tube_radius=None, line_width=1, figure=fig)

    # draw axis
    if is_grid:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
        axes = np.array([
            [2., 0., 0., 0.],
            [0., 2., 0., 0.],
            [0., 0., 2., 0.],
        ], dtype=np.float64)
        fov = np.array([  ##<todo> : now is 45 deg. use actual setting later ...
            [20., 20., 0., 0.],
            [20., -20., 0., 0.],
        ], dtype=np.float64)

        mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)
        mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                    figure=fig)
        mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                    figure=fig)

    # draw top_image feature area
    if is_top_region:
        # 关注指定范围内的点云
        x1 = 0
        x2 = 70.4
        y1 = -40
        y2 = 40
        mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=None, line_width=1, figure=fig)
    mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=None, distance=50,
              focalpoint=[12.0909996, -1.04700089, -2.03249991])  # 2.0909996 , -1.04700089, -2.03249991
    return fig


def draw_gt_boxes3d(gt_boxes3d, fig, color=(1, 0, 0), line_width=2):
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)
            i, j = k + 4, (k + 3) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)
            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)
    mlab.view(azimuth=180, elevation=None, distance=50,
              focalpoint=[12.0909996, -1.04700089, -2.03249991])  # 2.0909996 , -1.04700089, -2.03249991


def load_kitti_calib(calib_file):
    """
    load projection matrix
    """
    with open(calib_file) as fi:
        lines = fi.readlines()
        assert (len(lines) == 8)
    obj = lines[0].strip().split(' ')[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo = np.array(obj, dtype=np.float32)
    return {'P2': P2.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


def box3d_cam_to_velo(box3d, Tr):
    def project_cam2velo(cam, Tr):
        T = np.zeros([4, 4], dtype=np.float32)
        T[:3, :] = Tr
        T[3, 3] = 1
        T_inv = np.linalg.inv(T)
        lidar_loc_ = np.dot(T_inv, cam)
        lidar_loc = lidar_loc_[:3]
        return lidar_loc.reshape(1, 3)

    def ry_to_rz(ry):
        angle = -ry - np.pi / 2
        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2 * np.pi + angle
        return angle

    h, w, l, tx, ty, tz, ry = [float(i) for i in box3d]
    cam = np.ones([4, 1])
    cam[0] = tx
    cam[1] = ty
    cam[2] = tz
    t_lidar = project_cam2velo(cam, Tr)
    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [0, 0, 0, 0, h, h, h, h]])
    rz = ry_to_rz(ry)
    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])
    velo_box = np.dot(rotMat, Box)
    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T
    box3d_corner = cornerPosInVelo.transpose()
    return box3d_corner.astype(np.float32)


def load_kitti_label(label_file, Tr):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    gt_boxes3d_corner = []
    num_obj = len(lines)
    for j in range(num_obj):
        obj = lines[j].strip().split(' ')
        obj_class = obj[0].strip()
        if obj_class not in ['Car']:
            continue
        box3d_corner = box3d_cam_to_velo(obj[8:], Tr)
        gt_boxes3d_corner.append(box3d_corner)
    gt_boxes3d_corner = np.array(gt_boxes3d_corner).reshape(-1, 8, 3)
    return gt_boxes3d_corner


def test():
    lidar_path = os.path.join('./data/KITTI/training', "velodyne/")
    calib_path = os.path.join('./data/KITTI/training', "calib/")
    label_path = os.path.join('./data/KITTI/training', "label_2/")
    lidar_file = lidar_path + '/' + '000016' + '.bin'
    calib_file = calib_path + '/' + '000016' + '.txt'
    label_file = label_path + '/' + '000016' + '.txt'

    # 加载雷达数据
    print("Processing: ", lidar_file)
    lidar = np.fromfile(lidar_file, dtype=np.float32)
    lidar = lidar.reshape((-1, 4))

    # 加载标注文件
    calib = load_kitti_calib(calib_file)
    # 标注转三维目标检测框
    gt_box3d = load_kitti_label(label_file, calib['Tr_velo2cam'])

    # 过滤指定范围之外的点和目标框
    lidar, gt_box3d = get_filtered_lidar(lidar, gt_box3d)

    # view in point cloud，可视化
    fig = draw_lidar(lidar, is_grid=True, is_top_region=True)
    draw_gt_boxes3d(gt_boxes3d=gt_box3d, fig=fig)
    mlab.show()


if __name__ == '__main__':
    test()