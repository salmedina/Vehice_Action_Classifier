import cv2
import numpy as np
from easydict import EasyDict as edict

def cal_contour(target):
    _, contour, _ = cv2.findContours(target.astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lengths = np.array([c.shape[0] for c in contour])
    return contour[lengths.argmax()][:, 0, :]

def cal_line(point_1, point_2):
    x1, y1 = point_1
    x2, y2 = point_2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return (a, b, c)

def cal_tangent_line(vanishing_point, contour):
    angles = []
    point_at_right = (contour.max(axis=0)[0] < vanishing_point[0])
    for point in contour:
        x, y = point - vanishing_point
        angle = np.arctan2(y, x)
        if point_at_right and angle < 0:
            angle += np.pi * 2
        angles.append(angle)
    angle_min, angle_max = min(angles), max(angles)
    line_1 = cal_line(vanishing_point, contour[angles.index(angle_min)])
    line_2 = cal_line(vanishing_point, contour[angles.index(angle_max)])
    return line_1, line_2

def cal_intersection(line_1, line_2):
    a1, b1, c1 = line_1
    a2, b2, c2 = line_2
    m = a1 * b2 - a2 * b1
    if m == 0:
        return None
    x = (c2 * b1 - c1 * b2) / m
    y = (c1 * a2 - c2 * a1) / m
    return np.array([x, y])

def build_box_3d(target, calibration):
    box_3d = edict({'lines': {'vp1': {}, 'vp2': {}, 'vp3': {}}, 'points': {}})
    contour = cal_contour(target)

    box_3d.lines.vp1.l, box_3d.lines.vp1.r = cal_tangent_line(calibration.image_vp1, contour)
    box_3d.lines.vp2.l, box_3d.lines.vp2.r = cal_tangent_line(calibration.image_vp2, contour)
    box_3d.lines.vp3.l, box_3d.lines.vp3.r = cal_tangent_line(calibration.image_vp3, contour)

    box_3d.points.a = cal_intersection(box_3d.lines.vp1.r, box_3d.lines.vp2.l)
    box_3d.points.b = cal_intersection(box_3d.lines.vp2.l, box_3d.lines.vp3.r)
    box_3d.points.d = cal_intersection(box_3d.lines.vp1.r, box_3d.lines.vp3.l)
    box_3d.points.h = cal_intersection(box_3d.lines.vp2.r, box_3d.lines.vp3.l)
    box_3d.points.f = cal_intersection(box_3d.lines.vp1.l, box_3d.lines.vp3.r)
    box_3d.points.g = cal_intersection(box_3d.lines.vp1.l, box_3d.lines.vp2.r)

    line_vp1_d = cal_line(calibration.image_vp1, box_3d.points.d)
    line_vp3_a = cal_line(calibration.image_vp3, box_3d.points.a)
    point_e_d = cal_intersection(line_vp1_d, line_vp3_a)
    line_vp2_f = cal_line(calibration.image_vp2, box_3d.points.f)
    point_e_f = cal_intersection(line_vp2_f, line_vp3_a)

    if np.linalg.norm(point_e_d - box_3d.points.a) \
       >= np.linalg.norm(point_e_f - box_3d.points.a):
        box_3d.points.e = point_e_d
    else:
        box_3d.points.e = point_e_f

    line_vp2_d = cal_line(calibration.image_vp2, box_3d.points.d)
    line_vp1_b = cal_line(calibration.image_vp1, box_3d.points.b)

    box_3d.points.c = cal_intersection(line_vp2_d, line_vp1_b)
    box_3d.direction = (box_3d.points.a[1] - box_3d.points.d[1]) / \
                       (box_3d.points.a[0] - box_3d.points.d[0])
    return box_3d

def cal_plane_box(box_3d, calibration):
    if box_3d.direction >= 0:
        bottom_points = ['a', 'b', 'c', 'd']
    else:
        bottom_points = ['h', 'g', 'f', 'e']
    plane_box = edict()
    plane_box.points = [calibration.image_to_plane(box_3d.points[pos])
                        for pos in bottom_points]
    plane_box.center = np.average(plane_box.points, axis=0)
    plane_box.point_offsets = [point - plane_box.center for point
                               in plane_box.points]
    return plane_box

def gen_box(plane_box, new_center, rotation):
    new_box = edict()
    new_box.center = new_center
    cos_a, sin_a = np.cos(rotation), np.sin(rotation)
    transformer = np.array([[cos_a, -sin_a],
                            [sin_a, cos_a]])
    new_box.point_offsets = [np.dot(transformer, offset)
                             for offset in plane_box.point_offsets]
    new_box.points = [new_center + offset for offset in new_box.point_offsets]
    return new_box

def cal_offsets(plane_box):
    x_min, y_min = np.floor(np.min(plane_box.point_offsets, axis=0)).astype(int)
    x_max, y_max = np.ceil(np.max(plane_box.point_offsets, axis=0)).astype(int)
    x_length = x_max - x_min + 1
    y_length = y_max - y_min + 1
    x_space = np.linspace(x_min, x_max, x_length)
    y_space = np.linspace(y_min, y_max, y_length)
    xs, ys = np.meshgrid(x_space, y_space, indexing='ij')
    flags = []
    points = plane_box.point_offsets
    for point_1, point_2 in zip(points, points[1:] + points[:1]):
        a, b, c = cal_line(point_1, point_2)
        flags.append(a * xs + b * ys + c > 0)
    area = x_length * y_length
    box_area = 1
    for flag in flags:
        if flag.sum() < 0.5 * area:
            flag = ~flag
        box_area &= flag
    xs, ys = np.where(box_area)
    offset_range = edict({'x_min': x_min, 'x_max': x_max,
                          'y_min': y_min, 'y_max': y_max,
                          'x_length': x_length, 'y_length': y_length})
    area_offsets = edict({'x': xs + x_min, 'y': ys + y_min})
    return offset_range, area_offsets

def cal_heat_map(plane_box, sigma=2, size_factor=3, threshold=1e-4):
    offset_range, area_offsets = cal_offsets(plane_box)
    center_range_min = int(np.floor(-size_factor * sigma))
    center_range_max = int(np.ceil(size_factor * sigma))
    space = np.linspace(center_range_min, center_range_max,
                        center_range_max - center_range_min + 1)
    xs, ys = np.meshgrid(space, space, indexing='ij')
    center_dist = np.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2))
    center_dist /= center_dist.sum()
    shape_0 = offset_range.y_length + center_range_max - center_range_min
    shape_1 = offset_range.x_length + center_range_max - center_range_min
    heat_map = np.zeros(shape=(shape_0, shape_1))
    for center_x, center_y in zip(*np.where(center_dist > threshold)):
        heat_xs = area_offsets.x + center_x - offset_range.x_min
        heat_ys = area_offsets.y + center_y - offset_range.y_min
        heat_map[heat_ys, heat_xs] += center_dist[center_x, center_y]
    x_min = int(np.floor(center_range_min + offset_range.x_min
                         + plane_box.center[0]))
    x_max = int(np.ceil(center_range_max + offset_range.x_max
                        + plane_box.center[0]))
    y_min = int(np.floor(center_range_min + offset_range.y_min
                         + plane_box.center[1]))
    y_max = int(np.ceil(center_range_max + offset_range.y_max
                        + plane_box.center[1]))
    position = (x_min, x_max, y_min, y_max)
    return heat_map, position

def cal_boxes(frame, calibration):
    for vehicle in frame.vehicles:
        vehicle.image_box_3d = build_box_3d(vehicle.image_mask, calibration)
        vehicle.plane_box = cal_plane_box(vehicle.image_box_3d, calibration)