#!/usr/bin/env python3
"""
整合实际轨迹和预测轨迹数据 - 正确转换局部坐标到世界坐标
"""

import json
import os
import shutil
import argparse
import numpy as np
import math


def load_actual_trajectory(measurements_dir):
    """加载实际行驶轨迹"""
    trajectory = []
    if not os.path.exists(measurements_dir):
        print(f"Warning: {measurements_dir} not found")
        return trajectory

    json_files = sorted([f for f in os.listdir(measurements_dir) if f.endswith('.json')])

    for json_file in json_files:
        try:
            with open(os.path.join(measurements_dir, json_file), 'r') as f:
                data = json.load(f)
                frame_num = int(json_file.replace('.json', ''))

                trajectory.append({
                    'frame': frame_num,
                    'x': data['x'],
                    'y': data['y'],
                    'yaw': data.get('yaw', 0)
                })
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    return trajectory


def normalize_trajectory(trajectory):
    """将轨迹平移到原点附近"""
    if not trajectory:
        return trajectory, (0, 0)

    xs = [p['x'] for p in trajectory]
    ys = [p['y'] for p in trajectory]
    center_x = (max(xs) + min(xs)) / 2
    center_y = (max(ys) + min(ys)) / 2

    normalized = []
    for p in trajectory:
        normalized.append({
            'frame': p['frame'],
            'x': p['x'] - center_x,
            'y': p['y'] - center_y,
            'yaw': p['yaw']
        })

    return normalized, (center_x, center_y)


def transform_local_to_world(local_x, local_y, vehicle_x, vehicle_y, vehicle_yaw_rad):
    """
    将局部坐标（车辆坐标系）转换到世界坐标系
    local_x, local_y: 在车辆坐标系中的坐标（前方为x正，左为y正？）
    vehicle_x, vehicle_y: 车辆在世界坐标系中的位置
    vehicle_yaw_rad: 车辆的朝向（弧度）
    """
    # 旋转矩阵
    world_x = vehicle_x + local_x * math.cos(vehicle_yaw_rad) - local_y * math.sin(vehicle_yaw_rad)
    world_y = vehicle_y + local_x * math.sin(vehicle_yaw_rad) + local_y * math.cos(vehicle_yaw_rad)
    return world_x, world_y


def prepare_visualization_data(record_dir, inference_dir, output_dir):
    """准备完整的可视化数据"""

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    # 四路相机
    cameras = ['rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear']

    # 获取第一路相机的所有帧
    first_cam = cameras[0]
    first_cam_path = os.path.join(record_dir, first_cam)

    if not os.path.exists(first_cam_path):
        print(f"Error: Cannot find {first_cam_path}")
        return False

    # 获取所有图片文件
    image_files = sorted([f for f in os.listdir(first_cam_path) if f.endswith('.png')])
    print(f"Found {len(image_files)} frames")

    # 复制图片到frames目录
    for cam in cameras:
        cam_source = os.path.join(record_dir, cam)
        cam_target = os.path.join(frames_dir, cam)
        os.makedirs(cam_target, exist_ok=True)

        if not os.path.exists(cam_source):
            print(f"Warning: {cam_source} not found")
            continue

        for img in image_files:
            src = os.path.join(cam_source, img)
            dst = os.path.join(cam_target, img)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
        print(f"  {cam}: {len(image_files)} images")

    # 加载实际轨迹
    measurements_dir = os.path.join(record_dir, 'measurements')
    actual_trajectory = load_actual_trajectory(measurements_dir)
    print(f"\nLoaded {len(actual_trajectory)} actual trajectory points")

    # 显示实际轨迹范围
    if actual_trajectory:
        xs = [p['x'] for p in actual_trajectory]
        ys = [p['y'] for p in actual_trajectory]
        print(f"  Actual X range: {min(xs):.2f} to {max(xs):.2f}")
        print(f"  Actual Y range: {min(ys):.2f} to {max(ys):.2f}")

    # 归一化实际轨迹（用于可视化）
    actual_trajectory_normalized, actual_center = normalize_trajectory(actual_trajectory)
    if actual_trajectory_normalized:
        xs = [p['x'] for p in actual_trajectory_normalized]
        ys = [p['y'] for p in actual_trajectory_normalized]
        print(f"  Normalized X range: {min(xs):.2f} to {max(xs):.2f}")
        print(f"  Normalized Y range: {min(ys):.2f} to {max(ys):.2f}")

    # 加载预测轨迹
    traj_json_dir = os.path.join(inference_dir, 'traj_json')
    predictions = []
    if os.path.exists(traj_json_dir):
        json_files = sorted([f for f in os.listdir(traj_json_dir) if f.endswith('.json')])
        print(f"\nFound {len(json_files)} prediction files")

        for json_file in json_files:
            with open(os.path.join(traj_json_dir, json_file), 'r') as f:
                data = json.load(f)

                # 提取预测轨迹点
                traj_points = []
                for point in data.get('trajectory', []):
                    traj_points.append({
                        'x': point['x'],
                        'y': point['y'],
                        'yaw': point.get('yaw', 0)
                    })

                predictions.append({
                    'timestamp': data['header']['stamp'],
                    'start_pose': data['start_pose'],
                    'target_point': data['target_point'],
                    'trajectory': traj_points,
                    'filename': json_file
                })

    print(f"Loaded {len(predictions)} predictions")

    # 关键：找到预测对应的实际车辆位置和朝向
    # 由于实际轨迹没有时间戳，我们使用帧序号来匹配
    # 假设预测是按顺序的，第i个预测对应实际轨迹的第i个点

    print("\n=== Converting predictions to world coordinates ===")
    processed_predictions = []

    for idx, pred in enumerate(predictions):
        # 找到对应的实际车辆状态
        # 使用索引匹配（假设预测顺序和实际轨迹顺序一致）
        if idx < len(actual_trajectory):
            vehicle_state = actual_trajectory[idx]
        else:
            # 如果预测多于实际，使用最后一个实际点
            vehicle_state = actual_trajectory[-1] if actual_trajectory else None

        if vehicle_state is None:
            print(f"Warning: No vehicle state for prediction {idx}")
            continue

        # 车辆在世界坐标系中的位置和朝向
        vehicle_x = vehicle_state['x']
        vehicle_y = vehicle_state['y']
        vehicle_yaw_rad = math.radians(vehicle_state['yaw'])

        print(f"\nPrediction {idx} (timestamp {pred['timestamp']:.1f}s):")
        print(f"  Vehicle at frame {vehicle_state['frame']}: "
              f"pos=({vehicle_x:.2f}, {vehicle_y:.2f}), yaw={vehicle_state['yaw']:.1f}°")

        # 预测的局部坐标（相对于车辆）
        pred_start_local = pred['start_pose']
        print(f"  Pred start_pose (local): ({pred_start_local['x']:.3f}, {pred_start_local['y']:.3f})")

        # 将预测起点转换到世界坐标
        start_world_x, start_world_y = transform_local_to_world(
            pred_start_local['x'], pred_start_local['y'],
            vehicle_x, vehicle_y, vehicle_yaw_rad
        )

        print(f"  Pred start_pose (world): ({start_world_x:.2f}, {start_world_y:.2f})")

        # 转换所有轨迹点
        world_traj = []
        for point in pred['trajectory']:
            world_x, world_y = transform_local_to_world(
                point['x'], point['y'],
                vehicle_x, vehicle_y, vehicle_yaw_rad
            )
            world_traj.append({
                'x': world_x,
                'y': world_y,
                'yaw': point['yaw']  # 保持原yaw或需要进一步处理
            })

        # 转换目标点
        world_target = None
        if pred['target_point']:
            world_target_x, world_target_y = transform_local_to_world(
                pred['target_point']['x'], pred['target_point']['y'],
                vehicle_x, vehicle_y, vehicle_yaw_rad
            )
            world_target = {'x': world_target_x, 'y': world_target_y}

        # 显示转换后的范围
        if world_traj:
            xs = [p['x'] for p in world_traj]
            ys = [p['y'] for p in world_traj]
            print(f"  World trajectory X range: {min(xs):.2f} to {max(xs):.2f}")
            print(f"  World trajectory Y range: {min(ys):.2f} to {max(ys):.2f}")

        # 归一化到可视化坐标
        normalized_traj = []
        for point in world_traj:
            normalized_traj.append({
                'x': point['x'] - actual_center[0],
                'y': point['y'] - actual_center[1],
                'yaw': point['yaw']
            })

        normalized_target = None
        if world_target:
            normalized_target = {
                'x': world_target['x'] - actual_center[0],
                'y': world_target['y'] - actual_center[1]
            }

        normalized_start = {
            'x': start_world_x - actual_center[0],
            'y': start_world_y - actual_center[1],
            'z': pred_start_local.get('z', 0),
            'yaw': pred_start_local.get('yaw', 0)
        }

        # 显示归一化后的范围
        if normalized_traj:
            xs = [p['x'] for p in normalized_traj]
            ys = [p['y'] for p in normalized_traj]
            print(f"  Normalized X range: {min(xs):.2f} to {max(xs):.2f}")
            print(f"  Normalized Y range: {min(ys):.2f} to {max(ys):.2f}")

        processed_predictions.append({
            'timestamp': pred['timestamp'],
            'start_pose': normalized_start,
            'target_point': normalized_target,
            'trajectory': normalized_traj,
            'vehicle_frame': vehicle_state['frame']
        })

    print(f"\n=== Final Results ===")
    print(f"Processed {len(processed_predictions)} predictions")

    # 验证重叠
    if processed_predictions and actual_trajectory_normalized:
        pred_xs = [x for p in processed_predictions for x in [pt['x'] for pt in p['trajectory']]]
        pred_ys = [y for p in processed_predictions for y in [pt['y'] for pt in p['trajectory']]]
        actual_xs = [p['x'] for p in actual_trajectory_normalized]
        actual_ys = [p['y'] for p in actual_trajectory_normalized]

        print(f"\nOverlap check:")
        print(f"  Predictions X: {min(pred_xs):.2f} to {max(pred_xs):.2f}")
        print(f"  Actual X:      {min(actual_xs):.2f} to {max(actual_xs):.2f}")
        print(f"  Predictions Y: {min(pred_ys):.2f} to {max(pred_ys):.2f}")
        print(f"  Actual Y:      {min(actual_ys):.2f} to {max(actual_ys):.2f}")

    # 保存manifest
    manifest = {
        'total_frames': len(image_files),
        'cameras': cameras,
        'camera_names': {
            'rgb_front': 'Front Camera',
            'rgb_left': 'Left Camera',
            'rgb_right': 'Right Camera',
            'rgb_rear': 'Rear Camera'
        },
        'frame_names': [f.replace('.png', '') for f in image_files],
        'fps': 10,
        'actual_trajectory': actual_trajectory_normalized,
        'predictions': processed_predictions
    }

    with open(os.path.join(output_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Done! Output: {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare complete visualization data')
    parser.add_argument('record_dir', help='Path to record folder (contains rgb_*, measurements)')
    parser.add_argument('inference_dir', help='Path to inference output folder (contains traj_json)')
    parser.add_argument('-o', '--output', default='./web_complete', help='Output directory')
    args = parser.parse_args()

    prepare_visualization_data(args.record_dir, args.inference_dir, args.output)


if __name__ == '__main__':
    main()