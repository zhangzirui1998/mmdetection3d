import trimesh


def to_ply(input_path, output_path, original_type):
    # input_path = 'demo/kitti_000008/kitti_000008_pred.obj'
    mesh = trimesh.load(input_path, file_type=original_type)  # read file
    mesh.export(output_path, file_type='ply')  # convert to ply


to_ply('/demo/kitti_000008/kitti_000008_pred.obj', '/home/rui/mmdetection3d/demo/kitti_000008/kitti_000008_pred.ply', 'obj')
