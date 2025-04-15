from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import numpy as np
import open3d as o3d
import os
import trimesh

feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

def convert_image_to_3d(image_path, output_folder):
    image = Image.open(image_path).convert("RGB")
    new_height = 480 if image.height > 480 else image.height
    new_height -= new_height % 32
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + (32 - diff)
    image = image.resize((new_width, new_height))

    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    pad = 16
    output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    output = output[pad:-pad, pad:-pad]
    image = image.crop((pad, pad, image.width - pad, image.height - pad))

    width, height = image.size
    depth_image = (output * 255 / np.max(output)).astype(np.uint8)
    image_np = np.array(image)

    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image_np)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_o3d, depth_o3d, convert_rgb_to_intensity=False
    )

    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(width, height, 500.0, 500.0, width / 2, height / 2)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=6.0)
    pcd = pcd.select_by_index(ind)
    pcd.estimate_normals()
    pcd.orient_normals_towards_camera_location()

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)[0]
    mesh.rotate(mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0)), center=(0, 0, 0))

    # Save mesh as OBJ
    obj_path = os.path.join(output_folder, "model.obj")
    o3d.io.write_triangle_mesh(obj_path, mesh)

    # Convert to GLB with trimesh
    glb_path = os.path.join(output_folder, "model.glb")
    tm = trimesh.load(obj_path, force='mesh')
    tm.export(glb_path)

    return glb_path
