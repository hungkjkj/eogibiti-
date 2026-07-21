import torch
import os
import csv
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrixCenterShift
from scene.colmap_loader import qvec2rotmat
import numpy as np

class MiniCamCenterShift:
    def __init__(self, width, height, fx, fy, cx, cy, world_view_transform, full_proj_transform, image_name=""):
        self.image_width = width
        self.image_height = height    
        self.FoVy = 2 * np.arctan(height / (2 * fy))
        self.FoVx = 2 * np.arctan(width / (2 * fx))
        self.znear = 0.01
        self.zfar = 100.0
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.image_name = image_name

def render_submission(model_path, test_poses_csv, output_dir, pipeline, sh_degree=3):
    with torch.no_grad():
        gaussians = GaussianModel(sh_degree)
        
        iteration = -1
        from utils.system_utils import searchForMaxIteration
        loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
        ply_path = os.path.join(model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply")
        print(f"Loading model from {ply_path}")
        gaussians.load_ply(ply_path)
        
        bg_color = [0, 0, 0] # Assume black background
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        makedirs(output_dir, exist_ok=True)
        
        poses = []
        with open(test_poses_csv, 'r') as f:
            reader = csv.DictReader(f)
            # check if header has spaces
            fieldnames = [name.strip() for name in reader.fieldnames]
            reader.fieldnames = fieldnames
            for row in reader:
                poses.append(row)
                
        for row in tqdm(poses, desc="Rendering progress"):
            image_name = row['image_name']
            qw, qx, qy, qz = float(row['qw']), float(row['qx']), float(row['qy']), float(row['qz'])
            tx, ty, tz = float(row['tx']), float(row['ty']), float(row['tz'])
            fx, fy = float(row['fx']), float(row['fy'])
            cx, cy = float(row['cx']), float(row['cy'])
            width, height = int(row['width']), int(row['height'])
            
            qvec = np.array([qw, qx, qy, qz])
            tvec = np.array([tx, ty, tz])
            R = np.transpose(qvec2rotmat(qvec))
            T = tvec
            
            world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
            projection_matrix = getProjectionMatrixCenterShift(znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height).transpose(0,1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            
            custom_cam = MiniCamCenterShift(width, height, fx, fy, cx, cy, world_view_transform, full_proj_transform, image_name=image_name)
            
            rendering = render(custom_cam, gaussians, pipeline, background)["render"]
            
            torchvision.utils.save_image(rendering, os.path.join(output_dir, image_name))

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--test_poses", required=True, type=str, help="Path to test_poses.csv")
    parser.add_argument("--output_dir", required=True, type=str, help="Path to output directory (e.g. scene_001/)")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    
    safe_state(args.quiet)
    
    # ModelParams extracts sh_degree
    model_params = model.extract(args)
    
    render_submission(args.model_path, args.test_poses, args.output_dir, pipeline.extract(args), sh_degree=model_params.sh_degree)
