import os
import argparse
import subprocess
import shutil

def main():
    parser = argparse.ArgumentParser(description="Run 3DGS rendering for all scenes and package for submission")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing scene folders (e.g., scene_001, scene_002)")
    parser.add_argument("--models_dir", type=str, required=True, help="Directory containing trained models for each scene")
    parser.add_argument("--output_zip", type=str, default="submission.zip", help="Output zip file path")
    args = parser.parse_args()
    
    temp_dir = "temp_submission"
    os.makedirs(temp_dir, exist_ok=True)
    
    for scene_name in os.listdir(args.data_dir):
        scene_data_path = os.path.join(args.data_dir, scene_name)
        if not os.path.isdir(scene_data_path):
            continue
            
        test_poses_csv = os.path.join(scene_data_path, "test", "test_poses.csv")
        if not os.path.exists(test_poses_csv):
            print(f"Skipping {scene_name} because {test_poses_csv} does not exist.")
            continue
            
        model_path = os.path.join(args.models_dir, scene_name)
        if not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} does not exist. Skipping.")
            continue
            
        output_scene_dir = os.path.join(temp_dir, scene_name)
        
        print(f"Processing {scene_name}...")
        cmd = [
            "python", "render_submission.py",
            "-m", model_path,
            "--test_poses", test_poses_csv,
            "--output_dir", output_scene_dir
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {scene_name}: {e}")
            continue
            
    print(f"Zipping results to {args.output_zip}...")
    zip_base = args.output_zip
    if zip_base.endswith('.zip'):
        zip_base = zip_base[:-4]
    
    shutil.make_archive(zip_base, 'zip', temp_dir)
    print("Done!")
    
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
