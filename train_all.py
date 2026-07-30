import os
import subprocess

def main():
    data_dir = "data"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    for scene_name in os.listdir(data_dir):
        scene_data_path = os.path.join(data_dir, scene_name, "train")
        if not os.path.isdir(scene_data_path):
            continue
            
        model_path = os.path.join(output_dir, scene_name)
        
        print(f"Training {scene_name}...")
        cmd = [
            "python", "train.py",
            "-s", scene_data_path,
            "-m", model_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error training {scene_name}: {e}")
            continue
            
    print("All training finished!")

if __name__ == "__main__":
    main()
