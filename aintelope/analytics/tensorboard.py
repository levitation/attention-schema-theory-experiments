import os
import subprocess
from pathlib import Path
from collections import defaultdict

def get_common_subdirs(base_dirs):
    # Get all subdirectories for each selected directory
    subdir_counts = defaultdict(int)
    for dir_path in base_dirs:
        try:
            subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
            for subdir in subdirs:
                subdir_counts[subdir] += 1
        except Exception as e:
            print(f"Error reading directory {dir_path}: {e}")
    
    # Find subdirectories that exist in all selected directories
    common_subdirs = [subdir for subdir, count in subdir_counts.items() 
                     if count == len(base_dirs)]
    
    return sorted(common_subdirs)

def launch_tensorboard():
    base_dir = os.getcwd()
    
    try:
        # First level: Get all directories in current path
        directories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        if not directories:
            print("No directories found in current path!")
            return
        
        # Print available directories
        print("\nAvailable experiment directories:")
        for idx, directory in enumerate(directories, 1):
            print(f"{idx}. {directory}")
        
        # Get user input for first level
        print("\nEnter the numbers of directories you want to compare (separated by spaces):")
        selected = input("> ").strip().split()
        
        # Get full paths of selected directories
        selected_dirs = []
        for num in selected:
            try:
                idx = int(num) - 1
                if 0 <= idx < len(directories):
                    dir_name = directories[idx]
                    full_path = os.path.join(base_dir, dir_name)
                    selected_dirs.append(full_path)
                else:
                    print(f"Invalid number: {num}")
                    return
            except ValueError:
                print(f"Invalid input: {num}")
                return

        # Find common subdirectories
        common_subdirs = get_common_subdirs(selected_dirs)
        
        if not common_subdirs:
            print("\nNo common subdirectories found across selected directories!")
            return
            
        # Print available subdirectories
        print("\nCommon subdirectories found across selected experiments:")
        for idx, subdir in enumerate(common_subdirs, 1):
            print(f"{idx}. {subdir}")
            
        # Get user input for subdirectories
        print("\nEnter the numbers of subdirectories you want to view (separated by spaces):")
        selected_subdirs = input("> ").strip().split()
        
        # Create logdir string combining all selected directories and subdirectories
        logdir_parts = []
        for num in selected_subdirs:
            try:
                idx = int(num) - 1
                if 0 <= idx < len(common_subdirs):
                    subdir_name = common_subdirs[idx]
                    for main_dir in directories:
                        if main_dir in selected_dirs[0]:  # Use first selected dir for naming
                            label = f"{main_dir}/{subdir_name}"
                            break
                    paths = [os.path.join(dir_path, subdir_name) for dir_path in selected_dirs]
                    logdir_parts.extend(f"{label}:{path}" for path in paths)
                else:
                    print(f"Invalid number: {num}")
                    return
            except ValueError:
                print(f"Invalid input: {num}")
                return
        
        # Create the final logdir string
        logdir = ",".join(logdir_parts)
        
        # Launch tensorboard
        print("\nLaunching TensorBoard...")
        print(f"Using logdir: {logdir}")
        subprocess.run(["tensorboard", "--logdir", logdir])
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    launch_tensorboard()