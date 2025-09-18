import os
import shutil


def copy_and_rename_mic1_files(source_root_dir, destination_root_dir):
    """
    Copies .pt files containing "mic1" from a source directory structure
    to a new destination directory, renaming them by removing "_mic1".

    Args:
        source_root_dir (str): The path to the original root folder
                               (e.g., '/mnt/ceph_rbd/data/vctk/hubert_feature/large_l9').
        destination_root_dir (str): The path to the new root folder where
                                    files will be copied (e.g., 'new_hubert_features_mic1').
    """
    print(f"  Source: {source_root_dir}")
    print(f"  Destination: {destination_root_dir}\n")

    # Ensure the destination root directory exists
    os.makedirs(destination_root_dir, exist_ok=True)

    # Walk through the source directory structure
    for dirname in os.listdir(source_root_dir):
        relative_path = os.path.join(source_root_dir, dirname)
        if not os.path.isdir(relative_path):
            continue

        # Construct the corresponding destination directory path
        current_destination_dir = os.path.join(destination_root_dir, dirname)

        if dirname != ".": # Don't try to create the root dir again
            os.makedirs(current_destination_dir, exist_ok=True)

        for filename in os.listdir(relative_path):
            if filename.endswith(".pt") and "_mic1.pt" in filename:
                # Construct the full source file path
                source_file_path = os.path.join(relative_path, filename)

                # Generate the new filename by removing "_mic1"
                new_filename = filename.replace("_mic1.pt", ".pt")

                # Construct the full destination file path
                destination_file_path = os.path.join(current_destination_dir, new_filename)

                try:
                    os.system(f"ln -s {source_file_path} {destination_file_path}")
                    # shutil.copy2(source_file_path, destination_file_path)
                    # print(f"Copied and renamed: {source_file_path} -> {destination_file_path}")
                except Exception as e:
                    print(f"Error copying {source_file_path} to {destination_file_path}: {e}")

    print("\nOperation complete!")

# --- Example Usage ---
if __name__ == "__main__":
    # Define your source and destination directories
    # IMPORTANT: Replace these with your actual paths!
    SOURCE_FEATURES_DIR = "/mnt/ceph_rbd/data/vctk/hubert_feature/large_l18"
    DESTINATION_FEATURES_DIR = "/mnt/ceph_rbd/data/vctk/hubert_feature/large_l18_mic1"

    # Call the function to perform the operation
    copy_and_rename_mic1_files(SOURCE_FEATURES_DIR, DESTINATION_FEATURES_DIR)

