import os
import torch
import sys
import argparse

def parse_args():
    """
    Parses command-line arguments for the file reorganization script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_file',type=str)
    parser.add_argument('output_dir',type=str)

    args = parser.parse_args()
    return args

def reorganize_file(input_file_path, output_root_dir):
    """
    Reorganizes a text file by creating subdirectories based on a prefix,
    and saving the data to a PyTorch tensor file.

    Args:
        input_file_path (str): The path to the input text file.
        output_root_dir (str): The root directory to save the reorganized data.
    """
    # Create the output root directory if it doesn't exist
    os.makedirs(output_root_dir, exist_ok=True)

    with open(input_file_path, 'r') as f:
        for line in f:
            # Strip whitespace and split the line by space
            parts = line.strip().split()
            if not parts:
                continue

            # Get the path and the number sequence
            flac_path = parts[0]
            number_strings = parts[1:]

            # Extract the subdirectory name (e.g., 'p336')
            subdir_name = flac_path.split('/')[0]

            # Create the full path for the subdirectory
            output_subdir = os.path.join(output_root_dir, subdir_name)
            os.makedirs(output_subdir, exist_ok=True)

            # Reformat the filename (e.g., p336/p336_319_mic1.flac -> p336/p336_319.pt)
            new_filename = flac_path.split('/')[-1].replace('_mic1.flac', '.pt')
            output_file_path = os.path.join(output_subdir, new_filename)

            # Convert the list of number strings to a list of integers
            try:
                numbers = [int(num) for num in number_strings]
                # Convert the list of numbers into a PyTorch tensor
                numbers_tensor = torch.tensor(numbers)

                # Save the tensor to the new .pt file
                torch.save(numbers_tensor, output_file_path)
                print(f"Saved tensor to {output_file_path}")
            except ValueError as e:
                print(f"Skipping line due to invalid number format: {line.strip()} - Error: {e}")

def reorganize_file_l2(input_folder_path, output_root_dir):
    """
    Reorganizes a text file by creating subdirectories based on a prefix,
    and saving the data to a PyTorch tensor file.

    Args:
        input_file_path (str): The path to the input text file.
        output_root_dir (str): The root directory to save the reorganized data.
    """
    # Create the output root directory if it doesn't exist
    os.makedirs(os.path.join(output_root_dir, "tokens_reorg"), exist_ok=True)

    for dir in os.listdir(input_folder_path):
        if dir.split("_")[-1] == "test":
            input_file_path = os.path.join(input_folder_path, dir, "tokens")

            with open(input_file_path, 'r') as f:
                for line in f:
                    # Strip whitespace and split the line by space
                    parts = line.strip().split()
                    if not parts:
                        continue

                    # Get the path and the number sequence
                    flac_path = parts[0]
                    number_strings = parts[1:]

                    # Extract the subdirectory name (e.g., 'p336')
                    subdir_name = flac_path.split('/')[0]

                    # Create the full path for the subdirectory
                    output_subdir = os.path.join(output_root_dir, "tokens_reorg", subdir_name)
                    os.makedirs(output_subdir, exist_ok=True)

                    # Reformat the filename (e.g., p336/p336_319_mic1.flac -> p336/p336_319.pt)
                    new_filename = flac_path.split('/')[-1].replace('.wav', '.pt')
                    output_file_path = os.path.join(output_subdir, new_filename)

                    # Convert the list of number strings to a list of integers
                    try:
                        numbers = [int(num) for num in number_strings]
                        # Convert the list of numbers into a PyTorch tensor
                        numbers_tensor = torch.tensor(numbers)

                        # Save the tensor to the new .pt file
                        torch.save(numbers_tensor, output_file_path)
                        # print(f"Saved tensor to {output_file_path}")
                    except ValueError as e:
                        print(f"Skipping line due to invalid number format: {line.strip()} - Error: {e}")


# Example usage:
if __name__ == '__main__':
    args = parse_args()

    reorganize_file_l2(args.input_file, args.output_dir)