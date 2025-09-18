import pandas as pd
import os, sys, re
from multiprocessing import Pool

global speaker_info_df_global

def worker_init(df):
    global speaker_info_df_global
    speaker_info_df_global = df

def read_speaker_info(path = "/mnt/ceph_rbd/data/vctk/speaker-info.txt"):
    with open(path, 'r') as outfile:
        head_row = outfile.readline()
        data_files = outfile.readlines()
    # The last two column names are joined, so we manually define them
    columns = ['ID', 'AGE', 'GENDER', 'ACCENTS', 'REGION COMMENTS']

    # List to hold the parsed data
    parsed_data = []

    # Process each line of the data
    for line in data_files:
        # Skip any empty lines
        if not line.strip():
            continue

        # Split the line by whitespace. This handles multiple spaces between columns.
        parts = line.strip().split()
        
        # Ensure the line has enough parts to be parsed
        if len(parts) < 4:
            continue # Or handle error appropriately

        # The first four elements are the first four columns
        row_id = parts[0]
        age = parts[1]
        gender = parts[2]
        accent = parts[3]
        
        # All remaining elements are joined to form the 'REGION COMMENTS'
        # This handles cases where the comment is empty, one word, or multiple words.
        region_comment = ' '.join(parts[4:])
        
        parsed_data.append([row_id, age, gender, accent, region_comment])

    # Create the pandas DataFrame
    df = pd.DataFrame(parsed_data, columns=columns)

    # Convert AGE column to a numeric type, handling potential errors
    df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')

    # # Display the first few rows of the resulting DataFrame
    # print("--- DataFrame Head ---")
    # print(df.head())
    # print("\n--- Example Rows with Multi-Word Comments ---")
    print(df[df['ID'] == 'p225'])
    print(df[df['ID'] == 'p280'])
    return df

def real_phoneme(phoneme):
    return re.sub(r'\d+$', '', str(phoneme)).lower()


def csv_single_process(scratch_path, speaker_dir, filename):
    outfile_list = []
    csv_path = os.path.join(scratch_path, speaker_dir, filename)

    # The file ID is assumed to be the CSV filename without extension
    file_id = os.path.splitext(filename)[0] # speaker_dir + "/" +
    name_speaker = filename.split("_")[0]

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Filter for 'phones'
    phone_df = df[df['Type'] == 'phones'].copy()

    if phone_df.empty:
        # print(f"Warning: No phone entries found in {csv_path}. Skipping.")
        return None

    # Ensure the DataFrame is sorted by 'Begin' to get correct sequence
    phone_df = phone_df.sort_values(by='Begin').reset_index(drop=True)

    speaker = phone_df['Speaker'].iloc[0]
    assert speaker == name_speaker

    global speaker_info_df_global
    # Iterate through phone entries to determine prev/next phones
    phone_labels = phone_df['Label'].tolist()
    speaker_row = speaker_info_df_global.loc[speaker_info_df_global['ID'] == speaker]
    accent_labels, gender_labels = speaker_row.ACCENTS.values[0], speaker_row.GENDER.values[0]

    for i, row in phone_df.iterrows():
        onset = row['Begin']
        offset = row['End']
        current_phone = real_phoneme(row['Label'])
        if offset - onset <= 0.02:
            print(f"too short phone idx {i} duration {offset - onset}")
            continue

        prev_phone = phone_labels[i-1] if i > 0 else 'spn' # Use 'SIL' or '<UNK>' for start
        next_phone = phone_labels[i+1] if i < len(phone_labels) - 1 else 'spn' # Use 'SIL' or '<UNK>' for end
        prev_phone, next_phone = real_phoneme(prev_phone), real_phoneme(next_phone)

        # Format and write to the output file
        outfile_list.append(
            f"{file_id} {onset:.4f} {offset:.4f} {current_phone} {prev_phone} {next_phone} {speaker} {accent_labels} {gender_labels}\n"
        ) #  
    return outfile_list


def csv_single_process_words(scratch_path, speaker_dir, filename):
    outfile_list = []
    csv_path = os.path.join(scratch_path, speaker_dir, filename)
    file_id = os.path.splitext(filename)[0] # speaker_dir + "/" +
    name_speaker = filename.split("_")[0]

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Filter for 'phones'
    phone_df = df[df['Type'] == 'words'].copy()

    if phone_df.empty:
        # print(f"Warning: No phone entries found in {csv_path}. Skipping.")
        return None

    # Ensure the DataFrame is sorted by 'Begin' to get correct sequence
    phone_df = phone_df.sort_values(by='Begin').reset_index(drop=True)

    speaker = phone_df['Speaker'].iloc[0]
    assert speaker == name_speaker

    global speaker_info_df_global
    # # Iterate through phone entries to determine prev/next phones
    # phone_labels = phone_df['words'].tolist()
    speaker_row = speaker_info_df_global.loc[speaker_info_df_global['ID'] == speaker]
    accent_labels, gender_labels = speaker_row.ACCENTS.values[0], speaker_row.GENDER.values[0]

    for i, row in phone_df.iterrows():
        onset = row['Begin']
        offset = row['End']
        current_phone = real_phoneme(row['Label'])
        if offset - onset <= 0.02:
            print(f"too short phone idx {i} duration {offset - onset}")
            continue

        prev_phone = phone_labels[i-1] if i > 0 else 'spn' # Use 'SIL' or '<UNK>' for start
        next_phone = phone_labels[i+1] if i < len(phone_labels) - 1 else 'spn' # Use 'SIL' or '<UNK>' for end
        prev_phone, next_phone = real_phoneme(prev_phone), real_phoneme(next_phone)

        # Format and write to the output file
        outfile_list.append(
            f"{file_id} {onset:.4f} {offset:.4f} {current_phone} {prev_phone} {next_phone} {speaker} {accent_labels} {gender_labels}\n"
        ) #  
    return outfile_list

def convert_alignment_csv_to_item_file(
    scratch_path: str,
    output_item_file: str,
    speaker_info_df,
    n_workers: int = 4,
):
    """
    Converts alignment CSV files from speaker subdirectories into a single item file.
    """

    print(f"Starting conversion from {scratch_path} to {output_item_file}...")
    # Walk through the root alignment directory

    inputs_list = []
    for speaker_dir in os.listdir(scratch_path):
        if not os.path.isdir(os.path.join(scratch_path, speaker_dir)):
            continue
        
        for filename in os.listdir(os.path.join(scratch_path, speaker_dir)):
            if filename.endswith(".csv"):
                inputs_list.append((scratch_path, speaker_dir, filename))


    with Pool(
        processes=n_workers,
        initializer=worker_init,
        initargs=(speaker_info_df,) # Pass the DataFrame here
    ) as p:
        results = p.starmap(csv_single_process, inputs_list)

    with open(output_item_file, 'w+') as outfile:
        # Write the header to the item file
        outfile.write("#file onset offset #phone prev-phone next-phone speaker accent gender\n")
        for result in results:
            if (result is None) and (len(result) > 0):
                continue
            for res in result:
                outfile.write(res)

    print(f"Conversion complete. Item file saved to {output_item_file}")

# --- Example Usage ---
if __name__ == "__main__":
    alignment_base_folder = sys.argv[1] 
    output_file_name = sys.argv[2]
    n_workers = int(sys.argv[3])

    # Run the conversion
    speaker_df = read_speaker_info()
    convert_alignment_csv_to_item_file(alignment_base_folder, output_file_name, speaker_df, n_workers=n_workers)
