import pandas as pd
import os, sys, json

item_file = sys.argv[1]
split_path = sys.argv[2]
item_output = sys.argv[3]
# Define column names
column_names = ['file', 'onset', 'offset', 'phone', 'prev-phone', 'next-phone', 'speaker', 'accent', 'gender']

# Create a DataFrame from the string data, skipping the first row as it's a comment
# We'll use the user provided column names as reference
# with open(item_file, "r") as item_read:
#     lines = item_read.readlines()[1:]
dtype_dict = {
    'onset': str,
    'offset': str
}

# Read the CSV with the dtype specified to avoid the warning
df = pd.read_csv(
    item_file,
    sep=' ',
    names=column_names,
    dtype=dtype_dict,
)
with open(split_path, "r") as split_r:
    accent_to_speaker = json.load(split_r)

speaker_to_accent = {speaker: accent for accent, speakers in accent_to_speaker.items() for speaker in speakers}
print(speaker_to_accent)

df['onset'] = pd.to_numeric(df['onset'], errors='coerce')
df['offset'] = pd.to_numeric(df['offset'], errors='coerce')

df_copy = df[df['speaker'].isin(speaker_to_accent.keys())].copy()
df_copy['accent'] = df_copy['speaker'].map(speaker_to_accent)

# The original format has space separators. However, tab is a better separator to avoid issues with values having spaces.
df_copy.to_csv(item_output, sep=' ', index=False)

print("The updated data has been saved to 'updated_phones.txt'.")
print("First few rows of the updated data:", df_copy.head())