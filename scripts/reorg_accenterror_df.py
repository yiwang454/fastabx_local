import polars as pl
import io, os
import sys
import pandas as pd
import argparse

def using_polar():
    item_path = sys.argv[1]
    output_dir = sys.argv[2]
    schema_customize = {
        "accent": pl.String,
        "#phone": pl.String,
        "accent_b": pl.String,
        "score": pl.String,
        "size": pl.String,
    }
    df = pl.read_csv(item_path, schema=schema_customize)
    df = df.with_columns(df["score"].str.to_decimal(), df["size"].str.to_decimal())
    df_filtered = df.filter(pl.col("size") >= 20)
    df_no_size = df_filtered.drop("size")

    df_pivoted = df_no_size.pivot(
        index=["accent", "#phone"],
        columns="accent_b",
        values="score",
        aggregate_function=pl.first()
    )

    accent_b_cols = df_pivoted.columns[2:]  # Exclude 'accent' and '#phone'
    df_pivoted = df_pivoted.with_columns(
        pl.mean_horizontal(pl.col(accent_b_cols)).alias("mean_score")
    )

    # 5. Sort the DataFrame by "accent" and then by "mean_score"
    final_df = df_pivoted.sort(["accent", "mean_score"])
    accent_b_cols = sorted([c for col in final_df.columns if c not in ["accent", "#phone", "mean_score"]])
    print("accent_b_cols, ", accent_b_cols)
    new_column_order = ["accent", "#phone", "mean_score"] + accent_b_cols
    final_df_reordered = final_df.select(new_column_order)

    print("reorg result, ", os.path.join(output_dir, "reordered_collapsed_scores.csv"))
    final_df_reordered.write_csv(os.path.join(output_dir, "reordered_collapsed_scores.csv"))

def using_pandas_old():
    item_path = sys.argv[1]
    output_dir = sys.argv[2]
    # schema_customize = {
    #     "accent": pl.String,
    #     "#phone": pl.String,
    #     "accent_b": pl.String,
    #     "score": pl.String,
    #     "size": pl.String,
    # }
    df = pd.read_csv(item_path)
    df_filtered = df[df['size'] >= 20]
    df_no_size = df_filtered.drop('size', axis=1)

    # 3. Pivot the DataFrame
    df_pivoted = df_no_size.pivot_table(
        index=['accent', '#phone'],
        columns='accent_b',
        values='score'
    ).reset_index()

    df_pivoted['mean_score'] = df_pivoted.mean(axis=1)

    # 5. Sort the DataFrame by "accent" and then by "mean_score"
    final_df = df_pivoted.sort_values(by=['accent', 'mean_score'])

    # 6. Reorder the columns to match the desired format
    accent_b_cols = sorted([col for col in final_df.columns if col not in ['accent', '#phone', 'mean_score']])
    new_column_order = ['accent', '#phone', 'mean_score'] + accent_b_cols
    final_df = final_df[new_column_order]

    final_df_reordered = final_df.sort_values(by=['accent', 'mean_score'], ascending=[True, True])
    print(final_df_reordered[:10])

    final_df_reordered.to_csv(os.path.join(output_dir, "reordered_collapsed_scores.csv"))

def using_pandas():
    file_path = sys.argv[1]
    output_dir = sys.argv[2]
    # 1. Load the data frame from a CSV file
    df = pd.read_csv(file_path)

    # 2. Filter out all rows with size less than 20
    # We use .copy() to avoid a SettingWithCopyWarning later.
    df_filtered = df[df['size'] >= 20].copy()
    # 3. Eliminate the size column
    df_filtered.drop('size', axis=1, inplace=True)
    
    print("DataFrame after filtering and dropping 'size' column:", df_filtered.head())

    # 4. Make all possible values in 'accent_b' as column names
    # Using groupby().unstack() for a more robust pivot operation.
    df_grouped = df_filtered.groupby(['accent', '#phone', 'accent_b'])['score'].mean()
    df_pivoted = df_grouped.unstack(level='accent_b').reset_index()

    # Clean up the column names after pivot if needed
    df_pivoted.columns.name = None

    # 5. Calculate the mean score among all 'accent_b' for each row
    # Identify the new accent columns to calculate the mean
    accent_cols = [col for col in df_pivoted.columns if col not in ['accent', '#phone']]
    df_pivoted['mean_score'] = df_pivoted[accent_cols].mean(axis=1, skipna=True)

    # 6. Sort the data frame by "accent" and then "mean_score"
    df_sorted = df_pivoted.sort_values(by=['accent', 'mean_score'], ascending=[True, True])
    
    # Fill top_10_per_accent = final_df.groupby(level='accent').head(10) values with 0 for better readability in the final output, if desired.
    df_sorted = df_sorted.groupby('accent').head(5)
    df_sorted = df_sorted.fillna('NaN')

    # 7. Reorder the columns to a specified order
    accent_b_cols = sorted([col for col in df_sorted.columns if col not in ['accent', '#phone', 'mean_score']])
    new_column_order = ['accent', '#phone', 'mean_score'] + accent_b_cols
    df_sorted = df_sorted[new_column_order]

    print("reorg result, ", os.path.join(output_dir, "reordered_collapsed_scores_top5.csv"))
    df_sorted.to_csv(os.path.join(output_dir, "reordered_collapsed_scores_top5.csv"), index=False)

def selected_scores(file_path, select_mode, select_param):
    # to defind select_mode by choices={"threshold", "ratio", "number"} 
    df = pd.read_csv(file_path)

    df_filtered = df[df['size'] >= 20].copy()
    df_filtered.drop('size', axis=1, inplace=True)
    
    df_sorted = df_filtered.sort_values(by=['score'], ascending=[True])
    if select_mode == "number":
        select_param = int(select_param)
        df_selected = df_sorted[:select_param]
    elif select_mode == "ratio":
        number = int(len(df_sorted) * float(select_param))
        df_selected = df_sorted[:number]
    elif select_mode == "threshold":
        df_selected = df_sorted[df_sorted["score"] <= select_param]
    elif select_mode == "combination":
        assert type(select_param) is pd.core.frame.DataFrame
        df_selected = pd.merge(df_sorted, select_param, 
                    on=['accent', 'accent_b', '#phone'], how='inner')
    else:
        raise ValueError("wrong select_mode value")

    df_selected.reset_index(drop=True, inplace=True)
    mean_abx_score = df_selected['score'].mean()
    if select_mode == "combination":
        # print(mean_abx_score) # "selected_score,", select_param, select_mode,  file_path
        pass
    else:
        print("file_path, select_mode, select_param, mean_abx_score", file_path, select_mode, select_param, mean_abx_score) # "selected_score,", select_param, select_mode,  file_path

    # df_selected.to_csv(os.path.join(output_dir, "reordered_collapsed_scores_top5.csv"), index=False)

    selected_combinations = df_selected[['accent', 'accent_b', '#phone']].drop_duplicates()

    return selected_combinations, mean_abx_score

def parse_args():
    """
    Parses command-line arguments for the file reorganization script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('file_path',type=str)
    parser.add_argument('select_mode',type=str, choices={"threshold", "ratio", "number", "combination"}, default="number")
    parser.add_argument('select_param',type=float)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # args = parse_args()
    # selected_scores(args.file_path, args.select_mode, args.select_param)
    ### Multiple results
    file_paths = [
        "/home/s2522559/datastore/vctk/hubert_feature/large_l18_mic1_abx_errors_nolevel.csv",
        "/home/s2522559/datastore/RepCodec_codecs/repcodec_hubert_large_l18_ls100/VCTK_tokens_reorg_abx_errors_nolevel.csv",
        "/home/s2522559/datastore/RepCodec_codecs/repcodec_hubert_large_l18_ls100_released/VCTK_tokens_reorg_abx_errors_nolevel.csv",
        "/home/s2522559/datastore/RepCodec_codecs/repcodec_hubert_large_l18_ls100_merge0.2vctk/VCTK_tokens_reorg_abx_errors_nolevel.csv",
        "/home/s2522559/datastore/RepCodec_codecs/repcodec_hubert_large_l18_ls100_merge0.5vctk/VCTK_tokens_reorg_abx_errors_nolevel.csv",
        "/home/s2522559/datastore/RepCodec_codecs/repcodec_hubert_large_l18_ls100_merge1.0vctk/VCTK_tokens_reorg_abx_errors_nolevel.csv",
    ]

    for select_param in [100, 200, 300, 400, 600, 900]:
        selected_combinations_first, _ = selected_scores(file_paths[0], "number", select_param)
        other_scores = []
        for f in file_paths[1:]:
            _, score = selected_scores(f, "combination", selected_combinations_first)
            other_scores.append(score)
        print(" ".join([str(s) for s in other_scores]))

    for select_param in [0.025, 0.05, 0.1, 0.2]:
        selected_combinations_first, _ = selected_scores(file_paths[0], "ratio", select_param)
        other_scores = []
        for f in file_paths[1:]:
            _, score = selected_scores(f, "combination", selected_combinations_first)
            other_scores.append(score)
        print(" ".join([str(s) for s in other_scores]))


