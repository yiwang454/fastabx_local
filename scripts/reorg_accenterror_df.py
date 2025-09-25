import polars as pl
import io, os
import sys

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

    final_df_reordered.write_csv(os.path.join(output_dir, "reordered_collapsed_scores.csv"))

def using_pandas_old():
    import pandas as pd
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
    import pandas as pd
    file_path = sys.argv[1]
    output_dir = sys.argv[2]
    # 1. Load the data frame from a CSV file
    df = pd.read_csv(file_path)
    print("Original DataFrame:")
    print(df.head())
    print("-" * 30)

    # 2. Filter out all rows with size less than 20
    # We use .copy() to avoid a SettingWithCopyWarning later.
    df_filtered = df[df['size'] >= 20].copy()

    # 3. Eliminate the size column
    df_filtered.drop('size', axis=1, inplace=True)
    
    print("DataFrame after filtering and dropping 'size' column:")
    print(df_filtered.head())
    print("-" * 30)

    # 4. Make all possible values in 'accent_b' as column names
    # Using groupby().unstack() for a more robust pivot operation.
    df_grouped = df_filtered.groupby(['accent', '#phone', 'accent_b'])['score'].mean()
    df_pivoted = df_grouped.unstack(level='accent_b').reset_index()

    # Clean up the column names after pivot if needed
    df_pivoted.columns.name = None
    
    print("Pivoted DataFrame:")
    print(df_pivoted)
    print("-" * 30)

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

    df_sorted.to_csv(os.path.join(output_dir, "reordered_collapsed_scores_top5.csv"), index=False)


using_pandas()