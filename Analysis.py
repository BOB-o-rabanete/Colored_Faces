import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#---------------------------------------------------------------------

#-----------------#
#| EXACT NUMBERS |#
#-----------------#

def DfComparison(
    og_df: pd.DataFrame,
    color_df: pd.DataFrame,
    col_name: list[str],
    score_cols: list[str] = ["y_mediapipe", "y_dlib_hog", "y_mtcnn"],
) -> pd.DataFrame:
    """
    Description:
        Given two dataframes with same col_name and score_cols,
        calculates the percentage of binary values for each score_cols per combination of col_name uniquevalues,
        calculates the diference of scores of the same score_cols between the two datasets
        creates a dataset 
        

    ------------------
    Parameters:
        og_df: pd.DataFrame
            A dataframe that contains the results of the unaltered fotos and some extra information regarding the same.
        color_df: pd.DataFrame
            A dataframe that contains the results of the painted fotos and some extra information regarding the same.
        col_name: list[str]
            List of column names that will be aggregated to represent different x-values.
        score_cols: list[str], optional
            The columns must be binary.
            The percentage of their value per different col_list combination will be represented by the y-axis.
    -----------
    Returns:
        Score_df: pd.Dataframe
            dataset of the values of the score_cols and their differences per each unique combination of col_name
    """
    # --- Aggregate: percentage of 1s per group --- #
    og_scores = (
        og_df
        .groupby(col_name)[score_cols]
        .mean()
        .mul(100)
        .add_suffix("_og")
    )

    color_scores = (
        color_df
        .groupby(col_name)[score_cols]
        .mean()
        .mul(100)
        .add_suffix("_color")
    )

    # --- Combine results --- #
    score_df = og_scores.join(color_scores, how="inner")

    # --- Compute differences --- #
    for col in score_cols:
        score_df[f"{col}_diff"] = (
            score_df[f"{col}_color"] - score_df[f"{col}_og"]
        )

    return score_df.reset_index()



def DfColorShade(
    df: pd.DataFrame,
    col_name: list[str],
    score_cols: list[str] = ["y_mediapipe", "y_dlib_hog", "y_mtcnn"],
    color_col: str = "color",
    shade_col: str = "shade",
    og_color: str = "og",
) -> pd.DataFrame:
    """
    Description:
        Given a dataframe containing multiple color/shade variants,
        calculates the percentage of binary values for each score column
        per combination of col_name, color, and shade.

        Creates separate columns for each (color, shade) combination.
        The OG color is treated as a special baseline and appears first.
        Difference columns vs OG are also computed.
    ------------------
    Parameters:
        df: pd.DataFrame
            DataFrame containing color/shade variants and binary score columns.
        col_name: list[str]
            Columns used as grouping keys for aggregation.
        score_cols: list[str], optional
            Binary columns whose mean (×100) represents percentage scores.
        color_col : str, optional
            Column name identifying color categories.
        shade_col : str, optional
            Column name identifying shade categories.
        og_color : str, optional
            Baseline color label used for ordering and difference calculations.
            
    -----------
    Returns:
        pd.DataFrame
            A wide-format dataframe where each row corresponds to a unique
            combination of 'col_name'.
        
    """

    df = df.copy()

    # --- Create a unified color_shade label --- #
    df["color_shade"] = np.where(
        df[color_col] == og_color,
        og_color,
        df[color_col] + "_" + df[shade_col]
    )

    # --- Aggregate percentages --- #
    agg = (
        df
        .groupby(col_name + ["color_shade"])[score_cols]
        .mean()
        .mul(100)
    )

    # --- Pivot to wide format --- #
    wide = (
        agg
        .unstack("color_shade")
    )

    wide.columns = [
        f"{score}_{cs}"
        for score, cs in wide.columns
    ]

    # --- Ensure OG columns come first --- #
    og_cols = [c for c in wide.columns if c.endswith(f"_{og_color}")]
    other_cols = [c for c in wide.columns if c not in og_cols]
    wide = wide[og_cols + other_cols]

    # --- Compute differences vs OG --- #
    for score in score_cols:
        og_col = f"{score}_{og_color}"

        if og_col not in wide.columns:
            continue

        for col in wide.columns:
            if not col.startswith(score) or col == og_col:
                continue

            if (wide[col] >= wide[og_col]).any():
                diff_name = col.replace(score, f"{score}_diff")
                wide[diff_name] = wide[col] - wide[og_col]


    return wide.reset_index()






#---------#
#| PLOTS |#
#---------#


def SingleColComparison(
    og_df: pd.DataFrame,
    color_df: pd.DataFrame,
    col_name: list[str],
    score_cols: list[str] = ["y_mediapipe", "y_dlib_hog", "y_mtcnn"],
) -> None:
    """
    Description:
        Given two dataframes with same col_name and score_cols,
        calculates the percentage of binary values for each score_cols per combination of col_name uniquevalues
        plots 3 graphs (the first occupies one row, the others share the second row)
        the first plot displays per col_name, the diferent score_col results of the two dfs as points (d-og_df and o-color_df)
        the second and theird are each og_df and color_df respectivelly and display the result as side-by-side bars in a grouped bar chart.

    ------------------
    Parameters:
        og_df: pd.DataFrame
            A dataframe that contains the results of the unaltered fotos and some extra information regarding the same.
        color_df: pd.DataFrame
            A dataframe that contains the results of the painted fotos and some extra information regarding the same.
        col_name: list[str]
            List of column names that will be aggregated to represent different x-values.
        score_cols: list[str], optional
            The columns must be binary.
            The percentage of their value per different col_list combination will be represented by the y-axis.
    -----------
    Returns:
        None
            Plots 3 graphs:
            a point graph of the percentages of score_cols instances per combination of col_name, one for both dfs.
            two grouped bar graph of the percentage of score_cols instances per combination of col_name, one for each df.
    """

    clr_lst = {
        "og_df": {
            "y_mediapipe": "#5de9f3",
            "y_dlib_hog": "#a399fb",
            "y_mtcnn": "#52b2ee",
        },
        "color_df": {
            "y_mediapipe": "#0ed8e6",
            "y_dlib_hog": "#6454f3",
            "y_mtcnn": "#0099f8",
        },
    }

    def compute_percent_df(df: pd.DataFrame) -> pd.DataFrame:
        grouped = df.groupby(col_name)[score_cols].sum()
        total_count = df.groupby(col_name).size()
        return grouped.div(total_count, axis=0) * 100

    og_percent = compute_percent_df(og_df)
    color_percent = compute_percent_df(color_df)

    x_labels = og_percent.index.astype(str)
    x = np.arange(len(x_labels))
    width = 0.25

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)

    # -------- Plot 1: Scatter comparison -------- #
    ax0 = fig.add_subplot(gs[0, :])

    for col in score_cols:
        ax0.scatter(
            x,
            og_percent[col],
            label=f"{col} (og_df)",
            color=clr_lst["og_df"][col],
            marker="o",
            s=80,
            linewidths=2.5,
        )
        ax0.scatter(
            x,
            color_percent[col],
            label=f"{col} (color_df)",
            color=clr_lst["color_df"][col],
            marker="x",
            s=80,
            linewidths=2.5,
        )

    ax0.set_xticks(x)
    ax0.set_xticklabels(x_labels, rotation=45)
    ax0.set_ylabel("Percentage (%)")
    ax0.set_title("Score Comparison: Original vs Color")
    ax0.legend()
    ax0.grid(True, axis="y", alpha=0.3)

    # -------- Plot 2: Grouped bars (og_df) -------- #
    ax1 = fig.add_subplot(gs[1, 0])

    for i, col in enumerate(score_cols):
        ax1.bar(
            x + i * width,
            og_percent[col],
            width,
            label=col,
            color=clr_lst["og_df"][col],
        )

    ax1.set_xticks(x + width)
    ax1.set_xticklabels(x_labels, rotation=45)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Original DF")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    # -------- Plot 3: Grouped bars (color_df) -------- #
    ax2 = fig.add_subplot(gs[1, 1])

    for i, col in enumerate(score_cols):
        ax2.bar(
            x + i * width,
            color_percent[col],
            width,
            label=col,
            color=clr_lst["color_df"][col],
        )

    ax2.set_xticks(x + width)
    ax2.set_xticklabels(x_labels, rotation=45)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Color DF")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()



def colorShadeComparison(
    df: pd.DataFrame,
    col_name: list[str],
    score_cols: list[str] = ["y_mediapipe", "y_dlib_hog", "y_mtcnn"]
) -> None:
    """
    Description:
        Given a dataframe that has columns name 'color' and 'shade',
        calculates the percentage of values for each column in score_cols for each uniquevalue of col_name
        for each color in column 'color' (exept when color==og) plot 2 bar graphs graphs,
        one for each shade (dark, light)
        each unique value of col_name will have 3 bars, one for the corresponding values of each score_cols column 
        and 3 rectangles that correspond to the percentage calculated for color==og that shall aligh vertically with the bars of the corresponding values of each score_cols column

    ------------------
    Parameters:
        df: pd.DataFrame
            A dataframe that contains the columns 'color', 'shade', 'y_mediapipe', 'y_dlib_hog', 'y_mtcnn'.
        col_name: list[str]
            List of column names that will be aggregated to represent different x-values.
        score_cols: list[str], optional
            The columns must be binary.
            The percentage of their value per different col_list combination will be represented by the y-axis.
    -----------
    Returns:
        None
            for each color in column 'color' (exept when color==og) plot 2 bar graphs graphs,
            one for each shade (dark, light)
            each unique value of col_name will have 3 bars, one for the corresponding values of each score_cols column 
            and 3 rectangles that correspond to the percentage calculated for color==og that shall aligh vertically with the bars of the corresponding values of each score_cols column
    """

    clr_lst = {
        'Red': {"dark":["#C82909", "#701705", "#440E03"],
                "light": ["#F54927", "#F87C63", "#FCC6BB"],
                "og": ["#000000", "#000000", "#000000"]},
        'yellow': {"light":["#F8CB63", "#F4AE0B", "#D3F527"],
                   "dark": ["#C8A509", "#9C8107", "#839C07"],
                   "og": ["#000000", "#000000", "#000000"]},
        'green': {"light":["#98FA8F", "#38F527", "#76F527"],
                  "dark": ["#079C0A", "#09C816", "#409C07"],
                  "og": ["#000000", "#000000", "#000000"]},
        'blue': {"light":["#27F5C5", "#63DAF8", "#63F8F3"],
                 "dark": ["#26997E", "#09A2C8", "#008C88"],
                 "og": ["#000000", "#000000", "#000000"]},
        'indigo': {"light":["#7286E9", "#365AF7", "#5032F0"],
                   "dark": ["#2A41B0", "#07229D", "#3112E3"],
                   "og": ["#000000", "#000000", "#000000"]},
        'roxo': {"light":["#C567F4", "#B43EEF", "#D548E5"],
                 "dark": ["#690B98", "#4B096D", "#5E0F67"],
                 "og": ["#000000", "#000000", "#000000"]},
    }

    # OG reference
    og_df = (
        df[df["color"] == "og"]
        .groupby(col_name)[score_cols]
        .mean()
        .mul(100)
    )

    x_labels = og_df.index.astype(str)
    x = np.arange(len(x_labels))
    width = 0.25

    for color in df["color"].unique():
        if color == "og" or color not in clr_lst:
            continue

        fig, (ax_light, ax_dark) = plt.subplots(
            1, 2, figsize=(16, 7.5), sharey=True
        )

        for ax, shade in zip([ax_light, ax_dark], ["light", "dark"]):
            shade_df = df[
                (df["color"] == color) &
                (df["shade"] == shade)
            ]

            if shade_df.empty:
                ax.set_visible(False)
                continue

            vals = (
                shade_df
                .groupby(col_name)[score_cols]
                .mean()
                .mul(100)
                .reindex(og_df.index)
            )

            for i, score in enumerate(score_cols):
                ax.bar(
                    x + i * width,
                    vals[score],
                    width=width,
                    color=clr_lst[color][shade][i],
                    label=f"{score} ({shade})"
                )

                ax.bar(
                    x + i * width,
                    og_df[score],
                    width=width,
                    fill=False,                      # ← no fill
                    edgecolor=clr_lst[color]["og"][i],
                    linewidth=3.1,                   # ← noticeable thickness
                    linestyle="-",
                    label=f"{score} (og)" if i == 0 else None,
                    zorder=3
                )

            ax.set_title(f"{color} – {shade}")
            ax.set_xticks(x + width)
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
            ax.grid(axis="y", linestyle="--", alpha=0.5)

        ax_light.set_ylabel("Percentage (%)")
        ax_light.legend()

        plt.suptitle(f"{color}: Light vs Dark Comparison", fontsize=14)
        plt.tight_layout()
        plt.show()

########################################################33

def colorShadeColumn(df: pd.DataFrame, col_list: list[str], split: str = "color", color: list[str] = None, y_list: list[str] = ["y_mediapipe", "y_dlib_hog", "y_mtcnn"], shade_col: str = "shade") -> None:
    """
    Description:
        Given a dataframe, two lists of column names, and a column to split by,
        calculates the percentage of binary values for each column in y_list per combination of col_list
        for each unique value of the 'split' column and plots the results in subplots with specific color shading.
        
    ------------------
    Parameters:
        df: pd.DataFrame
            A dataframe that contains columns with the same names as the ones in both col_list and y_list.
        col_list: list[str]
            List of column names that will be aggregated to represent different x-values.
        color: list[str], optional
            A list of hex color codes to use for the bars. If not provided, default colors will be used.
        y_list: list[str], optional
            The columns must be binary.
            The percentage of their value per different col_list combination will be represented by the y-axis.
        split: str
            The column used to split the dataframe into separate subgroups, one plot for each unique value in this column.
        shade_col: str
            The column used to determine the "shade" of the color. The function will use this column to distinguish between light and dark combinations.
    
    -----------
    Returns:
        None
            Plots subplots for each unique value in the 'split' column.
    """
    
    clr_lst = {
        'Red': {"dark":["#C82909", "#701705", "#440E03"],
                "light": ["#F54927", "#F87C63", "#FCC6BB"]},
        'yellow': {"light":["#F8CB63", "#F4AE0B", "#D3F527"],
                "dark": ["#C8A509", "#9C8107", "#839C07"]},
        'green': {"light":["#98FA8F", "#38F527", "#76F527"],
                "dark": ["#079C0A", "#09C816", "#409C07"]},
        'blue': {"light":["#27F5C5", "#63DAF8", "#63F8F3"],
                "dark": ["#26997E", "#09A2C8", "#008C88"]},
        'indigo': {"light":["#7286E9", "#365AF7", "#5032F0"],
                "dark": ["#2A41B0", "#07229D", "#3112E3"]},
        'roxo': {"light":["#C567F4", "#B43EEF", "#D548E5"],
                "dark": ["#690B98", "#4B096D", "#5E0F67"]},
    }

    if split not in df.columns:
        raise ValueError(f"The column '{split}' does not exist in the dataframe.")
    
    if shade_col not in df.columns:
        raise ValueError(f"The column '{shade_col}' does not exist in the dataframe.")
    
    unique_splits = df[split].unique()
    
    fig, axes = plt.subplots(nrows=len(unique_splits), ncols=2, figsize=(15, 7.5 * len(unique_splits)))

    if len(unique_splits) == 1:
        axes = [axes]  # Make sure axes is iterable if there's only one split value
    
    for ax_row, split_value in zip(axes, unique_splits):
        shade_l = clr_lst[split_value]["light"]
        shade_d = clr_lst[split_value]["dark"]
        
        # Subset the dataframe based on the split value
        subset_df = df[df[split] == split_value]
        
        # Split the subset by 'shade_col'
        dark_subset = subset_df[subset_df[shade_col] == "dark"]
        light_subset = subset_df[subset_df[shade_col] == "light"]
        
        # Group by the col_list and y_list columns
        dark_grouped = dark_subset.groupby(col_list)[y_list].sum()
        light_grouped = light_subset.groupby(col_list)[y_list].sum()
        
        # Total counts for percentage calculation
        dark_total = dark_subset.groupby(col_list).size()
        light_total = light_subset.groupby(col_list).size()
        
        # Calculate percentage for each group
        dark_percent = dark_grouped.div(dark_total, axis=0) * 100
        light_percent = light_grouped.div(light_total, axis=0) * 100
        
        # Plot dark combinations
        dark_percent.plot(kind='bar', ax=ax_row[0], color=shade_d, width=0.8)
        ax_row[0].set_title(f"Dark Combinations for Split: {split_value}")
        ax_row[0].set_xlabel('Combinations of ' + ', '.join(col_list))
        ax_row[0].set_ylabel('Percentage (%) of Binary Values')
        ax_row[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot light combinations
        light_percent.plot(kind='bar', ax=ax_row[1], color=shade_l, width=0.8)
        ax_row[1].set_title(f"Light Combinations for Split: {split_value}")
        ax_row[1].set_xlabel('Combinations of ' + ', '.join(col_list))
        ax_row[1].set_ylabel('Percentage (%) of Binary Values')
        ax_row[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return None
