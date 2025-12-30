import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#---------------------------------------------------------------------

#-------------#
#| BAR PLOTS |#
#-------------#
def generalScoreModel(df: pd.DataFrame, col_list: list[str], color: list[str] = None, y_list: list[str] = ["y_mediapipe", "y_dlib_hog", "y_mtcnn"]) -> None:
    """
    Description:
        Given a dataframe and two lists of column names,
        calculates the percentage of binary values for each column in y_list per combination of col_list
        and plots the result as side-by-side bars in a grouped bar chart.

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
    -----------
    Returns:
        None
            Plots a grouped bar graph of the percentage of y_list instances per combination of col_list.
    """
    #color_l = ["#73E048", "#4AF7E9", "#E04FFF"]
    if color is not None:
        # Fail check: the number of colors should match the number of y_list columns
        if len(color) != len(y_list):
            raise ValueError(f"The number of colors ({len(color)}) must match the number of y_list columns ({len(y_list)}).")
        colors = color
    else:
        colors = None  
    
    grouped_df = df.groupby(col_list)[y_list].sum()

    total_count = df.groupby(col_list).size()  # Total number of rows for each combination of col_list
    percent_df = grouped_df.div(total_count, axis=0) * 100  # Percentage calculation

    #################################
    p_percent_df = pd.DataFrame() 
    diff_columns = []
    for i in range(len(y_list)):
        for j in range(i + 1, len(y_list)):
            col_name = f'diff_{y_list[i]}_vs_{y_list[j]}'
            diff_columns.append(col_name)
            p_percent_df[col_name] = percent_df[y_list[i]] - percent_df[y_list[j]]
    ###################################
    print("\nPercentage Table (Percentage of Binary Values per Combination of col_list and y_list):")
    print(percent_df)
    print("\nPercentage Table (Difference between 2 diferent columns):")
    print(p_percent_df)

    # ----- PLOT ----- #
    ax = percent_df.plot(kind='bar', figsize=(12, 7.5), color=colors, width=0.8)  # `width=0.8` makes space between bars
    
    ax.set_xlabel('Combinations of ' + ', '.join(col_list))
    ax.set_ylabel('Percentage (%) of Binary Values')
    ax.set_title('Grouped Bar Chart of Y-List Columns per Combination of X-List Columns')

    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()
    
    return None




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
