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

