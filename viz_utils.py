import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from llm_utils import extract_chart_suggestion

# Set the style for the visualizations
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

def generate_chart(response, df):
    """
    Generate a chart based on the LLM response
    Returns a matplotlib figure or None if no chart is needed
    """
    # Extract chart suggestion from response
    chart_info = extract_chart_suggestion(response)
    
    # If no chart type detected, return None
    if not chart_info["type"]:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check if suggested columns exist in the dataframe
    valid_columns = [col for col in chart_info["columns"] if col in df.columns]
    
    # If no valid columns found, try to infer from dataframe
    if not valid_columns:
        # For numerical charts, select numeric columns
        if chart_info["type"] in ["histogram", "line chart", "scatter plot", "box plot"]:
            valid_columns = df.select_dtypes(include=['number']).columns.tolist()[:2]
        else:  # For categorical charts, include at least one categorical
            num_cols = df.select_dtypes(include=['number']).columns.tolist()[:1]
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()[:1]
            valid_columns = cat_cols + num_cols
    
    # If still no valid columns, return None
    if not valid_columns:
        return None
    
    # Generate the appropriate chart based on chart_info["type"]
    try:
        if chart_info["type"] == "bar chart":
            if len(valid_columns) >= 2:
                # If we have at least 2 columns, assume first is category and second is value
                sns.barplot(x=valid_columns[0], y=valid_columns[1], data=df, ax=ax)
            else:
                # Single column - show value counts
                df[valid_columns[0]].value_counts().plot(kind='bar', ax=ax)
            
            ax.set_title(f"Bar Chart of {valid_columns[0]}")
            ax.set_xlabel(valid_columns[0])
            ax.tick_params(axis='x', rotation=45)
        
        elif chart_info["type"] == "histogram":
            sns.histplot(data=df, x=valid_columns[0], kde=True, ax=ax)
            ax.set_title(f"Histogram of {valid_columns[0]}")
            ax.set_xlabel(valid_columns[0])
        
        elif chart_info["type"] == "line chart":
            if len(valid_columns) >= 2:
                # Sort by the x column for better line charts
                sorted_df = df.sort_values(by=valid_columns[0])
                ax.plot(sorted_df[valid_columns[0]], sorted_df[valid_columns[1]])
                ax.set_title(f"Line Chart of {valid_columns[1]} by {valid_columns[0]}")
                ax.set_xlabel(valid_columns[0])
                ax.set_ylabel(valid_columns[1])
            else:
                # Single column - show values over index
                df[valid_columns[0]].plot(ax=ax)
                ax.set_title(f"Line Chart of {valid_columns[0]}")
        
        elif chart_info["type"] == "scatter plot":
            if len(valid_columns) >= 2:
                sns.scatterplot(x=valid_columns[0], y=valid_columns[1], data=df, ax=ax)
                ax.set_title(f"Scatter Plot of {valid_columns[1]} vs {valid_columns[0]}")
                ax.set_xlabel(valid_columns[0])
                ax.set_ylabel(valid_columns[1])
        
        elif chart_info["type"] == "pie chart":
            # Get value counts for the first column
            counts = df[valid_columns[0]].value_counts()
            # Limit to top 10 categories for readability
            if len(counts) > 10:
                counts = counts.head(10)
            counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_title(f"Pie Chart of {valid_columns[0]} Distribution")
            ax.set_ylabel('')
        
        elif chart_info["type"] == "box plot":
            if len(valid_columns) >= 2:
                # If we have 2 columns, use the first as category and second as value
                sns.boxplot(x=valid_columns[0], y=valid_columns[1], data=df, ax=ax)
                ax.set_title(f"Box Plot of {valid_columns[1]} by {valid_columns[0]}")
                ax.tick_params(axis='x', rotation=45)
            else:
                # Single column - simple box plot
                sns.boxplot(y=valid_columns[0], data=df, ax=ax)
                ax.set_title(f"Box Plot of {valid_columns[0]}")
        
        # Adjust layout and return the figure
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Error generating chart: {str(e)}")
        return None 