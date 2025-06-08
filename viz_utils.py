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
    chart_info = extract_chart_suggestion(response)
    if not chart_info["type"]:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_columns = [col for col in chart_info["columns"] if col in df.columns]
    if not valid_columns:
        if chart_info["type"] in ["histogram", "line chart", "scatter plot", "box plot"]:
            valid_columns = df.select_dtypes(include=['number']).columns.tolist()[:2]
        else:
            num_cols = df.select_dtypes(include=['number']).columns.tolist()[:1]
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()[:1]
            valid_columns = cat_cols + num_cols
    if not valid_columns:
        return None
    
    # Generate the appropriate chart based on chart_info["type"]
    # Bar Chart
    try:
        if chart_info["type"] == "bar chart":
            if len(valid_columns) >= 2:
                sns.barplot(x=valid_columns[0], y=valid_columns[1], data=df, ax=ax)
            else:
                df[valid_columns[0]].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f"Bar Chart of {valid_columns[0]}")
            ax.set_xlabel(valid_columns[0])
            ax.tick_params(axis='x', rotation=45)
    
    # Histogram    
        elif chart_info["type"] == "histogram":
            sns.histplot(data=df, x=valid_columns[0], kde=True, ax=ax)
            ax.set_title(f"Histogram of {valid_columns[0]}")
            ax.set_xlabel(valid_columns[0])
    
    # Line Chart   
        elif chart_info["type"] == "line chart":
            if len(valid_columns) >= 2:
                sorted_df = df.sort_values(by=valid_columns[0])
                ax.plot(sorted_df[valid_columns[0]], sorted_df[valid_columns[1]])
                ax.set_title(f"Line Chart of {valid_columns[1]} by {valid_columns[0]}")
                ax.set_xlabel(valid_columns[0])
                ax.set_ylabel(valid_columns[1])
            else:
                df[valid_columns[0]].plot(ax=ax)
                ax.set_title(f"Line Chart of {valid_columns[0]}")
    
    # Scatter Plot   
        elif chart_info["type"] == "scatter plot":
            if len(valid_columns) >= 2:
                sns.scatterplot(x=valid_columns[0], y=valid_columns[1], data=df, ax=ax)
                ax.set_title(f"Scatter Plot of {valid_columns[1]} vs {valid_columns[0]}")
                ax.set_xlabel(valid_columns[0])
                ax.set_ylabel(valid_columns[1])
    
    # Pie Chart  
        elif chart_info["type"] == "pie chart":
            counts = df[valid_columns[0]].value_counts()
            if len(counts) > 10:
                counts = counts.head(10)
            counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_title(f"Pie Chart of {valid_columns[0]} Distribution")
            ax.set_ylabel('')
    
    # Box Plot    
        elif chart_info["type"] == "box plot":
            if len(valid_columns) >= 2:
                sns.boxplot(x=valid_columns[0], y=valid_columns[1], data=df, ax=ax)
                ax.set_title(f"Box Plot of {valid_columns[1]} by {valid_columns[0]}")
                ax.tick_params(axis='x', rotation=45)
            else:
                sns.boxplot(y=valid_columns[0], data=df, ax=ax)
                ax.set_title(f"Box Plot of {valid_columns[0]}")
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Error generating chart: {str(e)}")
        return None 
