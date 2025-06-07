import pandas as pd
import re
import numpy as np

def clean_column_name(col_name):
    """
    Clean and normalize column names:
    - Convert to lowercase
    - Replace spaces and special chars with underscores
    - Remove duplicate underscores
    - Remove leading/trailing underscores
    """
    if not isinstance(col_name, str):
        col_name = str(col_name)
    
    # Convert to lowercase
    col_name = col_name.lower()
    
    # Replace spaces and special chars with underscores
    col_name = re.sub(r'[^\w\s]', '_', col_name)
    col_name = re.sub(r'\s+', '_', col_name)
    
    # Remove duplicate underscores
    col_name = re.sub(r'_+', '_', col_name)
    
    # Remove leading/trailing underscores
    col_name = col_name.strip('_')
    
    return col_name

def load_and_clean_excel(file_obj):
    """
    Load Excel file and clean column names
    Returns:
    - DataFrame with cleaned column names
    - Dictionary mapping original to cleaned column names
    """
    # Read the first sheet
    df = pd.read_excel(file_obj, sheet_name=0)
    
    # Create mapping of original to cleaned column names
    column_mapping = {}
    cleaned_columns = []
    
    for col in df.columns:
        cleaned_col = clean_column_name(col)
        column_mapping[col] = cleaned_col
        cleaned_columns.append(cleaned_col)
    
    # Rename columns
    df.columns = cleaned_columns
    
    # Basic data cleaning
    # Replace empty strings with NaN
    df = df.replace('', np.nan)
    
    # Drop rows that are all NaN
    df = df.dropna(how='all')
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    return df, column_mapping

def extract_rows_based_on_condition(df, condition_str):
    """
    Attempt to extract rows based on a condition string
    (Simplified implementation - would need more complex parsing for real use)
    """
    try:
        # This is a very simplified implementation
        # In a real application, you'd need more robust parsing
        result = df.query(condition_str)
        return result
    except:
        return None 