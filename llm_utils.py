import pandas as pd
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Constants
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Open-access model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 512

# Global variables to store model and tokenizer
model = None
tokenizer = None

def initialize_model():
    """Initialize the model and tokenizer"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            
            # Load model without quantization to avoid compatibility issues
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,  # Use regular float instead of bfloat16
                device_map=DEVICE
                # Removed quantization parameters
            )
            
            return True
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return False
    
    return True

def get_dataframe_info(df):
    """Get information about the dataframe to include in the prompt"""
    # Basic dataframe info
    num_rows, num_cols = df.shape
    
    # Column names and sample values
    columns_info = []
    for col in df.columns:
        # Get column type
        col_type = df[col].dtype
        
        # Get sample values (non-null)
        non_null_values = df[col].dropna()
        if len(non_null_values) > 0:
            sample_values = non_null_values.iloc[:3].tolist()
            sample_str = ", ".join([str(val) for val in sample_values])
        else:
            sample_str = "No non-null values"
        
        columns_info.append(f"- {col} ({col_type}): Example values: {sample_str}")
    
    # Combine all info
    info_text = f"Dataset has {num_rows} rows and {num_cols} columns.\n\n"
    info_text += "Column information:\n" + "\n".join(columns_info)
    
    return info_text

def create_prompt(question, df, column_mapping):
    """Create a formatted prompt for the LLM"""
    # Get dataframe info
    df_info = get_dataframe_info(df)
    
    # Determine if this is a chart request
    is_chart_request = any(chart_type in question.lower() for chart_type in [
        "chart", "plot", "graph", "visualize", "visualization", "histogram", "bar", "pie", "scatter", "line"
    ])
    
    # Create the complete prompt - adjusted for TinyLlama's format
    prompt = f"""<human>
You are a helpful data analysis assistant. Your task is to answer questions about a dataset.
    
Dataset Information:
{df_info}

The original column names have been cleaned to remove spaces and special characters. 
Here's the mapping from original to cleaned names:
{', '.join([f"'{k}' -> '{v}'" for k, v in column_mapping.items()])}

User Question: {question}

Instructions:
1. Provide a clear, direct answer to the question based on the dataset information.
2. If the question asks for a chart or visualization:
   - ONLY specify which chart type would be best (bar chart, histogram, etc.)
   - ONLY mention which columns should be used for the visualization
   - DO NOT provide any code snippets
   - DO NOT explain how to create the chart
   - Be very brief and direct
3. If asked for data that requires filtering or specific conditions, specify the condition clearly.
4. If the question cannot be answered from the given dataset, explain why.
5. Keep your response concise and focused.
</human>

<assistant>
"""
    
    return prompt

def extract_chart_suggestion(response):
    """
    Extract chart suggestion from LLM response
    Returns a dict with chart type and relevant columns if found
    """
    # Look for common chart type mentions
    chart_patterns = {
        "bar chart": r"bar\s*chart",
        "histogram": r"histogram",
        "line chart": r"line\s*chart",
        "scatter plot": r"scatter\s*plot",
        "pie chart": r"pie\s*chart",
        "box plot": r"box\s*plot"
    }
    
    chart_info = {"type": None, "columns": []}
    
    # Check for chart type mentions
    for chart_type, pattern in chart_patterns.items():
        if re.search(pattern, response, re.IGNORECASE):
            chart_info["type"] = chart_type
            break
    
    # If chart type found, try to extract columns
    if chart_info["type"]:
        # This is a simplified column extraction and would need to be more robust in production
        # Look for column names mentioned after the chart type
        full_text_after_chart = response[response.lower().find(chart_info["type"]):]
        
        # Simple heuristic: look for words that might be column names
        words = re.findall(r'\b[a-z][a-z0-9_]*\b', full_text_after_chart.lower())
        chart_info["columns"] = words[:3]  # Take up to 3 potential column names
    
    return chart_info

def generate_response(question, df, column_mapping):
    """Generate a response from the LLM for the given question"""
    # Initialize model if needed
    if not initialize_model():
        return "Failed to initialize the language model. Please try again later."
    
    # Create the prompt
    prompt = create_prompt(question, df, column_mapping)
    
    try:
        # Encode the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Generate response
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # Decode the response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract just the assistant's reply (after the prompt)
        # Adjusted for TinyLlama format
        if "<assistant>" in response:
            response = response.split("<assistant>")[-1].strip()
        
        return response
    
    except Exception as e:
        return f"Error generating response: {str(e)}" 