import streamlit as st
import pandas as pd
import re
from data_utils import load_and_clean_excel
from llm_utils import generate_response
from viz_utils import generate_chart

st.set_page_config(page_title="Excel Assistant", layout="wide")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "df" not in st.session_state:
    st.session_state.df = None

if "cleaned_columns" not in st.session_state:
    st.session_state.cleaned_columns = None

if "awaiting_followup" not in st.session_state:
    st.session_state.awaiting_followup = False

def clean_response(response):
    """Clean up the LLM response by removing code blocks and unnecessary explanations"""
    # Remove any code blocks
    response = re.sub(r'```[\s\S]*?```', '', response)
    
    # Remove any inline code
    response = re.sub(r'`.*?`', '', response)
    
    # Remove lines that look like code (heuristic)
    lines = response.split('\n')
    cleaned_lines = [line for line in lines if not line.strip().startswith(('import ', 'from ', 'def ', 'class ', '#', '//', 'plt.', 'fig,', 'ax.'))]
    
    # Join the remaining lines
    cleaned_response = '\n'.join(cleaned_lines)
    
    # Remove multiple newlines
    cleaned_response = re.sub(r'\n\s*\n', '\n\n', cleaned_response)
    
    return cleaned_response.strip()

# Function to handle question submission
def handle_question(question):
    if question:
        with st.spinner("Generating response..."):
            # Check if this is a chart request
            is_chart_request = any(chart_type in question.lower() for chart_type in [
                "chart", "plot", "graph", "visualize", "visualization", "histogram", "bar", "pie", "scatter", "line"
            ])
            
            # Get response from LLM
            response = generate_response(
                question, 
                st.session_state.df,
                st.session_state.cleaned_columns
            )
            
            # Clean up the response
            cleaned_response = clean_response(response)
            
            # Generate chart if needed
            chart_fig = generate_chart(response, st.session_state.df)
            
            # For chart requests, keep the response extra minimal if a chart was generated
            if is_chart_request and chart_fig:
                # Extract only the essential information about the chart
                chart_info = re.search(r'(bar chart|histogram|line chart|scatter plot|pie chart|box plot).*?(of|between|showing|with).*?[.!]', cleaned_response, re.IGNORECASE)
                if chart_info:
                    cleaned_response = chart_info.group(0)
            
            # Save chart to session state if it exists
            if chart_fig:
                setattr(st.session_state, f"chart_for_{len(st.session_state.chat_history)}", chart_fig)
            
            # Add to history
            st.session_state.chat_history.append((question, cleaned_response))
            
            # Set awaiting followup flag
            st.session_state.awaiting_followup = True

def main():
    st.title("Excel Assistant")
    st.write("Upload your Excel file and ask questions about your data.")
    
    # Sidebar for file upload and options
    with st.sidebar:
        st.header("Upload Data")
        uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
        
        if uploaded_file is not None:
            try:
                # Process the uploaded file
                df, cleaned_columns = load_and_clean_excel(uploaded_file)
                st.session_state.df = df
                st.session_state.cleaned_columns = cleaned_columns
                st.success("File loaded successfully!")
                
                # Display column mappings
                st.subheader("Column Cleanup")
                mapping_df = pd.DataFrame({
                    "Original": list(cleaned_columns.keys()),
                    "Cleaned": list(cleaned_columns.values())
                })
                st.dataframe(mapping_df)
                
                # History section in sidebar
                st.subheader("Question History")
                if st.session_state.chat_history:
                    for i, (q, _) in enumerate(st.session_state.chat_history):
                        st.text(f"{i+1}. {q}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Main content area
    if st.session_state.df is not None:
        # Display dataframe preview
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head(10))
        
        # Create a chat container
        chat_container = st.container()
        
        # Display chat history in main area
        with chat_container:
            for idx, (question, answer) in enumerate(st.session_state.chat_history):
                st.markdown(f"**You:** {question}")
                st.markdown(f"**Assistant:** {answer}")
                
                # If a chart was generated for this response, display it
                if hasattr(st.session_state, f"chart_for_{idx}"):
                    chart_fig = getattr(st.session_state, f"chart_for_{idx}")
                    if chart_fig:
                        st.pyplot(chart_fig)
                
                st.markdown("---")
            
            # Show follow-up prompt if awaiting follow-up
            if st.session_state.awaiting_followup and st.session_state.chat_history:
                st.markdown("**Assistant:** Would you like to know anything else about your data?")
        
        # Question input using a form to handle submission properly
        with st.form(key="question_form", clear_on_submit=True):
            user_question = st.text_input("Ask a question about your data:")
            submit_button = st.form_submit_button("Ask")
            
            if submit_button and user_question:
                handle_question(user_question)
                st.experimental_rerun()
    else:
        st.info("Please upload an Excel file to get started.")

if __name__ == "__main__":
    main() 