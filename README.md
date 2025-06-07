# Excel Assistant

A Streamlit application that allows users to ask natural language questions about their Excel data and get answers using a Hugging Face LLM (TinyLlama-1.1B-Chat).

## Features

- Upload Excel files (.xlsx) with up to 500 rows and 10-20 columns
- Automatic column name cleaning and normalization
- Natural language question answering about your data
- Automatic chart generation based on question context
- Question history tracking
- Support for complex data queries

## Architecture

This project follows a modular structure:

- `app.py`: Main Streamlit application and UI
- `data_utils.py`: Excel file loading and data cleaning utilities
- `llm_utils.py`: LLM interaction and prompt handling
- `viz_utils.py`: Visualization generation based on LLM responses

## Requirements

- Python 3.8+
- CUDA-compatible GPU recommended (but not required)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```
   git clone [https://github.com/yourusername/excel-assistant](https://github.com/naveenadam/-Natural-Language-Chabot).git
   cd excel-assistant
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in your terminal (typically http://localhost:8501)

3. Upload an Excel file using the sidebar upload button

4. View the data preview and ask questions about your data

5. Explore the generated charts and answers

## Example Questions

- "What is the average age in the dataset?"
- "Show me a bar chart of sales by region"
- "What's the distribution of customer types?"
- "Show me customers under 30 years old"
- "Is there a correlation between age and income?"

## Deployment Options

### Streamlit Cloud

1. Push your code to a GitHub repository
2. Create an account on [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your GitHub repository
4. Deploy the app

### Hugging Face Spaces

1. Create an account on [Hugging Face](https://huggingface.co/)
2. Create a new Space with Streamlit as the SDK
3. Upload your code to the Space
4. Configure the requirements

## Performance Considerations

- The first query might take longer as the model is loaded into memory
- Quantization is used to reduce memory usage (4-bit)
- For large Excel files, consider pre-processing to reduce size
- Chart generation might take a few seconds depending on the complexity

## Limitations

- Handles only the first sheet of an Excel file
- Works best with clean, structured data
- Chart extraction is based on pattern matching and might not always be perfect
- Advanced statistical operations might require additional implementation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
