# Sentiment Analysis Model 2026

A professional sentiment analysis tool for analyzing survey responses using state-of-the-art RoBERTa-based transformer models. This project is designed for monthly analysis of company survey data.

## Features

- **State-of-the-Art Model**: Uses `cardiffnlp/twitter-roberta-base-sentiment-latest`, a RoBERTa-based model fine-tuned for sentiment analysis
- **Comprehensive Analysis**: Provides sentiment classification (Positive, Negative, Neutral) with confidence scores
- **Visual Reports**: Generates pie charts, bar charts, and distribution plots
- **Export Capabilities**: Exports results to CSV and generates text reports
- **Easy to Use**: Simple Jupyter notebook interface - just update the data file path and run

This project uses a **pretrained transformer model** (RoBERTa) rather than a lexicon-based approach because:
- Higher accuracy on nuanced text
- Better understanding of context
- Handles informal language and slang
- More reliable for survey data with varied writing styles

## Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd sentiment-analysis-2026
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch Jupyter Notebook:
```bash
jupyter notebook
```

5. Open `sentiment_analysis.ipynb` and follow the instructions in the notebook.

**New to Jupyter?** See `JUPYTER_GUIDE.md` for detailed step-by-step instructions!

## Usage

You can run the analysis in two ways:

### Option 1: Python Script (Recommended for Monthly Reports)

Simply run the script from the command line:

```bash
python sentiment_analysis.py survey_data.csv
```

Or with custom output directory:

```bash
python sentiment_analysis.py survey_data.csv --output-dir ./results
```

The script will:
- Load your data
- Analyze sentiment
- Generate visualizations (saved as PNG files)
- Export results to CSV and text report

### Option 2: Jupyter Notebook (For Interactive Analysis)

1. **Prepare your data**: 
   - Create a CSV file with your survey responses
   - The file should have a column named `text`, `response`, `comment`, `feedback`, or `answer`
   - See `survey_data_template.csv` for an example format

2. **Update the notebook**:
   - Open `sentiment_analysis.ipynb` in Jupyter
   - Update the `DATA_FILE` variable with your CSV file path
   - Run all cells (Cell → Run All)

3. **Review results**:
   - View visualizations in the notebook
   - Check exported CSV file: `sentiment_analysis_results_YYYYMMDD.csv`
   - Read summary report: `sentiment_report_YYYYMMDD.txt`

## Data Format

Your CSV file should have at least one column containing text responses. Supported column names:
- `text`
- `response`
- `comment`
- `feedback`
- `answer`

Additional columns (date, respondent_id, etc.) are optional and will be preserved in the output.

Example:
```csv
text,date,respondent_id
"I really enjoyed the workshop!",2024-01-15,RES001
"The session was okay.",2024-01-15,RES002
```

## Output

The analysis generates:

1. **Summary Statistics**: Total responses, sentiment distribution, average scores
2. **Visualizations**: 
   - Sentiment distribution pie chart
   - Sentiment counts bar chart
   - Confidence score histograms
   - Box plots by sentiment category
3. **Sample Responses**: Top examples for each sentiment category
4. **CSV Export**: Full dataset with sentiment labels and scores
5. **Text Report**: Summary report in plain text format

## Performance

- **Model**: RoBERTa-base (125M parameters)
- **Speed**: ~100-200 responses per minute (CPU), faster on GPU
- **Accuracy**: State-of-the-art performance on sentiment classification tasks
- **Input Limit**: 512 characters per response (longer texts are truncated)

## Privacy & Security

⚠️ **Important**: This project is designed for local execution. Your data never leaves your machine.

- All processing happens locally
- No data is sent to external services
- Model is downloaded once and cached locally
- Survey data files are excluded from git (see `.gitignore`)

## Monthly Workflow

1. Export survey responses to CSV format
2. Place CSV file in project directory
3. Run the script: `python sentiment_analysis.py survey_data.csv`
4. Review generated visualizations (PNG files) and exported reports
5. Archive results for historical tracking

**Quick one-liner for monthly reports:**
```bash
python sentiment_analysis.py monthly_survey_jan2024.csv --output-dir ./reports/jan2024
```

## Requirements

See `requirements.txt` for full list. Key dependencies:
- `torch`: PyTorch for model inference
- `transformers`: Hugging Face transformers library
- `pandas`: Data manipulation
- `matplotlib` & `seaborn`: Visualizations
- `jupyter`: Notebook interface

## Troubleshooting

**Model download issues**: The model will be downloaded automatically on first run (~500MB). Ensure you have internet connection and sufficient disk space.

**Memory issues**: If processing large datasets, consider processing in batches or using a machine with more RAM.

**CUDA/GPU**: The notebook automatically detects and uses GPU if available. No additional configuration needed.

## License

This project is open source. The underlying model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) is available under its respective license from Hugging Face.

## Contributing

Feel free to submit issues or pull requests for improvements!

## Author

Sammy's Sentiment Analysis Model 2026
