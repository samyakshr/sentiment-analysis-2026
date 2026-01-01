# How to Use Jupyter Notebook - Quick Guide

## Step 1: Install Everything

First, make sure you have Python installed. Then install all the required packages:

```bash
# Navigate to the project folder
cd /Users/samyakshrestha/Desktop/sentiment-analysis-2026

# Install all dependencies (this includes Jupyter)
pip install -r requirements.txt
```

## Step 2: Start Jupyter Notebook

Open your terminal and run:

```bash
jupyter notebook
```

This will:
- Start a local web server
- Open your web browser automatically
- Show you a file browser interface

**Note:** Keep the terminal window open while using Jupyter!

## Step 3: Open Your Notebook

In the browser window that opened:
1. You'll see a list of files
2. Click on `sentiment_analysis.ipynb` to open it

## Step 4: Understanding the Notebook Interface

A Jupyter notebook is made up of **cells**. Each cell can contain:
- **Code** (Python code that you can run)
- **Markdown** (text/instructions - these are just for reading)

You'll see:
- **Markdown cells** (gray/white text) - These are instructions, just read them
- **Code cells** (with `In [ ]:` on the left) - These contain Python code you can run

## Step 5: Run the Notebook

### Option A: Run All Cells at Once (Easiest!)

1. Click on the **"Cell"** menu at the top
2. Select **"Run All"**
3. Wait for all cells to execute (this may take a few minutes the first time as it downloads the model)

### Option B: Run Cells One by One (Recommended for First Time)

1. Click on the first code cell (the one that says `# Import required libraries`)
2. Press **Shift + Enter** (or click the "Run" button)
3. Wait for it to finish (you'll see `In [*]:` change to `In [1]:` when done)
4. Click the next cell and press **Shift + Enter** again
5. Repeat for all cells

## Step 6: Update Your Data File

Before running the analysis cells, you need to update the data file path:

1. Find the cell that says `DATA_FILE = 'survey_data.csv'`
2. Change `'survey_data.csv'` to the path of your actual survey data file
3. For example: `DATA_FILE = '/path/to/your/survey_data.csv'`

## Step 7: View Results

After running all cells:
- **Visualizations** will appear directly in the notebook
- **CSV file** will be saved in the same folder (named like `sentiment_analysis_results_20240115.csv`)
- **Text report** will also be saved (named like `sentiment_report_20240115.txt`)

## Common Keyboard Shortcuts

- **Shift + Enter**: Run current cell and move to next
- **Ctrl + Enter**: Run current cell but stay on it
- **Esc**: Exit cell editing mode
- **Enter**: Enter cell editing mode
- **A**: Insert new cell above (when not editing)
- **B**: Insert new cell below (when not editing)
- **DD** (press D twice): Delete current cell

## Troubleshooting

### "Jupyter: command not found"
- Make sure you installed requirements: `pip install -r requirements.txt`
- Try: `python -m jupyter notebook` instead

### Browser didn't open automatically
- Look at the terminal - it will show a URL like `http://localhost:8888`
- Copy that URL and paste it into your browser

### "Kernel died" or "Out of memory"
- Close other applications to free up RAM
- Try processing smaller batches of data

### Model download is slow
- This is normal! The model is ~500MB and downloads on first run
- It only downloads once, then it's cached locally

## Stopping Jupyter

1. Close the browser tab
2. Go back to the terminal
3. Press **Ctrl + C** (or Cmd + C on Mac) twice
4. This stops the Jupyter server

## Need Help?

If you prefer not to use Jupyter, you can use the Python script instead:
```bash
python sentiment_analysis.py survey_data.csv
```

The script does the same thing but runs from the command line - no browser needed!

