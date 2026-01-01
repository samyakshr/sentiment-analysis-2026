#!/usr/bin/env python3
"""
Sentiment Analysis Script for Survey Data
Run this script locally to analyze survey responses.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import torch
from tqdm import tqdm
import warnings
import sys
from pathlib import Path
import argparse

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_model(model_name='cardiffnlp/twitter-roberta-base-sentiment-latest'):
    """Load the sentiment analysis model."""
    print("Loading sentiment analysis model...")
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU (CUDA)' if device == 0 else 'CPU'}")
    
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device,
        return_all_scores=True
    )
    print("Model loaded successfully!")
    return sentiment_pipeline


def load_data(file_path):
    """Load survey data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully! Found {len(df)} responses.")
        print(f"Columns in dataset: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Please provide a valid CSV file path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)


def identify_text_column(df):
    """Identify the column containing text responses."""
    for col in ['text', 'response', 'comment', 'feedback', 'answer']:
        if col in df.columns:
            return col
    
    print("Available columns:", list(df.columns))
    print("Error: Could not find a text column. Expected one of: text, response, comment, feedback, answer")
    sys.exit(1)


def analyze_sentiment(sentiment_pipeline, texts):
    """Analyze sentiment for a list of texts."""
    print("Analyzing sentiment for all responses...")
    print("This may take a few minutes depending on the number of responses...")
    
    sentiments = []
    scores = []
    
    for text in tqdm(texts, desc="Processing responses"):
        try:
            # Get sentiment scores (limit to 512 characters for model input)
            result = sentiment_pipeline(str(text)[:512])
            
            # Extract the highest confidence sentiment
            best_sentiment = max(result[0], key=lambda x: x['score'])
            sentiments.append(best_sentiment['label'])
            scores.append(best_sentiment['score'])
        except Exception as e:
            print(f"\nError processing text: {str(e)}")
            sentiments.append('ERROR')
            scores.append(0.0)
    
    return sentiments, scores


def map_sentiment_labels(sentiments):
    """Map model labels to readable format."""
    label_mapping = {
        'LABEL_0': 'NEGATIVE',
        'LABEL_1': 'NEUTRAL',
        'LABEL_2': 'POSITIVE'
    }
    return [label_mapping.get(s, s) for s in sentiments]


def generate_summary(df_clean):
    """Generate and print summary statistics."""
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total Responses Analyzed: {len(df_clean)}")
    print(f"\nSentiment Distribution:")
    
    sentiment_counts = df_clean['sentiment_label'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df_clean)) * 100
        print(f"  {sentiment}: {count} ({percentage:.2f}%)")
    
    print(f"\nAverage Sentiment Score: {df_clean['sentiment_score'].mean():.4f}")
    print(f"Median Sentiment Score: {df_clean['sentiment_score'].median():.4f}")
    
    positive_count = len(df_clean[df_clean['sentiment_label'] == 'POSITIVE'])
    negative_count = len(df_clean[df_clean['sentiment_label'] == 'NEGATIVE'])
    
    if negative_count > 0:
        pos_neg_ratio = positive_count / negative_count
        print(f"\nPositive to Negative Ratio: {pos_neg_ratio:.2f}")
    
    print("=" * 60)


def create_visualizations(df_clean, output_dir='.'):
    """Create and save visualizations."""
    print("\nGenerating visualizations...")
    
    sentiment_counts = df_clean['sentiment_label'].value_counts()
    colors = {'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c', 'NEUTRAL': '#95a5a6'}
    
    # Figure 1: Pie chart and bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    pie_colors = [colors.get(sent, '#3498db') for sent in sentiment_counts.index]
    axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                autopct='%1.1f%%', colors=pie_colors, startangle=90)
    axes[0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
    
    axes[1].bar(sentiment_counts.index, sentiment_counts.values,
                color=[colors.get(sent, '#3498db') for sent in sentiment_counts.index])
    axes[1].set_title('Sentiment Counts', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Number of Responses')
    axes[1].set_xlabel('Sentiment')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sentiment_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: sentiment_distribution.png")
    plt.close()
    
    # Figure 2: Score distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(df_clean['sentiment_score'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_title('Distribution of Sentiment Confidence Scores', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Confidence Score')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(df_clean['sentiment_score'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df_clean["sentiment_score"].mean():.3f}')
    axes[0].legend()
    
    sentiment_order = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
    sentiment_order = [s for s in sentiment_order if s in df_clean['sentiment_label'].values]
    df_clean_sorted = df_clean[df_clean['sentiment_label'].isin(sentiment_order)]
    
    box_data = [df_clean_sorted[df_clean_sorted['sentiment_label'] == sent]['sentiment_score'].values
                for sent in sentiment_order]
    axes[1].boxplot(box_data, labels=sentiment_order)
    axes[1].set_title('Sentiment Score Distribution by Category', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Confidence Score')
    axes[1].set_xlabel('Sentiment')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sentiment_scores.png', dpi=300, bbox_inches='tight')
    print("Saved: sentiment_scores.png")
    plt.close()


def export_results(df_clean, text_column, output_dir='.'):
    """Export results to CSV and generate text report."""
    timestamp = pd.Timestamp.now().strftime("%Y%m%d")
    
    # Export CSV
    csv_file = f'{output_dir}/sentiment_analysis_results_{timestamp}.csv'
    df_clean.to_csv(csv_file, index=False)
    print(f"\nResults exported to: {csv_file}")
    
    # Generate text report
    report_file = f'{output_dir}/sentiment_report_{timestamp}.txt'
    with open(report_file, 'w') as f:
        f.write("SENTIMENT ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Responses Analyzed: {len(df_clean)}\n\n")
        f.write("Sentiment Distribution:\n")
        sentiment_counts = df_clean['sentiment_label'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df_clean)) * 100
            f.write(f"  {sentiment}: {count} ({percentage:.2f}%)\n")
        f.write(f"\nAverage Sentiment Score: {df_clean['sentiment_score'].mean():.4f}\n")
        f.write(f"Median Sentiment Score: {df_clean['sentiment_score'].median():.4f}\n")
        
        positive_count = len(df_clean[df_clean['sentiment_label'] == 'POSITIVE'])
        negative_count = len(df_clean[df_clean['sentiment_label'] == 'NEGATIVE'])
        if negative_count > 0:
            pos_neg_ratio = positive_count / negative_count
            f.write(f"\nPositive to Negative Ratio: {pos_neg_ratio:.2f}\n")
    
    print(f"Summary report saved to: {report_file}")


def main():
    """Main function to run sentiment analysis."""
    parser = argparse.ArgumentParser(description='Analyze sentiment of survey responses')
    parser.add_argument('data_file', nargs='?', default='survey_data.csv',
                       help='Path to CSV file containing survey data (default: survey_data.csv)')
    parser.add_argument('--output-dir', '-o', default='.',
                       help='Directory to save output files (default: current directory)')
    parser.add_argument('--model', '-m', default='cardiffnlp/twitter-roberta-base-sentiment-latest',
                       help='Hugging Face model name (default: cardiffnlp/twitter-roberta-base-sentiment-latest)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SENTIMENT ANALYSIS TOOL")
    print("=" * 60)
    
    # Load model
    sentiment_pipeline = load_model(args.model)
    
    # Load data
    df = load_data(args.data_file)
    
    # Identify text column
    text_column = identify_text_column(df)
    print(f"Using column '{text_column}' for sentiment analysis")
    
    # Clean data
    df_clean = df[df[text_column].notna() & (df[text_column].str.strip() != '')].copy()
    print(f"Cleaned data: {len(df_clean)} valid responses (removed {len(df) - len(df_clean)} empty responses)")
    
    # Analyze sentiment
    sentiments, scores = analyze_sentiment(sentiment_pipeline, df_clean[text_column])
    
    # Add results to dataframe
    df_clean['sentiment'] = sentiments
    df_clean['sentiment_score'] = scores
    df_clean['sentiment_label'] = map_sentiment_labels(sentiments)
    
    # Generate summary
    generate_summary(df_clean)
    
    # Create visualizations
    create_visualizations(df_clean, args.output_dir)
    
    # Export results
    export_results(df_clean, text_column, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check the output files in:", args.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()

