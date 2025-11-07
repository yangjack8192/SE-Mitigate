# Toxicity Detection and Mitigation in Software Engineering Comments

This repository contains code for detecting and mitigating toxic language in software engineering code review comments using Large Language Models (LLMs). The project evaluates multiple LLMs including GPT-4.1, Google Gemini, and DeepSeek for both toxicity detection and mitigation tasks.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Toxicity Detection](#toxicity-detection)
  - [Toxicity Mitigation](#toxicity-mitigation)
- [Dataset](#dataset)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)

## Overview

This project addresses the problem of toxic language in software engineering contexts, particularly in code review comments. We implement:

1. **Toxicity Detection**: Using GPT-4.1, Gemini, and DeepSeek to identify toxic comments in code reviews
2. **Toxicity Mitigation**: Rewriting toxic comments to be more constructive and respectful while preserving the technical content

## Features

- **Multi-model Support**: Evaluate multiple LLMs (GPT-4.1, Gemini 1.5 Flash, DeepSeek)
- **Few-shot and Zero-shot Learning**: Compare performance with and without examples
- **Progress Tracking**: Resume interrupted experiments with automatic progress saving
- **Semantic Similarity Analysis**: Measure how well mitigated text preserves original meaning using Sentence-BERT
- **Comprehensive Metrics**: Precision, recall, F1-score, latency tracking, and success rates

## Installation

### Prerequisites

- Python 3.8 or higher
- API keys for:
  - OpenAI (for GPT-4.1)
  - Google Gemini
  - DeepSeek

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SEToxicityMitigation.git
cd SEToxicityMitigation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `keys.env` file in the project root with your API keys:
```bash
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
```

## Configuration

The project uses environment variables for API key management. Ensure your `keys.env` file is properly configured and never commit it to version control (it's already in `.gitignore`).

## Usage

### Toxicity Detection

The `toxicity_detection.py` module provides functions to detect toxic comments using different LLMs.

#### Running Detection Experiments

```python
from toxicity_detection import process_dataset, calculate_metrics

# Process your dataset with all models
process_dataset(
    input_file="code-review-dataset-full.csv",
    output_file="processed-code-review-dataset-full.csv",
    save_interval=10,    # Save progress every 10 rows
    max_rows=100         # Process only first 100 rows (use -1 for all)
)

# Calculate metrics
calculate_metrics(
    processed_file="processed-code-review-dataset-full.csv",
    ground_truth_column="is_toxic"
)
```

#### Individual Model Detection

```python
from toxicity_detection import (
    detect_toxicity_with_gpt41,
    detect_toxicity_with_gemini,
    detect_toxicity_with_deepseek
)

comment = "Your code is terrible and you should feel bad"

# Few-shot detection
prediction, latency = detect_toxicity_with_deepseek(comment, few_shot=True)
print(f"Toxic: {prediction}, Latency: {latency:.2f}s")

# Zero-shot detection
prediction, latency = detect_toxicity_with_deepseek(comment, few_shot=False)
print(f"Toxic: {prediction}, Latency: {latency:.2f}s")
```

### Toxicity Mitigation

The `toxicity_mitigation.py` module provides functions to rewrite toxic comments into more constructive versions.

#### Running Mitigation Experiments

```python
from toxicity_mitigation import process_dataset, analyze_mitigation_results

# Process toxic comments and mitigate them
process_dataset(
    input_file="mitigate-code-review-dataset-full.csv",
    output_file="mitigate-code-review-dataset-full.csv",
    max_rows=-1  # Process all rows
)

# Analyze mitigation results
analyze_mitigation_results("mitigate-code-review-dataset-full.csv")
```

#### Individual Comment Mitigation

```python
from toxicity_mitigation import (
    mitigate_toxicity_with_gemini,
    mitigate_toxicity_with_deepseek
)

toxic_comment = "This is the worst code I've ever seen. Are you stupid?"

# Mitigate with DeepSeek
mitigated = mitigate_toxicity_with_deepseek(toxic_comment)
print(f"Original: {toxic_comment}")
print(f"Mitigated: {mitigated}")
```

## Dataset

### Input Format

The dataset should be in CSV format with the following columns:

For **detection experiments**:
- `message`: The code review comment text
- `is_toxic`: Ground truth label (1 for toxic, 0 for non-toxic)

For **mitigation experiments**:
- `message`: The code review comment text
- `Toxicity Prediction`: Whether the comment was predicted as toxic ("Yes"/"No")

### Example Data Structure

```csv
message,is_toxic
"Could you please review the error handling logic?",0
"Your code is garbage and makes no sense",1
```

### Dataset Sources

The experiments use code review comments from software engineering contexts. The `train_comments.csv` file contains labeled examples used for few-shot learning.

## Experiments

### Experiment 1: Toxicity Detection

Compares GPT-4.1, Gemini, and DeepSeek on toxicity detection with both few-shot and zero-shot approaches.

**Run the full detection experiment:**
```bash
python toxicity_detection.py
```

**Expected outputs:**
- Processed dataset with predictions and latencies
- Classification reports with precision, recall, and F1-scores
- Confusion matrices

### Experiment 2: Toxicity Mitigation

Evaluates the ability of models to rewrite toxic comments while preserving semantic meaning.

**Run the full mitigation experiment:**
```bash
python toxicity_mitigation.py
```

**Expected outputs:**
- Mitigated versions of toxic comments
- Semantic similarity scores (using Sentence-BERT)
- Mitigation success rates
- Average attempts needed to achieve non-toxic output
- Visualization plots (similarity distributions)

### Progress Tracking

Both experiments support automatic progress saving:
- Progress is saved every 500 rows
- If interrupted, rerun the same command to resume from the last checkpoint
- Progress files: `*.progress.json`

## Results

### Evaluation Metrics

**Detection:**
- Accuracy, Precision, Recall, F1-Score
- Average latency per prediction
- Confusion matrix

**Mitigation:**
- Success Rate: % of toxic comments successfully mitigated
- Average Mitigation Attempts: Number of iterations needed
- Semantic Similarity: Cosine similarity between original and mitigated text
- Failure Rate: % of comments still toxic after 3 attempts

### Example Results

Results are displayed in console output and can include:

```
=== DeepSeek Mitigation Analysis ===
Mitigation Success Rate: 92.50%
Average Mitigation Attempts: 1.23
Average Similarity Score: 0.8456
Failure Cases after 3 attempts: 7.50%

=== Two-Model Comparison (Gemini vs DeepSeek) ===
Success Rate: Gemini 89.20% | DeepSeek 92.50%
Avg Attempts: Gemini 1.45 | DeepSeek 1.23
Avg Similarity: Gemini 0.8234 | DeepSeek 0.8456
```

## Project Structure

```
SEToxicityMitigation/
├── toxicity_detection.py       # Detection experiments
├── toxicity_mitigation.py      # Mitigation experiments
├── requirements.txt            # Python dependencies
├── keys.env                    # API keys (not in repo)
├── .gitignore                  # Git ignore rules
├── train_comments.csv          # Training examples for few-shot
├── train_issues.csv            # Generated few-shot examples
└── README.md                   # This file
```

## Troubleshooting

### API Rate Limits

If you encounter rate limit errors:
- The DeepSeek detection function includes automatic retry logic with exponential backoff
- Consider adding delays between API calls
- Use `max_rows` parameter to process data in batches

### Progress Files

If you want to start fresh:
```bash
rm *.progress.json
rm *-copy-*.csv  # Remove backup files
```



