import pandas as pd
from dotenv import load_dotenv
import os
import openai
from openai import OpenAI
import google.generativeai as genai
from sklearn.metrics import confusion_matrix, classification_report
import time
import random

# Load API keys
load_dotenv('keys.env')
openai.api_key = os.getenv('OPENAI_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
deepseek_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

# Load few-shot examples
def load_or_create_few_shot_examples(original_csv_file, few_shot_csv_file="examples_few_shot.csv"):
    if os.path.exists(few_shot_csv_file):
        df_saved = pd.read_csv(few_shot_csv_file)
        examples = df_saved.to_dict(orient="records")
    else:
        df_original = pd.read_csv(original_csv_file).dropna(subset=['text', 'toxicity'])
        df_toxic = df_original[df_original['toxicity'] == 'y']
        df_non_toxic = df_original[df_original['toxicity'] == 'n']
        sampled_toxic = df_toxic.sample(3, random_state=42)
        sampled_non_toxic = df_non_toxic.sample(3, random_state=42)
        combined = pd.concat([sampled_toxic, sampled_non_toxic]).sample(frac=1, random_state=42)
        data_list = [{"comment": row['text'], "label": "Yes" if row['toxicity'] == 'y' else "No"} for _, row in combined.iterrows()]
        pd.DataFrame(data_list).to_csv(few_shot_csv_file, index=False)
        examples = data_list
    return examples

# Prompts
examples = load_or_create_few_shot_examples("train_comments.csv")
few_shot_explanation = (
    "In this task, you will read a code review comment. "
    "If any part contains rude, offensive, or disrespectful language, respond with 'Yes'. Otherwise, 'No'.\n"
)
zero_shot_prompt = (
    "In this task, you will read a code review comment. "
    "If any part contains rude, offensive, or disrespectful language, respond with 'Yes'. Otherwise, 'No'."
)
few_shot_prompt = few_shot_explanation + "\n".join(
    [f"Comment: {ex['comment']}\nToxic: {ex['label']}\n" for ex in examples]
)

# Detection Functions
def detect_toxicity_with_gpt41(input_text, few_shot=False):
    start = time.time()
    try:
        prompt = f"{few_shot_prompt if few_shot else zero_shot_prompt} Here is the comment you need to determine: {input_text}"
        response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a toxicity detection assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip(), time.time() - start
    except Exception as e:
        print(f"Error in GPT-4.1: {e}")
        return "Error", None

def detect_toxicity_with_gemini(input_text, few_shot=False):
    start = time.time()
    try:
        prompt = f"{few_shot_prompt if few_shot else zero_shot_prompt} Here is the comment you need to determine: {input_text}"
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip(), time.time() - start
    except Exception as e:
        print(f"Error in Gemini: {e}")
        return "Error", None

def detect_toxicity_with_deepseek(input_text, few_shot=False):
    start = time.time()
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            prompt = f"{few_shot_prompt if few_shot else zero_shot_prompt} Here is the comment you need to determine: {input_text}"
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",  # Using DeepSeek's chat model
                messages=[
                    {"role": "system", "content": "You are a toxicity detection assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0,
                top_p=1.0,
                max_tokens=512,
                timeout=30,  # 30 second timeout
            )
            return response.choices[0].message.content.strip(), time.time() - start
        except Exception as e:
            print(f"Error in DeepSeek (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed after {max_retries} attempts")
                return "Error", None

# Process Dataset
def process_dataset(input_file, output_file, save_interval=1, start_row=0, max_rows=-1):
    df = pd.read_csv(input_file)

    # All model columns (GPT-4.1, Gemini, DeepSeek)
    pred_cols = [
        'Toxicity GPT-4.1 Few-Shot', 'Toxicity GPT-4.1 Zero-Shot',
        'Toxicity Gemini Few-Shot', 'Toxicity Gemini Zero-Shot',
        'Toxicity DeepSeek Few-Shot', 'Toxicity DeepSeek Zero-Shot'
    ]
    time_cols = [
        'Latency GPT-4.1 Few-Shot (s)', 'Latency GPT-4.1 Zero-Shot (s)',
        'Latency Gemini Few-Shot (s)', 'Latency Gemini Zero-Shot (s)',
        'Latency DeepSeek Few-Shot (s)', 'Latency DeepSeek Zero-Shot (s)'
    ]

    for col in pred_cols + time_cols:
        if col not in df.columns:
            df[col] = None

    processed_count = 0
    for index, row in df.iterrows():
        # Check if max_rows limit reached
        if max_rows > 0 and processed_count >= max_rows:
            print(f"Reached max_rows limit: {max_rows}")
            break

        # Skip if already processed (check all models)
        if (pd.notna(row.get('Toxicity GPT-4.1 Few-Shot')) and
            pd.notna(row.get('Toxicity Gemini Few-Shot')) and
            pd.notna(row.get('Toxicity DeepSeek Few-Shot'))):
            continue

        text = row.get('message', '')
        if not isinstance(text, str) or pd.isna(text):
            text = ""

        print(f"Processing {index+1}/{len(df)}: {text[:50]}...")

        # GPT-4.1 Few-Shot
        if pd.isna(row.get('Toxicity GPT-4.1 Few-Shot')):
            pred, lat = detect_toxicity_with_gpt41(text, few_shot=True)
            df.at[index, pred_cols[0]] = pred
            df.at[index, time_cols[0]] = lat

        # GPT-4.1 Zero-Shot
        if pd.isna(row.get('Toxicity GPT-4.1 Zero-Shot')):
            pred, lat = detect_toxicity_with_gpt41(text, few_shot=False)
            df.at[index, pred_cols[1]] = pred
            df.at[index, time_cols[1]] = lat

        # Gemini Few-Shot
        if pd.isna(row.get('Toxicity Gemini Few-Shot')):
            pred, lat = detect_toxicity_with_gemini(text, few_shot=True)
            df.at[index, pred_cols[2]] = pred
            df.at[index, time_cols[2]] = lat

        # Gemini Zero-Shot
        if pd.isna(row.get('Toxicity Gemini Zero-Shot')):
            pred, lat = detect_toxicity_with_gemini(text, few_shot=False)
            df.at[index, pred_cols[3]] = pred
            df.at[index, time_cols[3]] = lat

        # DeepSeek Few-Shot
        if pd.isna(row.get('Toxicity DeepSeek Few-Shot')):
            pred, lat = detect_toxicity_with_deepseek(text, few_shot=True)
            df.at[index, pred_cols[4]] = pred
            df.at[index, time_cols[4]] = lat

        # DeepSeek Zero-Shot
        if pd.isna(row.get('Toxicity DeepSeek Zero-Shot')):
            pred, lat = detect_toxicity_with_deepseek(text, few_shot=False)
            df.at[index, pred_cols[5]] = pred
            df.at[index, time_cols[5]] = lat

        processed_count += 1

        if (index + 1) % save_interval == 0:
            df.to_csv(output_file, index=False)
            print("Saved progress...")
        if (index + 1) % 500 == 0:
            df.to_csv(str(index)+'-copy-'+output_file, index=False)
            print("Saved progress to copy...")

    df.to_csv(output_file, index=False)
    print(f"Done. Output saved to {output_file}")

# Metrics
def calculate_metrics(processed_file, ground_truth_column):
    df = pd.read_csv(processed_file)
    # Evaluate all models (GPT-4.1, Gemini, DeepSeek)
    models = [
        'Toxicity GPT-4.1 Few-Shot', 'Toxicity GPT-4.1 Zero-Shot',
        'Toxicity Gemini Few-Shot', 'Toxicity Gemini Zero-Shot',
        'Toxicity DeepSeek Few-Shot', 'Toxicity DeepSeek Zero-Shot'
    ]

    for model in models:
        if model not in df.columns:
            print(f"\n{model} column not found. Skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating {model}")
        print('='*60)

        # Filter out rows where prediction is not available
        df_filtered = df[df[model].notna()].copy()

        y_true = df_filtered[ground_truth_column].map(lambda x: 1 if x == 'y' or x == 1 else 0)
        y_pred = df_filtered[model].map(lambda x: 1 if str(x).strip().lower() in ['yes', 'yes.'] else 0)

        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["Non-Toxic", "Toxic"]))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives (TN): {tn}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")
        print(f"  True Positives (TP): {tp}")

        # Calculate additional metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nSummary Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")

        # Latency statistics if available
        latency_col = f"Latency {model.replace('Toxicity', '').strip()} (s)"
        if latency_col in df_filtered.columns:
            avg_latency = df_filtered[latency_col].mean()
            print(f"  Average Latency: {avg_latency:.3f} seconds")

if __name__ == "__main__":
    input_file = "code-review-dataset-full.csv"
    output_file = "processed-code-review-dataset-full.csv"
    # process_dataset(input_file, output_file,1)
    calculate_metrics(output_file, "is_toxic")
