import pandas as pd
from toxicity_detection import detect_toxicity_with_gpt41, detect_toxicity_with_gemini

from dotenv import load_dotenv
import os
import openai
from openai import OpenAI
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np
import json

# Load API keys from environment file
load_dotenv('keys.env')

# Set API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
deepseek_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

# Initialize Sentence-BERT for semantic similarity
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Few-shot examples for mitigation
mitigation_examples = [
    {
        "original": "You are the worst developer ever. This code is garbage!",
        "rewritten": "I'm concerned about the current approach. Could we find a clearer way to write this code?"
    },
    {
        "original": "This fix is so stupid, why are you even trying?",
        "rewritten": "I see potential issues with this fix. Let's discuss possible improvements."
    }
]

# Explanation of the task for rewriting
mitigation_few_shot_explanation = (
    "Your task is to rewrite a given comment to remove rude or offensive language, while preserving its core meaning. "
    "The rewritten comment should be polite, respectful, and constructive. Below are some examples:\n\n"
)

# Build the prompt string for few-shot mitigation
mitigation_few_shot_prompt = mitigation_few_shot_explanation + "\n".join([
    f"Original: {ex['original']}\nRewritten: {ex['rewritten']}\n" for ex in mitigation_examples
])


def mitigate_toxicity_with_gemini(input_text):
    """
    Use Gemini in few-shot mode to rewrite the input text, removing toxic or offensive language.
    """
    try:
        prompt = (
            f"{mitigation_few_shot_prompt}"
            f"Original: {input_text}\n"
            "Rewritten:"
        )
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error mitigating with Gemini: {e}")
        return input_text


def mitigate_toxicity_with_deepseek(input_text):
    """
    Use DeepSeek in few-shot mode to rewrite the input text, removing toxic or offensive language.
    """
    try:
        prompt = (
            f"{mitigation_few_shot_prompt}"
            f"Original: {input_text}\n"
            "Rewritten:"
        )
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",  # Using DeepSeek's chat model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that rewrites toxic comments to be polite and respectful."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            top_p=1.0,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error mitigating with DeepSeek: {e}")
        return input_text


def determine_similarity_with_bert(original_text, mitigated_text):
    """
    Use Sentence-BERT (LLM C) to determine similarity between two sentences.
    """
    try:
        embeddings1 = similarity_model.encode(original_text, convert_to_tensor=True)
        embeddings2 = similarity_model.encode(mitigated_text, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2).item()
        return similarity_score
    except Exception as e:
        print(f"Error determining similarity: {e}")
        return None


def process_dataset(input_file, output_file, max_rows=-1):
    """
    1. Read input CSV
    2. Predict toxicity and mitigate
    3. Save mitigated text, predictions, similarity
    4. Write to output CSV
    """
    progress_file = output_file + ".progress.json"
    last_index = -1
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as pf:
                data = json.load(pf)
                last_index = data.get('last_index', -1)
            print(f"Resuming from last_index={last_index}")
        except Exception as e:
            print(f"无法加载进度文件，重新开始: {e}")

    df = pd.read_csv(input_file)

    # Ensure necessary columns exist for Gemini and DeepSeek
    gemini_cols = ['Mitigated Text', 'Toxicity Prediction', 'Toxicity After Mitigation', 'Similarity Score', 'Mitigation Attempts']
    deepseek_cols = ['Mitigated Text (DeepSeek)', 'Toxicity After Mitigation (DeepSeek)', 'Similarity Score (DeepSeek)', 'Mitigation Attempts (DeepSeek)']

    for col in gemini_cols + deepseek_cols:
        if col not in df.columns:
            df[col] = None if 'Text' in col or 'Prediction' in col or 'Mitigation' in col else 0

    row_count = 0
    for index, row in df.iterrows():
        if max_rows > 0 and row_count >= max_rows:
            break

        # Check if DeepSeek is already done
        deepseek_done = pd.notna(row.get('Mitigated Text (DeepSeek)')) and row.get('Mitigated Text (DeepSeek)') != ""

        # Skip if DeepSeek is already done
        if deepseek_done:
            continue

        input_text = row['message']

        # Step 1: Predict toxicity (reuse if already exists)
        if pd.notna(row['Toxicity Prediction']) and row['Toxicity Prediction'] != "":
            toxicity_prediction = row['Toxicity Prediction']
            print(f"Row {index}: Reusing existing toxicity prediction: {toxicity_prediction}")
        else:
            toxicity_prediction, _ = detect_toxicity_with_gpt41(input_text, few_shot=True)
            df.at[index, 'Toxicity Prediction'] = toxicity_prediction
            print(f"Row {index}: New toxicity prediction: {toxicity_prediction}")

        # Step 2: Mitigate toxicity if toxic (only DeepSeek)
        if toxicity_prediction == "Yes":
            print(f"  Running DeepSeek mitigation...")
            mitigated_text_deepseek = mitigate_toxicity_with_deepseek(input_text)
            count_deepseek = 0
            new_toxicity_prediction_deepseek = toxicity_prediction

            while count_deepseek < 3:
                new_toxicity_prediction_deepseek, _ = detect_toxicity_with_gpt41(mitigated_text_deepseek, few_shot=True)
                if new_toxicity_prediction_deepseek == "No":
                    break
                mitigated_text_deepseek = mitigate_toxicity_with_deepseek(mitigated_text_deepseek)
                count_deepseek += 1

            df.at[index, 'Mitigated Text (DeepSeek)'] = mitigated_text_deepseek
            df.at[index, 'Toxicity After Mitigation (DeepSeek)'] = new_toxicity_prediction_deepseek
            df.at[index, 'Mitigation Attempts (DeepSeek)'] = count_deepseek + 1

            similarity_score_deepseek = determine_similarity_with_bert(input_text, mitigated_text_deepseek)
            df.at[index, 'Similarity Score (DeepSeek)'] = similarity_score_deepseek

        row_count += 1
        if row_count % 10 == 0:
            print(f"Flushing to CSV at row {index}")
            df.to_csv(output_file, index=False)
            try:
                with open(progress_file, 'w') as pf:
                    json.dump({'last_index': index}, pf)
            except Exception as e:
                print(f"无法写入进度文件: {e}")

    df.to_csv(output_file, index=False)
    print(f"Processed dataset saved to {output_file}")


def analyze_mitigation_results(csv_file):
    df = pd.read_csv(csv_file)

    # Filter rows that were predicted toxic initially
    originally_toxic = df[df['Toxicity Prediction'] == "Yes"].copy()
    if originally_toxic.empty:
        print("No originally toxic comments found in the dataset.")
        return

    # ===== Gemini Results =====
    print("\n=== Gemini Mitigation Analysis ===")

    # 1) Mitigation Success Rate
    mitigated_success = originally_toxic[originally_toxic['Toxicity After Mitigation'] == "No"]
    success_rate = len(mitigated_success) / len(originally_toxic) * 100

    # 2) Average Mitigation Attempts
    avg_attempts = originally_toxic['Mitigation Attempts'].mean()

    # 3) Average Similarity Score
    mitigated_nonempty = originally_toxic.dropna(subset=['Mitigated Text'])
    if not mitigated_nonempty.empty:
        avg_similarity = mitigated_nonempty['Similarity Score'].mean()
    else:
        avg_similarity = 0.0

    # 4) Failure Cases
    failed_after_3 = originally_toxic[
        (originally_toxic['Mitigation Attempts'] == 3) &
        (originally_toxic['Toxicity After Mitigation'] == "Yes")
    ]
    failure_rate = len(failed_after_3) / len(originally_toxic) * 100

    # Print Gemini results
    print(f"Mitigation Success Rate: {success_rate:.2f}%")
    print(f"Average Mitigation Attempts: {avg_attempts:.2f}")
    print(f"Average Similarity Score: {avg_similarity:.4f}")
    print(f"Failure Cases after 3 attempts: {failure_rate:.2f}%")

    # ===== DeepSeek Results =====
    print("\n=== DeepSeek Mitigation Analysis ===")

    # Check if DeepSeek columns exist
    if 'Toxicity After Mitigation (DeepSeek)' in originally_toxic.columns:
        # 1) Mitigation Success Rate
        mitigated_success_deepseek = originally_toxic[originally_toxic['Toxicity After Mitigation (DeepSeek)'] == "No"]
        success_rate_deepseek = len(mitigated_success_deepseek) / len(originally_toxic) * 100

        # 2) Average Mitigation Attempts
        avg_attempts_deepseek = originally_toxic['Mitigation Attempts (DeepSeek)'].mean()

        # 3) Average Similarity Score
        mitigated_nonempty_deepseek = originally_toxic.dropna(subset=['Mitigated Text (DeepSeek)'])
        if not mitigated_nonempty_deepseek.empty:
            avg_similarity_deepseek = mitigated_nonempty_deepseek['Similarity Score (DeepSeek)'].mean()
        else:
            avg_similarity_deepseek = 0.0

        # 4) Failure Cases
        failed_after_3_deepseek = originally_toxic[
            (originally_toxic['Mitigation Attempts (DeepSeek)'] == 3) &
            (originally_toxic['Toxicity After Mitigation (DeepSeek)'] == "Yes")
        ]
        failure_rate_deepseek = len(failed_after_3_deepseek) / len(originally_toxic) * 100

        # Print DeepSeek results
        print(f"Mitigation Success Rate: {success_rate_deepseek:.2f}%")
        print(f"Average Mitigation Attempts: {avg_attempts_deepseek:.2f}")
        print(f"Average Similarity Score: {avg_similarity_deepseek:.4f}")
        print(f"Failure Cases after 3 attempts: {failure_rate_deepseek:.2f}%")

        # ===== Comparison Summary =====
        print("\n=== Two-Model Comparison (Gemini vs DeepSeek) ===")
        print(f"Success Rate: Gemini {success_rate:.2f}% | DeepSeek {success_rate_deepseek:.2f}%")
        print(f"Avg Attempts: Gemini {avg_attempts:.2f} | DeepSeek {avg_attempts_deepseek:.2f}")
        print(f"Avg Similarity: Gemini {avg_similarity:.4f} | DeepSeek {avg_similarity_deepseek:.4f}")
        print(f"Failure Rate: Gemini {failure_rate:.2f}% | DeepSeek {failure_rate_deepseek:.2f}%")
    else:
        print("DeepSeek mitigation results not found in dataset. Run process_dataset first with DeepSeek mitigation enabled.")

    similarity_scores = mitigated_nonempty['Similarity Score'].dropna()
    if not similarity_scores.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(similarity_scores, bins=20, edgecolor='black', alpha=0.7)
        plt.title("Distribution of Similarity Scores")
        plt.xlabel("Similarity Score")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("similarity_distribution.png", dpi=300)

    else:
        print("No valid similarity scores to plot.")

    # For cumulative histogram
    similarity_scores = mitigated_nonempty['Similarity Score'].dropna()
    if not similarity_scores.empty:
        # Calculate cumulative distribution
        weights = np.ones_like(similarity_scores) * 100.0 / len(similarity_scores)
        
        plt.figure(figsize=(8, 5))
        plt.hist(similarity_scores, bins=20, edgecolor='black', alpha=0.7, weights=weights, cumulative=True)
        plt.title("Cumulative Distribution of Similarity Scores")
        plt.xlabel("Similarity Score")
        plt.ylabel("Cumulative Percentage (%)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("cumulative_similarity_distribution.png", dpi=300)
    else:
        print("No valid similarity scores to plot.")



if __name__ == "__main__":
    input_file = "mitigate-code-review-dataset-full.csv"
    output_file = "mitigate-code-review-dataset-full.csv"
    # process_dataset(input_file, output_file)
    analyze_mitigation_results(output_file)
