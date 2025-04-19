import re
from profanity_check import predict, predict_prob
from better_profanity import profanity
import os
import json
import pandas as pd
from groq import Groq
from prompts import create_profanity_prompt
from dotenv import load_dotenv

load_dotenv()

# Configure better_profanity with English profanity list
profanity.load_censor_words()


def process_file(filepath, use_llm=False, file_upload=[{}], upload=False):
    """Process a single JSON file and extract profanity information"""
    if upload:
        data = file_upload
        file_id = "temp"
    else:
        file_id = os.path.basename(filepath)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return []

    results = []
    for entry in data:
        text = entry.get("text", "")
        is_profane, method, profane_terms = english_profanity_checker(
            text, api_key=os.environ.get("GROQ_API_KEY"), use_llm=use_llm
        )

        if is_profane:
            # Ensure profane_terms is a list
            if not isinstance(profane_terms, list):
                profane_terms = [profane_terms]

            # For multiple profane terms, create separate entries
            for term in profane_terms:
                results.append(
                    {
                        "file_id": file_id,
                        "timestamp_start": entry.get("stime"),
                        "timestamp_end": entry.get("etime"),
                        "speaker": entry.get("speaker"),
                        "profane_term": term,
                        "sentence": text,
                        "detection_method": method,
                    }
                )

    return results


def process_directory(directory_path):
    """Process all JSON files in a directory and compile profanity results"""
    all_profanity_data = []

    # List all JSON files in the directory
    json_files = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.endswith(".json")
    ]
    print(f"Found {len(json_files)} JSON files to process")

    # Process each file
    for i, file_path in enumerate(json_files):
        file_results = process_file(file_path)
        all_profanity_data.extend(file_results)

        # Print progress every 25 files
        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(json_files)} files")

    # Convert to pandas DataFrame
    df_profanity = pd.DataFrame(all_profanity_data)

    # Sort by file_id and timestamp_start for better readability
    if not df_profanity.empty:
        df_profanity = df_profanity.sort_values(["file_id", "timestamp_start"])

    return df_profanity


def check_profanity_with_llm(text, api_key=None):
    """Use Groq's LLM to check for subtle or contextual profanity"""
    # Initialize Groq client
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Warning: No Groq API key provided. Skipping LLM check.")
        return False, None

    client = Groq(api_key=api_key)

    # Create system prompt for effective profanity detection
    system_prompt = create_profanity_prompt()

    user_prompt = f"""Analyze this text for any profanity or offensive language: "{text}"

If profanity or offensive language is found, return a JSON object with this format:
{{"detected": true, "terms": ["term1", "term2", ...]}}

If no profanity or offensive language is found:
{{"detected": false, "terms": []}}

Only respond with the JSON object, nothing else."""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # Using Llama 3.1 for best performance
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=200,
            response_format={"type": "json_object"},
        )

        result_text = response.choices[0].message.content
        try:
            result = json.loads(result_text)

            # If profanity was found
            if result.get("detected", False) and result.get("terms", []):
                return True, result.get("terms", [])

            return False, None
        except json.JSONDecodeError:
            print(f"Warning: LLM returned non-JSON response: {result_text}")
            return False, None

    except Exception as e:
        print(f"Warning: LLM profanity check failed with error: {e}")
        return False, None


def english_profanity_checker(text, use_llm=False, api_key=None):
    """
    Hybrid approach for detecting profanity in English text.
    Returns (is_profane, method, detected_terms)

    Parameters:
    - text: The text to check for profanity
    - use_llm: Whether to use LLM as a final check (default: True)
    - api_key: Groq API key (if None, will look for GROQ_API_KEY environment variable)
    """
    # Stage 1: Regex for common English profanity patterns
    english_regex_patterns = [
        r"\bf+[^\w]*u+[^\w]*c+[^\w]*k+\w*",
        r"\bf+[^\w]*\*+[^\w]*c+[^\w]*k+\w*",
        r"\bs+[^\w]*h+[^\w]*i+[^\w]*t+\w*",
        r"\bs+[^\w]*\*+[^\w]*\*+[^\w]*t+\w*",
        r"\bn+[^\w]*i+[^\w]*g+[^\w]*g+[^\w]*[^\w]*r+\w*",
        r"\bb+[^\w]*i+[^\w]*t+[^\w]*c+[^\w]*h+\w*",
        r"\ba+[^\w]*s+[^\w]*s+[^\w]*h+[^\w]*o+[^\w]*l+[^\w]*e+\w*",
        r"\bp+[^\w]*[uo]+[^\w]*r+[^\w]*n+\w*",
        r"\bp+[^\w]*e+[^\w]*n+[^\w]*i+[^\w]*s+\w*",
    ]

    # Check against regex patterns - find ALL matches
    regex_matches = []
    for pattern in english_regex_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            regex_matches.append(match.group(0))

    if regex_matches:
        return True, "regex", regex_matches

    # Stage 2: Dictionary-based check with better_profanity
    if profanity.contains_profanity(text):
        censored = profanity.censor(text, "*")
        words = text.split()
        censored_words = censored.split()
        profane_terms = []

        for original, censored in zip(words, censored_words):
            if "*" in censored:
                profane_terms.append(original)

        return True, "dictionary", profane_terms

    # Stage 3: ML-based detection for subtle cases
    try:
        probability = predict_prob([text])[0]
        if probability > 0.7:  # Adjustable threshold
            return (
                True,
                "machine_learning",
                [f"ML detection (probability: {probability})"],
            )
    except Exception as e:
        print(f"Warning: ML detection failed with error: {e}")

    # Stage 4: LLM-based detection as a last resort
    if use_llm:
        is_profane, profane_terms = check_profanity_with_llm(text, api_key)
        if is_profane:
            print("llm", profane_terms)
            return True, "llm", profane_terms

    return False, None, None


# if __name__ == "__main__":
#     """Main function to process files and generate CSV output"""
#     # Replace with your actual directory path
#     directory_path = "./All_Conversations/"
#     output_csv_path = "profanity_report_llm.csv"

#     print(f"Starting profanity detection on files in {directory_path}")

#     # Process all files and get DataFrame
#     df_results = process_directory(directory_path)

#     if df_results.empty:
#         print("No profanity found or no valid files processed")

#     # Save to CSV
#     df_results.to_csv(output_csv_path, index=False)
#     print(f"Results saved to {output_csv_path}")
#     print(
#         f"Found {len(df_results)} instances of profanity across {df_results['file_id'].nunique()} files"
#     )

if __name__ == "__main__":
    # Example usage
    print(process_file("./All_Conversations/04bec80f-8614-484b-8ba2-831ff9dd03ef.json"))
