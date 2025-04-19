import os
import json
import pandas as pd
from groq import Groq
from prompts import create_llama_3_3_system_prompt
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()


class PrivacyComplianceDetector:
    def __init__(self, api_key=None):
        """Initialize the compliance detector with Groq API"""
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key is required. Set GROQ_API_KEY environment variable or pass it directly."
            )

        self.client = Groq(api_key=self.api_key)
        self.system_prompt = create_llama_3_3_system_prompt()

    def format_transcript(self, transcript_data):
        """Format the transcript data into a readable format for the model"""
        formatted_text = []
        for entry in transcript_data:
            speaker = entry.get("speaker", "Unknown")
            text = entry.get("text", "")
            time_start = entry.get("stime", "")
            time_end = entry.get("etime", "")
            formatted_text.append(f"{speaker} [{time_start}-{time_end}]: {text}")

        return "\n".join(formatted_text)

    def analyze_call_transcript(self, call_id, transcript_data):
        """Analyze a call transcript for privacy compliance violations"""
        formatted_transcript = self.format_transcript(transcript_data)

        # Prepare the user prompt with the transcript to analyze
        user_prompt = f"""Please analyze this call center transcript for privacy compliance violations:

CALL TRANSCRIPT (ID: {call_id}):
{formatted_transcript}

Provide your analysis in the following JSON format:
{{
    "verification_performed": true/false,
    "verification_method": "DOB/Address/SSN/Multiple/None",
    "sensitive_info_shared": true/false,
    "sensitive_info_type": "Description of information shared",
    "is_violation": true/false,
    "explanation": "Detailed explanation of your findings"
}}"""

        # Generate the full prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Call the Groq API with Llama 3.3
        try:
            response = self.client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=messages,
                temperature=0.2,  # Lower temperature for more deterministic responses
                max_tokens=1024,
                response_format={"type": "json_object"},
            )

            # Parse the model's response
            response_content = response.choices[0].message.content
            result = json.loads(response_content)

            # Add the call_id to the result
            result["call_id"] = call_id
            return result

        except Exception as e:
            print(f"Error analyzing call {call_id}: {e}")
            return {
                "call_id": call_id,
                "error": str(e),
                "is_violation": False,
                "verification_performed": False,
                "sensitive_info_shared": False,
            }

    def batch_process_calls(self, call_data_list):
        """Process multiple call transcripts and identify violations"""
        results = []

        for call_data in call_data_list:
            call_id = call_data.get("call_id", "unknown")
            transcript = call_data.get("transcript", [])

            analysis = self.analyze_call_transcript(call_id, transcript)
            results.append(analysis)

            # If there's a violation, print it out
            if analysis.get("is_violation", False):
                print(f"⚠️ Violation detected in call {call_id}")

        # Convert results to DataFrame for easier analysis
        df_results = pd.DataFrame(results)

        # Filter for violations only
        # violations_df = df_results[df_results["is_violation"] == True]
        # violations_df.to_csv("violations.csv", index=False)

        return df_results

    def process_directory(self, directory_path, limit=-1, save_to_csv=True):
        """Process all JSON files in a directory"""
        call_data_list = []

        # List all JSON files in the directory
        json_files = [f for f in os.listdir(directory_path) if f.endswith(".json")]
        print(f"Found {len(json_files)} JSON files to process")

        for idx, filename in enumerate(tqdm(json_files)):
            if limit > 0 and idx >= limit:
                break
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                call_data_list.append({"call_id": filename, "transcript": data})

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        # Process all calls and get violations
        results_df = self.batch_process_calls(call_data_list)
        if save_to_csv:
            results_df.to_csv("compliance_violations.csv", index=False)
            print("Results saved to compliance_violations.csv")
        return self.batch_process_calls(call_data_list)

    def process_single_file(
        self, file_path, upload=False, save_to_csv=False, csv_path=None
    ):
        """
        Process a single JSON file and identify privacy compliance violations.

        Args:
            file_path (str): Path to the JSON file to analyze
            save_to_csv (bool): Whether to save results to CSV
            csv_path (str, optional): Custom path for saving CSV. If None, uses filename-based default

        Returns:
            dict: Analysis results for the file
        """
        try:
            if upload:
                filename = "Temp"
                data = file_path

            else:
                # Extract filename for call_id
                filename = os.path.basename(file_path)

                # Load the JSON file
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            # Analyze the transcript
            print(f"Analyzing file: {filename}")
            analysis_result = self.analyze_call_transcript(filename, data)

            # Create a single-row DataFrame for consistent output format
            result_df = pd.DataFrame([analysis_result])

            # Save to CSV if requested
            if save_to_csv:
                # If no csv_path provided, create one based on the input filename
                if csv_path is None:
                    base_name = os.path.splitext(filename)[0]
                    csv_path = f"{base_name}_compliance_analysis.csv"

                result_df.to_csv(csv_path, index=False)
                print(f"Results saved to {csv_path}")

            # Print alert if violation detected
            if analysis_result.get("is_violation", False):
                print(f"⚠️ Violation detected in {filename}")

            return analysis_result

        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return {
                "call_id": os.path.basename(file_path),
                "error": "File not found",
                "is_violation": False,
                "verification_performed": False,
                "sensitive_info_shared": False,
            }
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in file: {file_path}")
            return {
                "call_id": os.path.basename(file_path),
                "error": "Invalid JSON format",
                "is_violation": False,
                "verification_performed": False,
                "sensitive_info_shared": False,
            }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return {
                "call_id": os.path.basename(file_path),
                "error": str(e),
                "is_violation": False,
                "verification_performed": False,
                "sensitive_info_shared": False,
            }


if __name__ == "__main__":
    # detector = PrivacyComplianceDetector(api_key=os.environ.get("GROQ_API_KEY"))
    # directory_path = "./All_Conversations"
    # results = detector.process_directory(directory_path, limit=10)
    # print(results)
    detector = PrivacyComplianceDetector(api_key=os.environ.get("GROQ_API_KEY"))
    x = detector.process_single_file(
        "./All_Conversations/00be25b0-458f-4cbf-ae86-ae2ec1f7fba4.json"
    )
    print(x)
