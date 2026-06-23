import os
import time
import replicate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

if not REPLICATE_API_TOKEN:
    raise ValueError("❌ Missing REPLICATE_API_TOKEN in .env file. Please add it before running.")

# Initialise Replicate client
client = replicate.Client(api_token=REPLICATE_API_TOKEN)


def rewrite_resume(resume_text: str, job_description: str = "") -> str:
    """
    Rewrite a resume using the LLaMA-2 13B Chat model from Replicate.

    Args:
        resume_text (str): The raw resume text.
        job_description (str): Optional job description text for tailoring.

    Returns:
        str: The rewritten resume or an error message.
    """
    system_prompt = """
        You are an expert resume writer.

        Return ONLY the rewritten resume.

        Do not include:
        - Here is the rewritten resume
        - Explanations
        - Notes
        - Introductions
        - Markdown formatting

        Output only the final resume text."""

    if job_description.strip():
        user_prompt = f"""
Here is the original resume and a job description.
Please rewrite the resume to be highly relevant to this specific job.

--- Job Description ---
{job_description.strip()}

--- Original Resume ---
{resume_text.strip()}

--- Instructions for Rewrite ---
1. Extract skills, responsibilities, and keywords from the job description.
2. Align the resume with these requirements.
3. Transform responsibilities into measurable accomplishments.
4. Correct grammar, spelling, and punctuation.
5. Keep the tone professional and achievement-oriented.
6. Use bullet points where appropriate.
"""
    else:
        user_prompt = f"""
Here is an original resume. Please rewrite it to be more professional and job-ready.

--- Original Resume ---
{resume_text.strip()}

--- Instructions for Rewrite ---
1. Correct grammar, spelling, and punctuation.
2. Start bullet points with strong action verbs.
3. Transform responsibilities into measurable accomplishments.
4. Ensure a confident, professional tone.
5. Maintain a clear and simple formatting style.
"""

    # Retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            output = client.run(
                "meta/meta-llama-3-8b-instruct",
                input={
                    "system_prompt": system_prompt,
                    "prompt": user_prompt,
                    "max_new_tokens": 256,
                    "temperature": 0.6,
                    "top_p": 0.9,
                },
            )

            # Replicate may return a generator, list, or string
            if isinstance(output, list):
                return "".join(output)
            elif hasattr(output, "__iter__") and not isinstance(output, str):
                return "".join(list(output))
            return str(output)

        except replicate.exceptions.ReplicateError as e:
            if "insufficient credit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"⚠️ ReplicateError: Insufficient credit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            return (
                "An error occurred with Replicate API. "
                "Please check your account balance and API token."
            )
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

    return "Failed to rewrite resume after multiple attempts."
