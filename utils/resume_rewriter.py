import os
import time
import replicate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

if not REPLICATE_API_TOKEN:
    raise ValueError(
        "❌ Missing REPLICATE_API_TOKEN in environment variables."
    )

# Replicate Client
client = replicate.Client(api_token=REPLICATE_API_TOKEN)


def rewrite_resume(resume_text: str, job_description: str = "") -> str:
    """
    Rewrite and optimize resume for ATS compatibility.
    """

    if job_description.strip():

        user_prompt = f"""
You are a senior resume writer and ATS optimization expert.

TASK:
Rewrite the resume to strongly match the job description.

STRICT RULES:
- Return ONLY the rewritten resume.
- Do NOT write phrases such as:
  "Here is the rewritten resume"
  "Rewritten Resume"
  "Updated Resume"
  "Certainly"
  "Sure"
  or any explanation.
- Do NOT use markdown.
- Preserve the candidate's actual experience.
- Improve wording significantly.
- Convert responsibilities into achievements.
- Add measurable impact wherever possible.
- Use ATS-friendly keywords from the job description.
- Keep professional formatting.

JOB DESCRIPTION:
{job_description.strip()}

ORIGINAL RESUME:
{resume_text.strip()}

OUTPUT:
Only the final rewritten resume.
"""

    else:

        user_prompt = f"""
You are a professional resume writer.

TASK:
Rewrite the resume to make it ATS-friendly and highly professional.

STRICT RULES:
- Return ONLY the rewritten resume.
- Do NOT write:
  "Here is the rewritten resume"
  "Updated Resume"
  explanations
  notes
  introductions
  conclusions
- Do NOT use markdown.
- Improve grammar.
- Improve formatting.
- Rewrite weak bullet points.
- Convert duties into accomplishments.
- Use powerful action verbs.
- Make the resume look professionally written.
- Preserve factual information.

ORIGINAL RESUME:
{resume_text.strip()}

OUTPUT:
Only the rewritten resume.
"""

    max_retries = 3

    for attempt in range(max_retries):
        try:

            output = client.run(
                "meta/meta-llama-3-8b-instruct",
                input={
                    "prompt": user_prompt,
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "max_new_tokens": 1200,
                },
            )

            if isinstance(output, list):
                result = "".join(output)

            elif hasattr(output, "__iter__") and not isinstance(output, str):
                result = "".join(list(output))

            else:
                result = str(output)

            # Remove common unwanted prefixes
            unwanted_prefixes = [
                "Here is the rewritten resume:",
                "Rewritten Resume:",
                "Updated Resume:",
                "Certainly!",
                "Sure!",
            ]

            for prefix in unwanted_prefixes:
                if result.startswith(prefix):
                    result = result.replace(prefix, "", 1).strip()

            return result

        except replicate.exceptions.ReplicateError as e:

            if (
                "insufficient credit" in str(e).lower()
                and attempt < max_retries - 1
            ):
                wait_time = 2 ** (attempt + 1)

                print(
                    f"⚠️ ReplicateError. Retrying in {wait_time}s..."
                )

                time.sleep(wait_time)
                continue

            return (
                "Replicate API error. "
                "Please verify API token and account credits."
            )

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return f"Unexpected error: {str(e)}"

    return "Failed to rewrite resume after multiple attempts."