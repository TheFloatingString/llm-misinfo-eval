import together
from dotenv import load_dotenv
import os
import openai
import cohere
from google import genai


load_dotenv()


def call_openai(model: str, prompt: str) -> str:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(model=model, input=prompt)
    return response.output_text


def call_together(model: str, prompt: str) -> str:
    client = together.Together(api_key=os.getenv("TOGETHER_API_KEY"))
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    pred = completion.choices[0].message.content
    return pred


def call_cohere(model: str, prompt: str) -> str:
    co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))
    response = co.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response


def call_gemini(model: str, prompt: str) -> str:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text
