import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise EnvironmentError("GROQ_API_KEY not found in .env. Please set it before running.")

# Initialize Groq client with your API key
client = Groq(api_key=groq_api_key)

def generate_code(prompt, max_tokens=4096, temperature=0.2, model="llama-3.3-70b-versatile"):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=True,
        )
        generated_code = ""
        for chunk in completion:
            content_piece = chunk.choices[0].delta.content or ""
            print(content_piece, end="", flush=True)
            generated_code += content_piece
        return generated_code

    except Exception as e:
        print(f"Error during code generation: {e}")
        return ""
