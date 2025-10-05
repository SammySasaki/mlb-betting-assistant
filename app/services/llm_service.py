import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(prompt: str, model="gpt-3.5-turbo", temp=0.3):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a sports betting analyst assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    models = client.models.list()

    # Print model IDs
    for m in models.data:
        print(m.id)