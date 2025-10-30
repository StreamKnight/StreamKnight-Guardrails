from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

while True:
    response = client.responses.create(
        model="gpt-5",
        input="Write a one-sentence bedtime story about a unicorn."
    )

    print(response.output_text)