import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print(f"API Key present: {bool(api_key)}")
if api_key:
    print(f"Key starts with: {api_key[:6]}... (Length: {len(api_key)})")

client = Groq(api_key=api_key)

try:
    models = client.models.list()
    print("\n--- Accessible Models for this Key ---")
    for m in models.data:
        print(f"- {m.id}")
except Exception as e:
    print(f"\nAuthentication failed with error:\n{e}")