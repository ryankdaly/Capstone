from openai import OpenAI

# Configuration
openai_api_key = '[enter your key here]'

# or you can load your key from a file to keep it private
key_file = "my_key.txt"

try:
    with open(key_file, 'r') as f:
        openai_api_key = f.read().strip()
except FileNotFoundError:
    print(f"Error: The file '{key_file}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

openai_api_base = "https://llm-api.arc.vt.edu/api/v1"

# Set up the OpenAI client
client = OpenAI(
    base_url=openai_api_base,
    api_key=openai_api_key,
)

def test_connection():
    """Send a simple test message to confirm the LLM API is connected."""
    try:
        print("Testing connection to ARC LLM API...")
        response = client.chat.completions.create(
            model="gpt-oss-120b",  # ARC's available model
            messages=[
                {"role": "user", "content": "Say 'Connection successful!' in exactly those words."}
            ],
            max_tokens=20,
        )
        reply = response.choices[0].message.content
        print(f"LLM Connected! Response: {reply}")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()