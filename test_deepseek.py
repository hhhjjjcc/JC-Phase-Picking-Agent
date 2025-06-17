import os
from dotenv import load_dotenv
from openai import OpenAI
import json

def test_deepseek_api():
    load_dotenv()
    api_key = os.getenv('DEEPSEEK_API_KEY')
    print(f'API Key loaded: {api_key is not None}')

    if not api_key:
        print("Error: DEEPSEEK_API_KEY not found in .env file. Please make sure it's set.")
        return

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com" # DeepSeek API 的基准 URL
        )
        model_name = "deepseek-chat"
        
        print(f"Attempting to generate content with DeepSeek model: {model_name}...")
        
        messages = [
            {"role": "user", "content": "你好，DeepSeek！你是谁？请用中文简短回答。"}
        ]
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        
        print(f'DeepSeek 响应: {response.choices[0].message.content}')

    except Exception as e:
        print(f'DeepSeek API 调用失败: {e}')

if __name__ == "__main__":
    test_deepseek_api() 