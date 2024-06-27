from src.utils.api_utils import APIPool

if __name__ == '__main__':
    api_urls = [
        ("https://api.deepseek.com", "your-ak", "deepseek-chat", 1), # deepseek
    ]
    messages = [
        {"role": "system", "content": "You are a kind boy."},
        {"role": "user", "content": "Hello, I'm John."},
    ]
    api_pool = APIPool(api_urls=api_urls)
    result, model = api_pool.get_response(messages)
    print(messages)
    print(result)