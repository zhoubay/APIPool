
import openai
from openai import OpenAI
import random
# ak and secret key for doubao
ak_sk_list = [
    ("your-ak-1", "your-secret-key-1"), # for doubao
]

api_urls = [
    ("https://api.deepseek.com", "your-ak", "deepseek-chat", 1), # deepseek
    ("https://api.moonshot.cn/v1", "your-ak", "moonshot-v1-8k", 1), # moonshot
    ("https://dashscope.aliyuncs.com/compatible-mode/v1", "your-ak", "qwen-long", 1), # aliyun
    ("https://dashscope.aliyuncs.com/compatible-mode/v1", "your-ak", "qwen-turbo", 1), # aliyun
    (None, "your-ak", "GLM-4-FLASH", 1), # zhipuai
    (None, "your-ak", "GLM-3-Turbo", 1), # zhipuai
    (None, "your-ak", "GLM-4-FLASH", 1), # zhipuai
    (None, "your-ak", "GLM-3-Turbo", 1), # zhipuai
    ("https://ark.cn-beijing.volces.com/api/v3", "your-ak", "your-endpoint", 1), # doubao
    
]
class APIPool():
    def __init__(self, api_urls, max_probability=1000, min_probability=1):
        self.api_pool = [(url[0], url[1], url[2]) for url in api_urls]
        self.api_probabilities = [url[3] for url in api_urls]
        self.max_probability = max_probability
        self.min_probability = min_probability
    
    def print_probabilities(self):
        for api_p, api_prob in zip(self.api_pool, self.api_probabilities):
            print(f"{api_p[2]}: {api_prob}", flush=True)

    def get_client(self, strategy="random"):
        if strategy == "random":
            pool_index = random.choices(range(len(self.api_pool)), self.api_probabilities, k=1)[0]
            pool = self.api_pool[pool_index]
            base_url, api_key, model = pool
            if "GLM" in model:
                from zhipuai import ZhipuAI
                client = ZhipuAI(api_key=api_key)
            elif "ep-" in model: # Doubao
                from volcenginesdkarkruntime import Ark
                ak, sk = random.choice(ak_sk_list)
                client = Ark(base_url=base_url, api_key=api_key, ak=ak, sk=sk)
            else:
                client = OpenAI(base_url=base_url, api_key=api_key)
            return client, model, pool_index
        else:
            raise NotImplementedError
    
    def set_api_probability(self, pool_index, probability):
        self.api_probabilities[pool_index] = probability
    
    def increate_api_probability(self, pool_index, delta=1):
        self.api_probabilities[pool_index] += delta
        self.api_probabilities[pool_index] = min(self.api_probabilities[pool_index], self.max_probability)
    
    def decrease_api_probability(self, pool_index, delta=1):
        self.api_probabilities[pool_index] -= delta
        self.api_probabilities[pool_index] = max(self.api_probabilities[pool_index], self.min_probability)
    
    def disable_api(self, pool_index):
        self.set_api_probability(pool_index, 0)
    
    def get_response(self, messages, temperature=0.7):
        while True:
            client, model, pool_index = self.get_client(strategy="random")
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                )
                result = response.choices[0].message.content
                self.increate_api_probability(pool_index)
                break
            except openai.PermissionDeniedError as e:
                print(f"{model}: {e}")
                self.decrease_api_probability(pool_index)
                if "insufficient_user_quota" in str(e):
                    self.disable_api(pool_index)
                    print(f"API {model} disabled due to insufficient_user_quota")
                client, model, pool_index = self.get_client(strategy="random")
            except Exception as e:
                print(f"{model}: {e}")
                self.decrease_api_probability(pool_index)
                client, model, pool_index = self.get_client(strategy="random")
        return result, model