import os
import time
import logging
import requests
from openai import OpenAI
from typing import List, Dict, Optional, Union
from types import SimpleNamespace
logger = logging.getLogger(__name__)

def boyue_model_requests(model: str="Qwen/Qwen3-8B", messages: List[Dict]=None, available_tools: List[Dict]=None, num_samples: int=1):

    base_url = "http://35.220.164.252:3888/v1/"
    api_key = os.getenv("OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        print(model)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            # tools=available_tools or None,
            # tool_choice="auto" if available_tools else "none",  # 让模型决定是否调用
            max_tokens=500,
            temperature=0.8,
            top_p=0.7,
            # frequency_penalty=0.5,
            n=num_samples,
            timeout=60,
        )
        if available_tools== None:
            if num_samples == 1:
                return response.choices[0].message.content
            else:
                outputs = [choice.message.content for choice in response.choices]
                return outputs
        else:
            return response.choices[0].message
        
    except Exception as e:
        logger.error(f"❌ LLM 调用失败: {e}")
        return None 
    

if __name__ == '__main__':
    messages = [{"role":"system", "content":"你是一个金融数据助手"},
                    {"role":"user", "content": "请给我今天的a股指数"}]
    result = boyue_model_requests(messages=messages)
    print(result)