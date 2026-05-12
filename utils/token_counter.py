import threading
from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class TokenCounterCallback(BaseCallbackHandler):
    """
    A callback handler to count tokens and calculate costs for LLM calls.
    Thread-safe for concurrent API calls.
    """
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.successful_requests = 0
        self.lock = threading.Lock()
        
        # 常见模型的价格配置 (单位: RMB / 1M tokens)
        # 请根据实际使用的 API 价格进行调整
        self.pricing_rates = {
            "xdf-gp-3.0": {"prompt": 10.0, "completion": 60.0}, # 示例价格
            "deepseek-reasoner": {"prompt": 4.0, "completion": 16.0},
            "gpt-3.5-turbo": {"prompt": 3.5, "completion": 10.5},
            "gpt-4o": {"prompt": 35.0, "completion": 105.0},
            "gpt-4o-mini": {"prompt": 1.05, "completion": 4.2},
            "minimax": {"prompt": 1.0, "completion": 1.0}, # 示例
        }

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage when LLM finishes."""
        if response.llm_output is not None and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]
            with self.lock:
                self.total_prompt_tokens += token_usage.get("prompt_tokens", 0)
                self.total_completion_tokens += token_usage.get("completion_tokens", 0)
                self.total_tokens += token_usage.get("total_tokens", 0)
                self.successful_requests += 1

    def get_stats(self):
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "successful_requests": self.successful_requests
        }
        
    def calculate_cost(self, model_name: str = "deepseek-chat") -> float:
        """Calculate estimated cost based on token usage."""
        # 尝试匹配模型名称
        rate = {"prompt": 0.0, "completion": 0.0}
        for key, val in self.pricing_rates.items():
            if key in model_name.lower():
                rate = val
                break
                
        cost = (self.total_prompt_tokens * rate["prompt"] / 1_000_000) + \
               (self.total_completion_tokens * rate["completion"] / 1_000_000)
        return cost

    def print_stats(self, model_name: str = "deepseek-chat"):
        """Print formatted statistics."""
        cost = self.calculate_cost(model_name)
        print("\n" + "="*40)
        print("📊 API Token Usage & Cost Report")
        print("="*40)
        print(f"Model:               {model_name}")
        print(f"Successful Requests: {self.successful_requests}")
        print(f"Prompt Tokens:       {self.total_prompt_tokens:,}")
        print(f"Completion Tokens:   {self.total_completion_tokens:,}")
        print(f"Total Tokens:        {self.total_tokens:,}")
        print("-" * 40)
        print(f"Estimated Cost:      ¥ {cost:.4f} RMB")
        print("="*40 + "\n")

    def reset(self):
        """Reset all counters."""
        with self.lock:
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_tokens = 0
            self.successful_requests = 0

# 全局单例，方便在整个项目中共享统计
global_token_counter = TokenCounterCallback()
