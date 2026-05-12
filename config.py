"""
Configuration module for LLM and Environment setup.
Handles loading variables from .env and creating LangChain ChatOpenAI instances.
"""
import os
import sys
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utils.token_counter import global_token_counter

# Load environment variables from .env file
load_dotenv()

def get_llm(
    temperature: float = 0.0,
    model_name: Optional[str] = None,
    max_completion_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
) -> ChatOpenAI:
    """
    Factory function to create a configured ChatOpenAI instance.
    Compatible with DeepSeek, OpenAI, and other OpenAI-compatible APIs.

    Args:
        temperature (float): Sampling temperature (0.0 to 2.0).
        model_name (str, optional): Model identifier. Defaults to MODEL_NAME env var or 'deepseek-chat'.

    Returns:
        ChatOpenAI: A configured LangChain chat model instance.
    
    Raises:
        ValueError: If API_KEY is missing.
    """
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    
    # Default to 'deepseek-chat' if not specified in args or env
    if not model_name:
        model_name = os.getenv("MODEL_NAME", "deepseek-chat")
    timeout = float(os.getenv("LLM_TIMEOUT", "180"))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "0"))
    if max_completion_tokens is None:
        max_completion_tokens_env = os.getenv("LLM_MAX_COMPLETION_TOKENS", "512").strip()
        max_completion_tokens = int(max_completion_tokens_env) if max_completion_tokens_env else None
    if reasoning_effort is None:
        reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "minimal").strip() or None
    if verbosity is None:
        verbosity = os.getenv("LLM_VERBOSITY", "").strip() or None

    if not api_key:
        # Fallback for some setups or specific error message
        raise ValueError("API_KEY not found. Please set it in your .env file.")

    # Configure ChatOpenAI
    # Note: explicit openai_api_base is required for non-OpenAI endpoints (like DeepSeek)
    
    # Strip unnecessary paths from base_url if present
    if base_url:
        base_url = base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            base_url = base_url.replace("/chat/completions", "")
        if base_url.endswith("/responses"):
            base_url = base_url.replace("/responses", "")
            
    llm_kwargs = {
        "model": model_name,
        "temperature": temperature,
        "openai_api_key": api_key,
        "openai_api_base": base_url,
        # Default parameters to ensure stable output
        "timeout": timeout,
        "max_retries": max_retries,
        "callbacks": [global_token_counter],
    }
    if max_completion_tokens is not None:
        llm_kwargs["max_completion_tokens"] = max_completion_tokens
    if reasoning_effort is not None:
        llm_kwargs["reasoning_effort"] = reasoning_effort
    if verbosity is not None:
        llm_kwargs["verbosity"] = verbosity

    llm = ChatOpenAI(**llm_kwargs)

    return llm
