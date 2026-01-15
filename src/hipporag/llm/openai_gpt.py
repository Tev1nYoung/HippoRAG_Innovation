import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple

import httpx
import openai
from filelock import FileLock
from openai import OpenAI
from openai import AzureOpenAI
from packaging import version
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import (
    TextChatMessage
)
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)

def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # get seed and temperature from kwargs or self.llm_config.generate_params
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))

        # 缓存键只基于messages内容，不包含model字段
        # 这样可以跨API服务商和模型名称复用缓存
        key_data = {
            "messages": messages,  # messages requires JSON serializable
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache
        lock = FileLock(lock_file, timeout=10)
        try:
            with lock:
                conn = sqlite3.connect(self.cache_file_name)
                c = conn.cursor()
                # if the table does not exist, create it
                c.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        message TEXT,
                        metadata TEXT
                    )
                """)
                conn.commit()  # commit to save the table creation
                c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
                row = c.fetchone()
                conn.close()
                if row is not None:
                    message, metadata_str = row
                    metadata = json.loads(metadata_str)
                    # return cached result and mark as hit
                    return message, metadata, True
        finally:
            # 清理锁文件（如果没有其他进程在使用）
            try:
                if os.path.exists(lock_file) and not lock.is_locked:
                    os.remove(lock_file)
            except:
                pass  # 忽略删除失败

        # if cache miss, call the original function to get the result
        result = func(self, *args, **kwargs)
        message, metadata = result

        # insert new result into cache
        lock = FileLock(lock_file, timeout=10)
        try:
            with lock:
                conn = sqlite3.connect(self.cache_file_name)
                c = conn.cursor()
                # make sure the table exists again (if it doesn't exist, it would be created)
                c.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        message TEXT,
                        metadata TEXT
                    )
                """)
                metadata_str = json.dumps(metadata)
                c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                          (key_hash, message, metadata_str))
                conn.commit()
                conn.close()
        finally:
            # 清理锁文件（如果没有其他进程在使用）
            try:
                if os.path.exists(lock_file) and not lock.is_locked:
                    os.remove(lock_file)
            except:
                pass  # 忽略删除失败

        return message, metadata, False

    return wrapper

def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)
        cooldown_min_s = getattr(self, "rate_limit_cooldown_min_s", 5)
        cooldown_max_s = getattr(self, "rate_limit_cooldown_max_s", 8)

        def _is_rate_limit_error(exc: BaseException) -> bool:
            # 429速率限制错误
            if isinstance(exc, openai.RateLimitError):
                return True
            if isinstance(exc, openai.APIStatusError) and getattr(exc, "status_code", None) == 429:
                return True
            if getattr(exc, "status_code", None) == 429:
                return True
            response = getattr(exc, "response", None)
            if getattr(response, "status_code", None) == 429:
                return True

            # 常见的API错误码需要重试：
            # 400: Bad Request (可能是临时格式问题)
            # 401: Unauthorized (认证失败)
            # 403: Forbidden (权限禁止)
            # 408: Request Timeout (请求超时)
            # 500: Internal Server Error (服务器内部错误)
            # 502: Bad Gateway (网关错误)
            # 503: Service Unavailable (服务不可用)
            # 504: Gateway Timeout (网关超时)
            # 524: Cloudflare特定超时错误
            error_codes = [400, 401, 403, 408, 500, 502, 503, 504, 524]

            if isinstance(exc, (openai.InternalServerError, openai.AuthenticationError,
                              openai.PermissionDeniedError, openai.BadRequestError)):
                return True
            if isinstance(exc, openai.APIStatusError) and getattr(exc, "status_code", None) in error_codes:
                return True
            if getattr(exc, "status_code", None) in error_codes:
                return True
            if getattr(response, "status_code", None) in error_codes:
                return True

            # RuntimeError中包含错误码
            if isinstance(exc, RuntimeError):
                exc_str = str(exc)
                if any(str(code) in exc_str for code in error_codes):
                    return True

            return False

        dynamic_retry = retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_random(min=cooldown_min_s, max=cooldown_max_s),
            retry=retry_if_exception(_is_rate_limit_error),
            reraise=True,
        )
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)
    return wrapper

class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation."""
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        config_dict = global_config.__dict__
        config_dict['max_retries'] = global_config.max_retry_attempts
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(self, cache_dir, global_config, cache_filename: str = None,
                 high_throughput: bool = True,
                 **kwargs) -> None:

        super().__init__()
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_').replace(':', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self._init_llm_config()
        fake_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        if high_throughput:
            limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
            client = httpx.Client(
                limits=limits,
                timeout=httpx.Timeout(5 * 60, read=5 * 60),
                headers=fake_headers,
            )
        else:
            client = None

        self.max_retries = kwargs.get("max_retries", getattr(global_config, "max_retry_attempts", 2))
        self.rate_limit_cooldown_min_s = kwargs.get("rate_limit_cooldown_min_s", 15)
        self.rate_limit_cooldown_max_s = kwargs.get("rate_limit_cooldown_max_s", 30)

        if self.global_config.azure_endpoint is None:
            self.openai_client = OpenAI(
                base_url=self.llm_base_url,
                http_client=client,
                max_retries=0,
                default_headers=fake_headers,
            )
        else:
            self.openai_client = AzureOpenAI(
                api_version=self.global_config.azure_endpoint.split("api-version=")[1],
                azure_endpoint=self.global_config.azure_endpoint,
                max_retries=0,
                default_headers=fake_headers,
            )

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__

        config_dict['llm_name'] = self.global_config.llm_name
        config_dict['llm_base_url'] = self.global_config.llm_base_url
        config_dict['generate_params'] = {
                "model": self.global_config.llm_name,
                "max_completion_tokens": config_dict.get("max_new_tokens", 400),
                "n": config_dict.get("num_gen_choices", 1),
                "seed": config_dict.get("seed", 0),
                "temperature": config_dict.get("temperature", 0.0),
            }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response
    @dynamic_retry_decorator
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"Calling OpenAI GPT API with:\n{params}")

        if 'gpt' not in params['model'] or version.parse(openai.__version__) < version.parse("1.45.0"): # if we use vllm to call openai api or if we use openai but the version is too old to use 'max_completion_tokens' argument
            # TODO strange version change in openai protocol, but our current vllm version not changed yet
            params['max_tokens'] = params.pop('max_completion_tokens')

        response = self.openai_client.chat.completions.create(**params)

        # 处理API返回字符串的情况（某些第三方API的问题）
        if isinstance(response, str):
            logger.warning(f"API returned string instead of response object: {response[:100]}")
            # 检测HTML响应（通常是错误页面或重定向）
            if response.strip().lower().startswith(('<!doctype', '<html', '<?xml')):
                logger.error(f"API returned HTML page instead of JSON. This usually indicates API key issues, rate limiting, or service unavailability.")
                raise openai.APIError(f"API returned HTML page: {response[:200]}")

            # 尝试解析JSON
            try:
                import json
                response_dict = json.loads(response)
                response_message = response_dict.get('choices', [{}])[0].get('message', {}).get('content', response)
                metadata = {
                    "prompt_tokens": response_dict.get('usage', ).get('prompt_tokens', 0),
                    "completion_tokens": response_dict.get('usage', {}).get('completion_tokens', 0),
                    "finish_reason": response_dict.get('choices', [{}])[0].get('finish_reason', 'unknown'),
                }
            except:
                # 如果解析失败，直接使用字符串作为响应
                response_message = response
                metadata = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "finish_reason": "unknown",
                }
        else:
            choices = getattr(response, "choices", None)
            if not choices:
                # 检查是否为API服务商错误（500/524等）
                error_info = getattr(response, "error", None)
                if error_info:
                    error_code = error_info.get("code") if isinstance(error_info, dict) else None
                    error_msg = error_info.get("message") if isinstance(error_info, dict) else str(error_info)
                    logger.error(f"API服务商返回错误 (code: {error_code}): {error_msg}")
                    raise RuntimeError(f"API服务商错误 (code: {error_code}): {error_msg}。完整响应: {response}")

                logger.warning(f"OpenAI API returned empty choices: {response}")
                usage = getattr(response, "usage", None)
                response_message = ""
                metadata = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "finish_reason": "error",
                }
            else:
                choice = choices[0]
                message = getattr(choice, "message", None)
                response_message = getattr(message, "content", None)
                if not isinstance(response_message, str):
                    logger.warning("OpenAI API returned non-string message content.")
                    response_message = "" if response_message is None else str(response_message)

                usage = getattr(response, "usage", None)
                metadata = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "finish_reason": getattr(choice, "finish_reason", None) or "unknown",
                }

        return response_message, metadata
