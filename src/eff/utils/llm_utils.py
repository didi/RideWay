import json
import random
import re
import time
from typing import Any, Optional

from loguru import logger
import requests


from eff.config import (
    models,
    DEFAULT_MAX_RETRIES,
)
from eff.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from eff.environment.tool import Tool


class DictToObject:
    """
    Convert dictionary to object with attribute access
    Usage:
    response_obj = DictToObject(response)
    print(response_obj.choices[0].message.content)  # Instead of response["choices"][0]["message"]["content"]
    """
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            elif isinstance(value, list):
                setattr(self, key, [DictToObject(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def to_dict(self):
        """Convert object back to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictToObject):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, DictToObject) else item for item in value]
            else:
                result[key] = value
        return result


def get_response_cost(usage, model) -> float:
    num_prompt_token = usage["prompt_tokens"]
    num_completion_token = usage["completion_tokens"]
    prompt_price = models.get(model, {}).get("cost_1m_token_dollar",{}).get("prompt_price", 0)
    completion_price = models.get(model, {}).get("cost_1m_token_dollar",{}).get("completion_price", 0)
    if prompt_price and completion_price:
        return (prompt_price * num_prompt_token + completion_price * num_completion_token) / 1000000
    else:
        return 0.0


def get_response_usage(response) -> Optional[dict]:
    usage = response.get("usage", {})
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0)
    }


def format_messages(messages: list[Message]) -> list[dict]:
    messages_formatted = []
    for message in messages:
        if isinstance(message, UserMessage):
            messages_formatted.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            messages_formatted.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
            # add interleaved thinking content if exists
            if message.raw_data is not None and message.raw_data.get("message") is not None:
                reasoning_content = message.raw_data["message"].get("reasoning_content")
                if reasoning_content:
                    messages_formatted[-1]["reasoning_content"] = reasoning_content
        elif isinstance(message, ToolMessage):
            messages_formatted.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                    "name": message.name,
                }
            )
        elif isinstance(message, SystemMessage):
            messages_formatted.append({"role": "system", "content": message.content})
    return messages_formatted


def to_claude_think_official(messages_formatted: list[dict], messages: list[Message]) -> list[dict]:
    try:
        for i, (formatted_msg, original_msg) in enumerate(zip(messages_formatted, messages)):
            if formatted_msg.get("role") != "assistant":
                continue

            if not hasattr(original_msg, 'raw_data') or original_msg.raw_data is None:
                continue

            raw_message = original_msg.raw_data.get("message", {})

            # Extract reasoning content and signature
            reasoning_content = raw_message.get("reasoning_content") or raw_message.get("reasoning")
            signature = raw_message.get("signature")
            if reasoning_content:
                messages_formatted[i]["reasoning_content"] = reasoning_content
            if signature:
                messages_formatted[i]["signature"] = signature

    except Exception:
        import traceback
        traceback.print_exc()

    return messages_formatted


def to_deepseek_think_official(messages_formatted: list[dict], messages: list[Message]) -> list[dict]:
    try:
        for i, (formatted_msg, original_msg) in enumerate(zip(messages_formatted, messages)):
            if formatted_msg.get("role") != "assistant":
                continue

            if not hasattr(original_msg, 'raw_data') or original_msg.raw_data is None:
                continue

            reasoning_content = original_msg.raw_data.get("message", {}).get("reasoning_content", None) or \
                               original_msg.raw_data.get("message", {}).get("reasoning", None)

            if reasoning_content:
                messages_formatted[i]["reasoning_content"] = reasoning_content

    except Exception:
        import traceback
        traceback.print_exc()

    return messages_formatted


def _detect_think_family(model_name: str, api_model_name: str) -> str:
    """从模型名推断 think 机制族。model_name=yaml 配置名, api_model_name=实际发送给 API 的名称。

    返回: "claude" | "deepseek-v4" | "deepseek" | "o-series" | "gpt5" | "none"
    """
    # 合并两个名字做匹配（任一命中即可）
    combined = f"{model_name}|{api_model_name}".lower()
    if "claude" in combined:
        return "claude"
    # deepseek-v4 使用 thinking.enabled + reasoning_effort（区别于老版消息格式）
    if "deepseek-v4" in combined or "deepseek_v4" in combined:
        return "deepseek-v4"
    if "deepseek" in combined or "longcat" in combined:
        return "deepseek"
    if any(x in combined for x in ["/o3", "/o4", "-o3", "-o4"]):
        return "o-series"
    if "gpt-5" in combined or "gpt5" in combined:
        return "gpt5"
    if "glm" in combined:
        return "glm"  # GLM on DashScope: supports enable_thinking to toggle reasoning
    if "qwen" in combined or "mimo" in combined:
        return "qwen"  # Qwen/MiMo（DashScope 系）使用 enable_thinking: true/false，默认 think 开启需显式关闭
    if "hy3" in combined or "hunyuan" in combined:
        return "hunyuan"  # 混元 hy3 使用 thinking.type: no_think/think_high
    if "gemini" in combined:
        return "gemini"  # Gemini via OpenRouter: 用统一的 reasoning.enabled 控制思考
    return "none"


def kwargs_adapter(data: dict, enable_think: bool, messages: list,
                   model_name: str = "", api_model_name: str = "", base_url: str = "") -> dict:
    """按模型族适配 think/reasoning 参数。

    Args:
        data: 待发送的 request body
        enable_think: 用户是否开启 --enable-think
        messages: 原始 Message 对象列表（deepseek/claude 需要特殊格式转换）
        model_name: yaml 配置里的 name（如 "anthropic/claude-opus-4.6"）
        api_model_name: 实际发送给 API 的 model 字段（如 "LongCat-Flash-Chat"）
        base_url: 实际请求的 endpoint（用于区分同一模型族在不同 provider 下的 reasoning 参数格式，
                  例如同是 "qwen" 家族，DashScope 原生用 enable_thinking，经 OpenRouter 代理的
                  开源权重版本必须用统一的 reasoning.enabled，否则思考开关不生效）
    """
    family = _detect_think_family(model_name, api_model_name or data.get("model", ""))
    is_openrouter = "openrouter.ai" in (base_url or "")

    if family == "claude":
        if enable_think:
            data["messages"] = to_claude_think_official(data["messages"], messages)
            # OpenRouter / Anthropic 原生：开启 thinking + 省 token 模式
            # budget_tokens 限制思考长度，省推理成本
            # include=false 不在响应中返回思考内容，省输出 token + 带宽
            data["thinking"] = {
                "type": "enabled",
                "budget_tokens": 8000,
                "include": False,
            }
        else:
            data["thinking"] = {"type": "disabled"}

    elif family == "deepseek-v4":
        # deepseek-v4 使用 thinking.enabled + reasoning_effort（官方 API 格式）
        if enable_think:
            data["thinking"] = {"type": "enabled"}
            data["reasoning_effort"] = "max"
        else:
            data["thinking"] = {"type": "disabled"}

    elif family == "deepseek":
        if enable_think:
            data["messages"] = to_deepseek_think_official(data["messages"], messages)
        # 关闭时不需要额外操作

    elif family == "o-series":
        if enable_think:
            data.setdefault("reasoning_effort", "high")
        else:
            # think 关闭时不设 reasoning_effort，保持普通模式
            # 避免触发推理模式导致 temperature 被拒
            if "reasoning_effort" in data:
                data.pop("reasoning_effort")

    elif family == "gpt5":
        if enable_think:
            data.setdefault("reasoning_effort", "high")
        else:
            # 不设 reasoning_effort 时 GPT-5.5（经 OpenRouter）仍会默认做实质性推理。
            # reasoning_effort=minimal 只在简单请求下能把 reasoning_tokens 压到 0——
            # 复杂请求下实测仍会产生数百 reasoning tokens（未真正关闭思考）。
            # OpenRouter 统一的 reasoning.enabled=false 才是真正硬关闭，在简单和复杂
            # 请求下均验证 reasoning_tokens=0（已测，含约束堆叠的复杂 prompt）。
            data.pop("reasoning_effort", None)
            data["reasoning"] = {"enabled": False}

    elif family == "qwen":
        if is_openrouter:
            # DashScope 原生的 enable_thinking 字段经 OpenRouter 代理的开源权重版本
            # 不生效（实测 qwen/qwen3.5-9b 仍持续输出推理内容）。必须用 OpenRouter
            # 统一的 reasoning.enabled 字段才能真正关闭思考（已测 reasoning_tokens=0）。
            data["reasoning"] = {"enabled": bool(enable_think)}
        else:
            # Qwen/MiMo（DashScope 原生）使用 enable_thinking 字段控制思考模式
            if enable_think:
                data["enable_thinking"] = True
            else:
                data["enable_thinking"] = False

    elif family == "glm":
        # GLM on DashScope: use enable_thinking (same as Qwen) to disable reasoning
        if enable_think:
            data["enable_thinking"] = True
        else:
            data["enable_thinking"] = False

    elif family == "hunyuan":
        # 混元 hy3 使用 thinking.type 控制思考深度
        if enable_think:
            data["thinking"] = {"type": "think_high"}
        else:
            data["thinking"] = {"type": "no_think"}

    elif family == "gemini":
        # Gemini 3.5 系列经 OpenRouter：某些端点强制要求 reasoning 字段存在
        # （"Reasoning is mandatory for this endpoint and cannot be disabled"），
        # 直接 enabled=false 会被 400 拒绝；用 effort=minimal 可将 reasoning_tokens 压到 0，
        # 实际效果等同于关闭思考。开启时用 high。
        data["reasoning"] = {"effort": "high" if enable_think else "minimal"}

    else:
        # 不支持 think 的模型：清理可能残留的 reasoning 参数
        if "reasoning_effort" in data:
            data.pop("reasoning_effort")

    return data


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RETRIES = 8       # total extra attempts after the first try
_INITIAL_RETRY_DELAY = 3.0     # seconds before first retry
_MAX_RETRY_DELAY = 120.0       # cap on per-attempt wait
# HTTP status codes that are transient and worth retrying
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _next_delay(current: float) -> float:
    """Double with jitter, capped at _MAX_RETRY_DELAY."""
    jitter = random.uniform(0, current * 0.2)
    return min(current * 2 + jitter, _MAX_RETRY_DELAY)


# ---------------------------------------------------------------------------
# Backend-specific request/response helpers
# ---------------------------------------------------------------------------

def _http_post_with_retry(url: str, body: dict, headers: dict,
                          max_retries: int = _DEFAULT_MAX_RETRIES) -> dict:
    """POST with exponential-backoff retry on transient errors or network exceptions."""
    retry_delay = _INITIAL_RETRY_DELAY
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=(10, 600), verify=False)
            if resp.status_code not in _RETRYABLE_STATUS_CODES:
                return resp.json()
            if attempt < max_retries:
                logger.warning(f"API returned {resp.status_code}, attempt {attempt + 1}/{max_retries + 1}, retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)
                retry_delay = _next_delay(retry_delay)
            else:
                resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                logger.warning(f"Request exception, attempt {attempt + 1}/{max_retries + 1}, retrying in {retry_delay:.1f}s... Error: {e}")
                time.sleep(retry_delay)
                retry_delay = _next_delay(retry_delay)
            else:
                raise
    raise RuntimeError("Unreachable")


# --- Gemini ---

# Headers matching the testmodel (gemini_model.go)
_GEMINI_EXTRA_HEADERS = {
    "Accept": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}


def _format_gemini_messages(messages_formatted: list[dict]) -> list[dict]:
    """Convert OpenAI-format messages to Gemini `contents` format.

    System and user messages both map to the 'user' role; assistant messages
    map to 'model'.  Tool messages are not supported by this path.
    """
    contents = []
    for msg in messages_formatted:
        role = msg["role"]
        text = msg.get("content") or ""
        if role in ("system", "user"):
            contents.append({"role": "user", "parts": [{"text": text}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})
    return contents


def _parse_gemini_response(response: dict) -> tuple[Optional[str], None, dict]:
    """Parse a Gemini generateContent response.

    Returns ``(content, None, usage)`` where *usage* uses the canonical
    ``prompt_tokens`` / ``completion_tokens`` keys.
    """
    candidates = response.get("candidates", [])
    if not candidates:
        raise ValueError(
            f"Gemini response has no candidates. Full response: "
            f"{json.dumps(response, ensure_ascii=False)}"
        )
    first = candidates[0]
    finish_reason = first.get("finishReason", "")
    if finish_reason not in ("STOP", "MAX_TOKENS", ""):
        logger.warning(f"Gemini unexpected finishReason: {finish_reason}")

    parts = first.get("content", {}).get("parts", [])
    content = "".join(p.get("text", "") for p in parts if "text" in p) or None

    meta = response.get("usageMetadata", {})
    usage = {
        "prompt_tokens": meta.get("promptTokenCount", 0),
        "completion_tokens": meta.get("candidatesTokenCount", 0),
    }
    return content, None, usage


# --- Azure OpenAI ---

def _call_azure_openai(
    model_config: dict,
    messages_formatted: list[dict],
    tools_formatted: Optional[list],
    tool_choice: Optional[str],
    deployment: str,
    enable_think: bool,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> tuple[Optional[str], Optional[list], dict]:
    """Call Azure OpenAI using the openai Python SDK (matching gpt_model.go).

    Returns ``(content, tool_calls_raw, usage)``.
    """
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=model_config["azure_endpoint"],
        api_key=model_config["api_key"],
        api_version=model_config.get("api_version", "2025-01-01-preview"),
        max_retries=max_retries,
    )

    params: dict = {
        "model": deployment,
        "messages": messages_formatted,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    temperature = model_config.get("temperature", 0.0)
    top_p = model_config.get("top_p")

    # Some reasoning models (e.g. o3) don't support temperature / top_p
    _no_temp_models = {"o3"}
    if deployment not in _no_temp_models:
        params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
    else:
        max_completion_tokens = model_config.get("max_tokens", 800)
        params["max_completion_tokens"] = max_completion_tokens

    # Azure think/reasoning 支持：仅在显式开启时设 reasoning_effort
    # 关闭时不设（避免触发推理模型模式导致 temperature 被拒）
    if enable_think:
        family = _detect_think_family(deployment, model_config.get("name", ""))
        if family in ("gpt5", "o-series"):
            params["reasoning_effort"] = "high"
            # 推理模式不支持 temperature，移除
            params.pop("temperature", None)
            params.pop("top_p", None)

    if tools_formatted:
        params["tools"] = tools_formatted
        params["tool_choice"] = tool_choice or "auto"

    if model_config.get("max_tokens") and deployment not in _no_temp_models:
        params["max_tokens"] = model_config["max_tokens"]

    full_content_parts: list[str] = []
    tool_blocks: dict[int, dict] = {}
    usage: dict = {"prompt_tokens": 0, "completion_tokens": 0}

    retry_delay = _INITIAL_RETRY_DELAY
    for attempt in range(max_retries + 1):
        try:
            full_content_parts = []
            tool_blocks = {}
            usage = {"prompt_tokens": 0, "completion_tokens": 0}

            stream = client.chat.completions.create(**params)
            for chunk in stream:
                # collect usage from the final chunk
                if chunk.usage and chunk.usage.total_tokens > 0:
                    usage["prompt_tokens"] = chunk.usage.prompt_tokens
                    usage["completion_tokens"] = chunk.usage.completion_tokens

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # text content
                if delta.content:
                    full_content_parts.append(delta.content)

                # tool calls
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_blocks:
                            tool_blocks[idx] = {
                                "id": tc_delta.id or "",
                                "name": (tc_delta.function.name or "") if tc_delta.function else "",
                                "partial_json": "",
                            }
                        else:
                            if tc_delta.id:
                                tool_blocks[idx]["id"] = tc_delta.id
                            if tc_delta.function and tc_delta.function.name:
                                tool_blocks[idx]["name"] = tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_blocks[idx]["partial_json"] += tc_delta.function.arguments

            break  # success

        except Exception as e:
            from openai import APIStatusError
            is_retryable = False
            if isinstance(e, APIStatusError) and e.status_code in _RETRYABLE_STATUS_CODES:
                is_retryable = True
            elif not isinstance(e, APIStatusError):
                is_retryable = True  # network / timeout errors

            if is_retryable and attempt < max_retries:
                logger.warning(
                    f"Azure OpenAI request failed, attempt {attempt + 1}/{max_retries + 1},"
                    f" retrying in {retry_delay:.1f}s... Error: {e}"
                )
                time.sleep(retry_delay)
                retry_delay = _next_delay(retry_delay)
            else:
                raise

    content = "".join(full_content_parts) or None
    tool_calls_raw: Optional[list] = None
    if tool_blocks:
        tool_calls_raw = []
        for idx in sorted(tool_blocks):
            tc = tool_blocks[idx]
            partial = tc.get("partial_json", "")
            try:
                args = json.loads(partial) if partial else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls_raw.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(args),
                },
            })

    return content, tool_calls_raw, usage


# --- Claude ---

def _format_claude_messages(messages_formatted: list[dict]) -> tuple[str, list[dict]]:
    """Extract system prompt and convert messages to Claude Messages API format.

    Returns ``(system_prompt, claude_messages)``.  Tool call / tool result
    messages are converted to the appropriate Claude content-block format.
    """
    system_prompt = ""
    claude_messages: list[dict] = []

    for msg in messages_formatted:
        role = msg["role"]
        if role == "system":
            system_prompt = msg.get("content") or ""
        elif role == "user":
            claude_messages.append({"role": "user", "content": msg.get("content") or ""})
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                blocks: list[dict] = []
                if msg.get("content"):
                    blocks.append({"type": "text", "text": msg["content"]})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    args_raw = fn.get("arguments", "{}")
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id"),
                        "name": fn.get("name") or tc.get("name"),
                        "input": args,
                    })
                claude_messages.append({"role": "assistant", "content": blocks})
            else:
                claude_messages.append({"role": "assistant", "content": msg.get("content") or ""})
        elif role == "tool":
            result_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id"),
                "content": msg.get("content") or "",
            }
            # Merge into the trailing user message (if it already holds blocks)
            if (
                claude_messages
                and claude_messages[-1]["role"] == "user"
                and isinstance(claude_messages[-1]["content"], list)
            ):
                claude_messages[-1]["content"].append(result_block)
            else:
                claude_messages.append({"role": "user", "content": [result_block]})

    return system_prompt, claude_messages


def _process_claude_sse_event(
    event: dict,
    text_parts: list,
    active_tool_blocks: dict,
    usage: dict,
) -> None:
    """Update accumulators in-place from a single Claude SSE event."""
    etype = event.get("type", "")

    if etype == "message_start":
        u = event.get("message", {}).get("usage", {})
        usage["prompt_tokens"] = u.get("input_tokens", 0)

    elif etype == "content_block_start":
        idx = event.get("index", 0)
        block = event.get("content_block", {})
        if block.get("type") == "tool_use":
            active_tool_blocks[idx] = {
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "partial_json": "",
            }

    elif etype == "content_block_delta":
        idx = event.get("index", 0)
        delta = event.get("delta", {})
        dtype = delta.get("type", "")
        if dtype == "text_delta":
            text_parts.append(delta.get("text", ""))
        elif dtype == "input_json_delta" and idx in active_tool_blocks:
            active_tool_blocks[idx]["partial_json"] += delta.get("partial_json", "")

    elif etype == "message_delta":
        u = event.get("usage", {})
        usage["completion_tokens"] = u.get("output_tokens", 0)


def _call_claude_streaming(
    url: str,
    request_body: dict,
    headers: dict,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> tuple[Optional[str], Optional[list], dict]:
    """Call Claude Messages API with SSE streaming (matching claude_model.go).

    Returns ``(content, tool_calls_raw, usage)``.
    """
    req_body = {**request_body, "stream": True}
    stream_headers = {**headers, "Accept": "text/event-stream"}

    retry_delay = _INITIAL_RETRY_DELAY
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                url, json=req_body, headers=stream_headers,
                timeout=(10, 600), stream=True, verify=False,
            )
            if resp.status_code != 200:
                body = resp.text
                if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < max_retries:
                    logger.warning(
                        f"Claude API {resp.status_code}, attempt {attempt + 1}/{max_retries + 1}, retrying in {retry_delay:.1f}s..."
                        f" Body: {body[:200]}"
                    )
                    time.sleep(retry_delay)
                    retry_delay = _next_delay(retry_delay)
                    continue
                resp.raise_for_status()

            # --- parse SSE stream (same logic as claude_model.go) ---
            text_parts: list[str] = []
            active_tool_blocks: dict[int, dict] = {}
            current_data = ""
            usage: dict = {"prompt_tokens": 0, "completion_tokens": 0}

            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line == "":
                    if current_data:
                        try:
                            event = json.loads(current_data)
                            _process_claude_sse_event(
                                event, text_parts, active_tool_blocks, usage
                            )
                        except (json.JSONDecodeError, KeyError):
                            if '"type":"message_stop"' not in current_data:
                                logger.debug(f"Claude SSE skip: {current_data[:120]}")
                    current_data = ""
                elif raw_line.startswith("data: "):
                    current_data = raw_line[6:]
                # event:/id:/retry: lines are intentionally ignored

            content = "".join(text_parts) or None
            tool_calls_raw: Optional[list] = None
            if active_tool_blocks:
                tool_calls_raw = []
                for idx in sorted(active_tool_blocks):
                    tc = active_tool_blocks[idx]
                    partial = tc.get("partial_json", "")
                    try:
                        args = json.loads(partial) if partial else {}
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls_raw.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(args),
                        },
                    })

            return content, tool_calls_raw, usage

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                logger.warning(
                    f"Claude streaming request failed, attempt {attempt + 1}/{max_retries + 1},"
                    f" retrying in {retry_delay:.1f}s... Error: {e}"
                )
                time.sleep(retry_delay)
                retry_delay = _next_delay(retry_delay)
            else:
                raise
    raise RuntimeError("Unreachable")


# ---------------------------------------------------------------------------
# Unified generate()
# ---------------------------------------------------------------------------

def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    enable_think: bool = False,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        enable_think: Whether to enable think mode for the agent.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    try:
        if kwargs.get("num_retries") is None:
            kwargs["num_retries"] = DEFAULT_MAX_RETRIES

        messages_formatted = format_messages(messages)
        tools_formatted = [tool.openai_schema for tool in tools] if tools else None
        if tools_formatted and tool_choice is None:
            tool_choice = "auto"

        model_config = models[model]
        backend = model_config.get("backend", "openai")
        headers = model_config.get("headers", {})

        # yaml 级 think 配置：enable_think_yaml = True/False（从 models.yaml 读取）
        # 优先级：CLI --enable-think（调用方传入）> yaml enable_think 字段 > 默认 False
        # 当调用方未显式传 enable_think 时（默认 False），yaml 可覆盖
        enable_think_yaml = model_config.get("enable_think")
        if enable_think_yaml is not None and not enable_think:
            enable_think = bool(enable_think_yaml)
        # For non-OpenAI backends, strip the default Authorization header that
        # was merged in from the default config — these backends use their own
        # auth (x-api-key, crossabroadid, etc.) and the extra header causes 500.
        if backend in ("gemini", "claude", "azure"):
            headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}

        # ------------------------------------------------------------------ #
        # Gemini backend                                                       #
        # ------------------------------------------------------------------ #
        if backend == "gemini":
            url = f"{model_config['base_url']}/{model}:generateContent"
            gen_config: dict = {}
            if model_config.get("temperature") is not None:
                gen_config["temperature"] = model_config["temperature"]
            if model_config.get("top_p") is not None:
                gen_config["topP"] = model_config["top_p"]
            if model_config.get("max_tokens"):
                gen_config["maxOutputTokens"] = model_config["max_tokens"]

            request_body = {
                "contents": _format_gemini_messages(messages_formatted),
                "generationConfig": gen_config,
            }

            # Gemini 2.5+ 支持 thinkingConfig 控制推理模式
            if enable_think:
                gen_config["thinkingConfig"] = {"thinkingBudget": 8192}
            else:
                gen_config["thinkingConfig"] = {"thinkingBudget": 0}

            gemini_headers = {**headers, **_GEMINI_EXTRA_HEADERS}
            response = _http_post_with_retry(url, request_body, gemini_headers)
            content, tool_calls_raw, usage = _parse_gemini_response(response)
            raw_data = {"message": {"role": "assistant", "content": content}}

        # ------------------------------------------------------------------ #
        # Azure OpenAI backend                                                 #
        # ------------------------------------------------------------------ #
        elif backend == "azure":
            deployment = model_config.get("model_id", model)
            content, tool_calls_raw, usage = _call_azure_openai(
                model_config=model_config,
                messages_formatted=messages_formatted,
                tools_formatted=tools_formatted,
                tool_choice=tool_choice,
                deployment=deployment,
                enable_think=enable_think,
            )
            raw_data = {"message": {"role": "assistant", "content": content}}

        # ------------------------------------------------------------------ #
        # Claude backend                                                       #
        # ------------------------------------------------------------------ #
        elif backend == "claude":
            system_prompt, claude_messages = _format_claude_messages(messages_formatted)

            request_body = {
                "model": model,
                "messages": claude_messages,
                "max_tokens": model_config.get("max_tokens", 8192),
                "temperature": model_config.get("temperature", model_config["temperature"]),
            }
            if system_prompt:
                request_body["system"] = system_prompt
            if model_config.get("top_p") is not None:
                request_body["top_p"] = model_config["top_p"]

            # Tools
            if tools_formatted:
                request_body["tools"] = [
                    {
                        "name": t["function"]["name"],
                        "description": t["function"].get("description", ""),
                        "input_schema": t["function"].get("parameters", {}),
                    }
                    for t in tools_formatted
                ]
                claude_tool_choice_map = {"auto": "auto", "required": "any", "none": "none"}
                request_body["tool_choice"] = {
                    "type": claude_tool_choice_map.get(tool_choice or "auto", "auto")
                }

            # Thinking（Anthropic 直连 claude backend）
            if enable_think:
                # 省 token 模式：budget 限制思考长度，不返回思考内容
                request_body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 8000,
                }
                # 注意：Anthropic 原生 API 不支持 include=false（始终返回思考内容），
                # 但 budget_tokens 限制了思考成本；输出端由 _call_claude_streaming 过滤

            content, tool_calls_raw, usage = _call_claude_streaming(
                model_config["base_url"], request_body, headers
            )
            raw_data = {"message": {"role": "assistant", "content": content}}

        # ------------------------------------------------------------------ #
        # OpenAI-compatible backend (default / longcat / gpt-4.1 / …)        #
        # ------------------------------------------------------------------ #
        else:
            try:
                # Use model_id as the API-facing name if configured (e.g. longcat alias)
                api_model_name = model_config.get("model_id", model)
                # Only include valid API fields from model_config (not infra fields)
                _INFRA_KEYS = {"name", "model_id", "backend", "base_url", "headers",
                               "max_input_tokens", "cost_1m_token_dollar", "fallback_model_id"}
                api_params = {k: v for k, v in model_config.items()
                              if k not in _INFRA_KEYS and v is not None}
                data = {
                    "model": api_model_name,
                    "messages": messages_formatted,
                    "stream": False,
                    "temperature": kwargs.get("temperature"),
                    "tools": tools_formatted,
                    "tool_choice": tool_choice,
                }
                data.update(api_params)
                data = kwargs_adapter(data, enable_think, messages,
                                      model_name=model, api_model_name=api_model_name,
                                      base_url=model_config.get("base_url", ""))

                max_retries = _DEFAULT_MAX_RETRIES
                retry_delay = _INITIAL_RETRY_DELAY
                fallback_model_id = model_config.get("fallback_model_id")
                for attempt in range(max_retries + 1):
                    try:
                        response = requests.post(model_config["base_url"], json=data, headers=headers, timeout=(10, 600), verify=False)
                        if response.status_code not in _RETRYABLE_STATUS_CODES:
                            response = response.json()
                            break
                        # :free 档配额耗尽返回 429；配置了 fallback_model_id（同一模型的付费版）
                        # 时直接降级重发，不走指数退避
                        if (response.status_code == 429 and fallback_model_id
                                and data["model"] != fallback_model_id):
                            logger.warning(f"{data['model']} returned 429, falling back to {fallback_model_id}")
                            data["model"] = fallback_model_id
                            continue
                        if attempt < max_retries:
                            logger.warning(f"API returned {response.status_code}, attempt {attempt + 1}/{max_retries + 1}, retrying in {retry_delay:.1f}s...")
                            time.sleep(retry_delay)
                            retry_delay = _next_delay(retry_delay)
                        else:
                            response.raise_for_status()
                    except requests.exceptions.RequestException as e:
                        if attempt < max_retries:
                            logger.warning(f"Request exception, attempt {attempt + 1}/{max_retries + 1}, retrying in {retry_delay:.1f}s... Error: {e}")
                            time.sleep(retry_delay)
                            retry_delay = _next_delay(retry_delay)
                        else:
                            raise e
            except Exception as e:
                logger.error(e)
                raise e

            usage = get_response_usage(response)
            cost = get_response_cost(usage, model)
            try:
                response = response['choices'][0]
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Full response: {json.dumps(response, ensure_ascii=False, indent=2) if isinstance(response, dict) else response}")
                raise ValueError(f"Invalid API response format: {e}") from e
            assert response['message']['role'] == "assistant", (
                "The response should be an assistant message"
            )
            content = response['message'].get('content')
            tool_calls = response['message'].get('tool_calls') or []
            parsed_tool_calls = []
            for tool_call in tool_calls:
                args_raw = tool_call.get('function', {}).get('arguments')
                try:
                    args = json.loads(args_raw) if args_raw else {}
                except json.JSONDecodeError:
                    logger.warning(f"Malformed tool_call arguments from model, defaulting to {{}}: {args_raw!r}")
                    args = {}
                parsed_tool_calls.append(ToolCall(
                    id=tool_call.get('id'),
                    name=tool_call.get('function', {}).get('name'),
                    arguments=args,
                ))
            tool_calls = parsed_tool_calls
            tool_calls = tool_calls or None
            return AssistantMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
                cost=cost,
                usage=usage,
                raw_data=response,
            )

        # ------------------------------------------------------------------ #
        # Common result construction for Gemini / Claude backends             #
        # ------------------------------------------------------------------ #
        cost = get_response_cost(usage, model)
        tool_calls = None
        if tool_calls_raw:
            tool_calls = [
                ToolCall(
                    id=tc.get("id"),
                    name=tc.get("function", {}).get("name"),
                    arguments=(
                        json.loads(tc["function"]["arguments"])
                        if isinstance(tc.get("function", {}).get("arguments"), str)
                        else tc.get("function", {}).get("arguments", {})
                    ),
                )
                for tc in tool_calls_raw
            ] or None

        return AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            cost=cost,
            usage=usage,
            raw_data=raw_data,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(e)
        raise


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost
