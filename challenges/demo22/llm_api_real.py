"""
LLM API 真实调用封装 (DeepSeek 适配版)
基于原代码结构，将模拟调用替换为真实的 OpenAI 兼容 API 调用
"""

import os
import time
from typing import Optional
from dotenv import load_dotenv  # [新增] 用于读取 .env 文件
from openai import OpenAI       # [新增] 引入真实的 SDK

# [新增] 加载 .env 文件中的环境变量
load_dotenv()


def call_openai_api(
    prompt: str,
    model: str = "deepseek-flash",  # [修改] 默认模型改为 deepseek-flash
    temperature: float = 0.7,
    max_tokens: int = 2000,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    api_key: Optional[str] = None
) -> dict[str, str | int]:
    """
    调用真实的 OpenAI 兼容 API（带重试机制）
    """
    # 参数验证（与原代码保持一致）
    if not prompt:
        raise ValueError("Prompt 不能为空")

    if not (0 <= temperature <= 2):
        raise ValueError("Temperature 必须在 0-2 之间")

    if max_tokens <= 0:
        raise ValueError("max_tokens 必须大于 0")

    # [新增] 获取 API Key：优先使用传入的参数，其次从环境变量读取
    final_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not final_api_key:
        raise ValueError("未找到 API Key，请在 .env 中设置 OPENAI_API_KEY 或传入 api_key 参数")

    # [新增] 获取 Base URL（兼容 DeepSeek 等国产大模型）
    base_url = os.getenv("OPENAI_BASE_URL")
    
    # [新增] 初始化真实的 OpenAI 客户端
    client = OpenAI(
        api_key=final_api_key,
        base_url=base_url,
        timeout=30.0  # 增加超时设置，防止请求卡死
    )

    # 重试逻辑（与原代码保持一致）
    for attempt in range(max_retries):
        try:
            print(f"[尝试 {attempt + 1}/{max_retries}] 调用 {model}...")

            # ==================== [修改] 真实 API 调用替换模拟调用 ====================
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # [修改] 提取真实返回的数据
            response_text = response.choices[0].message.content
            
            # [新增] 分别提取输入和输出 tokens，以便精确计算费用
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            tokens_used = response.usage.total_tokens if response.usage else 0
            # =======================================================================

            # 真实调用成功，返回标准化格式
            return {
                "response": response_text,
                "model": model,
                "tokens_used": tokens_used,
                "prompt_tokens": prompt_tokens,      # [新增] 返回输入 token 数
                "completion_tokens": completion_tokens, # [新增] 返回输出 token 数
                "success": True,
            }

        except Exception as e:
            print(f"错误: {e}")

            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print(f"达到最大重试次数 ({max_retries})")
                return {
                    "response": "",
                    "model": model,
                    "tokens_used": 0,
                    "prompt_tokens": 0,      # [新增] 失败时也返回该字段，保持结构一致
                    "completion_tokens": 0,  # [新增] 失败时也返回该字段，保持结构一致
                    "success": False,
                }

    raise RuntimeError("API 调用失败")


def format_prompt_with_context(
    user_input: str,
    context: list[str],
    system_prompt: str = "You are a helpful AI assistant."
) -> str:
    """
    格式化包含上下文的提示词（与原代码完全一致，无修改）
    """
    parts: list[str] = [f"System: {system_prompt}"]

    if context:
        parts.append("\nConversation History:")
        for i, msg in enumerate(context, 1):
            parts.append(f"{i}. {msg}")

    parts.append(f"\nUser: {user_input}")
    parts.append("Assistant:")

    return "\n".join(parts)


def estimate_token_cost(
    prompt_tokens: int,       # [修改] 参数改为输入 tokens
    completion_tokens: int,   # [修改] 参数改为输出 tokens
    model: str = "deepseek-flash" # [修改] 默认模型改为 deepseek-flash
) -> float:
    """
    [修改] 估算 API 调用成本（基于 DeepSeek 官方定价，单位：人民币 元）
    注意：此处按“高峰时段、缓存未命中”的最高价格进行保守估算。
    """
    # [修改] DeepSeek 定价表（元 / 百万 tokens）
    # deepseek-flash (DeepSeek-V4.1-Flash) 高峰、缓存未命中
    flash_prompt_price = 2.0    # 输入 2元/百万 tokens
    flash_completion_price = 8.0  # 输出 8元/百万 tokens
    
    # deepseek-v4-pro (DeepSeek-V4-Pro-0813) 高峰、缓存未命中
    pro_prompt_price = 9.0      # 输入 9元/百万 tokens
    pro_completion_price = 27.0  # 输出 27元/百万 tokens

    # [修改] 根据模型名称匹配价格
    if model == "deepseek-v4-pro":
        cost = (prompt_tokens / 1_000_000) * pro_prompt_price + \
               (completion_tokens / 1_000_000) * pro_completion_price
    else:
        # 默认按 deepseek-flash 计算
        cost = (prompt_tokens / 1_000_000) * flash_prompt_price + \
               (completion_tokens / 1_000_000) * flash_completion_price
        
    return cost


# 演示使用
def main() -> None:
    """主函数：演示真实 LLM API 调用"""
    
    #1. 真实 API 简单调用
    print("=== 真实 API 简单调用 ===")
    # [修改] 使用 deepseek-flash 模型，请确保 .env 中配置了正确的 base_url 和 api_key
    result = call_openai_api("用一句话解释什么是机器学习。", model="deepseek-flash")
    
    if result["success"]:
        print(f"响应: {result['response']}")
        # [修改] 打印详细的 token 信息
        print(f"Tokens: {result['tokens_used']} (输入: {result['prompt_tokens']}, 输出: {result['completion_tokens']})")
        
    # 2. 带上下文的调用
    print("\n=== 带上下文调用 ===")
    context = [
        "User: Hello",
        "Assistant: Hi! How can I help you?",
    ]
    prompt = format_prompt_with_context(
        "请用一句中文解释什么是人工智能（AI）。",
        context,
        "你是一个简洁的AI助手。"
    )
    result = call_openai_api(prompt, model="deepseek-flash", temperature=0.5)
    print(f"响应: {result['response']}")
    # [新增]打印 token 信息
    print(f"Tokens: {result['tokens_used']} (输入: {result['prompt_tokens']}, 输出: {result['completion_tokens']})")

    # 3. 成本估算
    print("\n=== 成本估算 ===")
    tokens = result['tokens_used']
    cost = estimate_token_cost(
        result['prompt_tokens'], 
        result['completion_tokens'], 
        model=result['model']
    )
    print(f"使用 {tokens} tokens，估算成本: ¥{cost:.6f} 元")


if __name__ == "__main__":
    main()