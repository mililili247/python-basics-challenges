"""
LLM API 调用封装
演示如何将 API 调用封装成可复用的函数
"""

from typing import Optional
import time


def call_openai_api(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    api_key: Optional[str] = None
) -> dict[str, str | int]:
    """
    调用 OpenAI API（带重试机制）

    这个函数封装了 OpenAI API 调用的复杂性，提供：
    - 自动重试
    - 错误处理
    - Token 统计
    - 标准化的响应格式

    Args:
        prompt: 输入提示词
        model: 模型名称
        temperature: 温度参数 (0-2)
        max_tokens: 最大生成 tokens
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        api_key: API 密钥（可选，从环境变量读取）

    Returns:
        包含以下键的字典:
        - 'response': LLM 响应文本
        - 'model': 使用的模型
        - 'tokens_used': 使用的 token 数
        - 'success': 是否成功

    Raises:
        ValueError: 如果参数无效
        RuntimeError: 如果达到最大重试次数仍失败

    Examples:
        >>> result = call_openai_api("What is AI?")
        >>> print(result['response'])
        'AI stands for Artificial Intelligence...'

        >>> result = call_openai_api(
        ...     "Explain quantum computing",
        ...     model="gpt-4",
        ...     temperature=0.5
        ... )
    """
    if not prompt:
        raise ValueError("Prompt 不能为空")
    if not (0 <= temperature <= 2):
        raise ValueError("Temperature 必须在 0-2 之间")
    if max_tokens <= 0:
        raise ValueError("max_tokens 必须大于 0")

    for attempt in max_retries:#1
        try:
            print(f"[尝试 {attempt + 1}/{max_retries}] 调用 {model}...)")
            response_text = print(f"响应：[模拟响应]收到提示：'{prompt}...'")#2
            tokens_used = prompt.split() + response_text.split()#3
            return{
                "response": response_text,
                "model": model,
                "token_used": tokens_used,
                "success": True,
            }

        except Exception as e:
            print(f"错误: {e}")

            if attempt < 2:#4
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                #5:print(f"达到最大重试次数 ({max_retries})")#5
                return {
                    "response": "",
                    "model": model,
                    "tokens_used": 0,
                    "success": False,
                }

    # TODO: 1. 参数校验
    # TODO: 2. 重试逻辑
    # TODO: 3. 实际 API 调用（可先用模拟响应）
    # TODO: 4. 返回统一格式
    raise NotImplementedError("请补充 call_openai_api 的实现")


def format_prompt_with_context(
    user_input: str,
    context: list[str],
    system_prompt: str = "You are a helpful AI assistant."
) -> str:
    """
    格式化包含上下文的提示词

    Args:
        user_input: 用户输入
        context: 上下文消息列表
        system_prompt: 系统提示词

    Returns:
        格式化的完整提示词
    """
    # TODO: 拼接 system_prompt、context、user_input、Assistant:
    parts: list[str] = [system_prompt]#67
    parts.append("\nConversation History:")
    for i,m in enumerate(context,i):
        print(f"{i}. {m}")#8
    parts.append(f"User: {user_input}")
    parts.append(f"Assistant: ")
    return "\n".join(parts) #9用换行拼成一个完整字符串
    raise NotImplementedError("请补充 format_prompt_with_context 的实现")


def estimate_token_cost(
    token_count: int,
    model: str = "gpt-3.5-turbo"
) -> float:
    """
    估算 API 调用成本

    Args:
        token_count: token 数量
        model: 模型名称

    Returns:
        估算成本（美元）
    """
    # TODO: 根据模型定价表计算成本
    price: dict[str, float] = {
        "gpt-3.5-turbo": 0.002,
        "gpt-4": 0.03,
        "gpt-4-turbo": 0.01,
    }

    cost = (token_count / 1000)* price.get("model",0.002)#10
    return cost
    raise NotImplementedError("请补充 estimate_token_cost 的实现")


# 演示使用
def main() -> None:
    """主函数：演示 LLM API 调用"""

    # 1. 简单调用
    print("=== 简单调用 ===")
    result = call_openai_api("What is machine learning?")
    print(f"响应: {result['response']}")
    print(f"Tokens: {result['tokens_used']}")

    # 2. 带上下文的调用
    print("\n=== 带上下文调用 ===")
    context = [
        "User: Hello",
        "Assistant: Hi! How can I help you?",
    ]
    prompt = format_prompt_with_context(
        "What is AI?",
        context,
        "You are a friendly AI tutor."
    )
    result = call_openai_api(prompt, model="gpt-4", temperature=0.5)
    print(f"响应: {result['response']}")

    # 3. 成本估算
    print("\n=== 成本估算 ===")
    tokens = result['tokens_used']
    cost = estimate_token_cost(tokens, model="gpt-4")
    print(f"使用 {tokens} tokens，估算成本: ${cost:.4f}")


if __name__ == "__main__":
    main()