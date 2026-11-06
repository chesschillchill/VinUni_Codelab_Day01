"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of November 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-6.1-sol": {"input": 2.00, "output": 10.00},
    "gpt-6-luna": {"input": 0.10, "output": 0.50},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-6.1-sol"
OPENAI_MINI_MODEL = "gpt-6-luna"
GEMINI_MODEL = "gemini-3.8-flash"
ANTHROPIC_MODEL = "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-6.1 Sol)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-6.1-sol).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    # TODO: Import OpenAI, instantiate client, call chat.completions.create with parameters,
    #       measure execution start/end time, extract text and token usage, and return them.

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        api_key = "mock-key"

    client = OpenAI(api_key=api_key, base_url=base_url)
    if not base_url:
        client = OpenAI(api_key=api_key)

    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    latency = time.time() - start

    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0
    response_text = response.choices[0].message.content
    if response_text is None:
        response_text = ""

    return (
        response_text,
        latency,
        {"input_tokens": input_tokens, "output_tokens": output_tokens},
    )


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 3.8 Flash
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 3.8 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-3.8-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        # Configure using types.GenerateContentConfig
        # Extract 'input_tokens' and 'output_tokens' from response.usage_metadata
    """
    # TODO: Initialize Gemini client, set config parameters, call generate_content,
    #       measure latency, extract response text and usage metadata, and return the tuple.

    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL")

    # If base_url is set, use OpenAI-compatible API (for gateways like llmgate)
    if base_url:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        start = time.time()

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        latency = time.time() - start

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        response_text = response.choices[0].message.content or ""

        return (
            response_text,
            latency,
            {"input_tokens": input_tokens, "output_tokens": output_tokens},
        )

    # Otherwise use native Gemini SDK
    from google import genai
    from google.genai import types

    if not api_key:
        api_key = "mock-key"

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        temperature=temperature, top_p=top_p, max_output_tokens=max_tokens
    )
    start = time.time()
    response = client.models.generate_content(
        model=model, contents=prompt, config=config
    )
    latency = time.time() - start

    input_tokens = (
        response.usage_metadata.prompt_token_count if response.usage_metadata else 0
    )
    output_tokens = (
        response.usage_metadata.candidates_token_count if response.usage_metadata else 0
    )
    response_text = response.text or ""

    return (
        response_text,
        latency,
        {"input_tokens": input_tokens, "output_tokens": output_tokens},
    )


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude Haiku 4.5
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude Haiku 4.5 as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-haiku-4-5).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    # TODO: Initialize Anthropic client, create message, measure latency,
    #       extract content text and usage statistics, and return the tuple.

    api_key = os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL")

    # If base_url is set, use OpenAI-compatible API (for gateways like llmgate)
    if base_url:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        start = time.time()

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        latency = time.time() - start

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        response_text = response.choices[0].message.content or ""

        return (
            response_text,
            latency,
            {"input_tokens": input_tokens, "output_tokens": output_tokens},
        )

    # Otherwise use native Anthropic SDK
    import anthropic

    if not api_key:
        api_key = "mock-key"

    client = anthropic.Anthropic(api_key=api_key)

    start = time.time()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )

    latency = time.time() - start

    input_tokens = response.usage.input_tokens if response.usage else 0
    output_tokens = response.usage.output_tokens if response.usage else 0
    response_text = response.content[0].text if response.content else ""

    return (
        response_text,
        latency,
        {"input_tokens": input_tokens, "output_tokens": output_tokens},
    )


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (GPT-6.1 Sol vs GPT-6 Luna vs Gemini 3.8 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-6.1-sol), OpenAI Luna (gpt-6-luna), and Gemini 3.8 Flash (gemini-3.8-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt_sol": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt_luna": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    # TODO: Call call_openai with gpt-6.1-sol model
    # TODO: Call call_openai with gpt-6-luna model
    # TODO: Call call_gemini with gemini-3.8-flash model
    # TODO: Calculate costs exactly based on input and output token counts using PRICING_1M_TOKENS
    #       Formula: Cost = (input_tokens * input_rate_per_1M + output_tokens * output_rate_per_1M) / 1,000,000
    # TODO: Assemble and return the comparison dictionary.

    # Call GPT-6.1 Sol
    sol_response, sol_latency, sol_usage = call_openai(prompt, model="gpt-6.1-sol")
    sol_cost = (
        sol_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-6.1-sol"]["input"]
        + sol_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-6.1-sol"]["output"]
    ) / 1_000_000

    # Call GPT-6 Luna
    luna_response, luna_latency, luna_usage = call_openai(prompt, model="gpt-6-luna")
    luna_cost = (
        luna_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-6-luna"]["input"]
        + luna_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-6-luna"]["output"]
    ) / 1_000_000

    # Call Gemini 3.8 Flash
    gemini_response, gemini_latency, gemini_usage = call_gemini(
        prompt, model="gemini-3.8-flash"
    )
    gemini_cost = (
        gemini_usage["input_tokens"] * PRICING_1M_TOKENS["gemini-3.8-flash"]["input"]
        + gemini_usage["output_tokens"]
        * PRICING_1M_TOKENS["gemini-3.8-flash"]["output"]
    ) / 1_000_000

    return {
        "gpt_sol": {
            "response": sol_response,
            "latency": sol_latency,
            "cost": sol_cost,
            "input_tokens": sol_usage["input_tokens"],
            "output_tokens": sol_usage["output_tokens"],
        },
        "gpt_luna": {
            "response": luna_response,
            "latency": luna_latency,
            "cost": luna_cost,
            "input_tokens": luna_usage["input_tokens"],
            "output_tokens": luna_usage["output_tokens"],
        },
        "gemini_flash": {
            "response": gemini_response,
            "latency": gemini_latency,
            "cost": gemini_cost,
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 3.8 Flash
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 3.8 Flash.

    Behaviour:
        - Streams response tokens from Gemini 3.8 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Use client.models.generate_content_stream() for streaming responses.
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    # TODO: Setup interactive session, prompt user for input, stream response, and update history.

    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL")

    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        return

    print("Chatbot ready. Type 'quit' or 'exit' to end.\n")

    # If base_url is set, use OpenAI-compatible streaming
    if base_url:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        history = []

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Add user message to history
            history.append({"role": "user", "content": user_input})

            # Keep only last 3 turns (6 messages)
            history = history[-6:]

            # Stream response
            print("Bot: ", end="", flush=True)
            try:
                stream = client.chat.completions.create(
                    model="gemini-3.8-flash",
                    messages=history,
                    stream=True,
                )

                full_response = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        full_response += content

                print()  # New line after response

                # Add assistant response to history
                history.append({"role": "assistant", "content": full_response})

            except Exception as e:
                print(f"\nError: {e}")
                # Remove the failed user message from history
                history = history[:-1]
        return

    # Otherwise use native Gemini SDK
    from google import genai

    client = genai.Client(api_key=api_key)
    history = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Add user message to history
        history.append({"role": "user", "parts": [{"text": user_input}]})

        # Keep only last 3 turns (6 messages)
        history = history[-6:]

        # Stream response
        print("Bot: ", end="", flush=True)
        try:
            response_stream = client.models.generate_content_stream(
                model="gemini-3.8-flash", contents=history
            )

            full_response = ""
            for chunk in response_stream:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_response += chunk.text

            print()  # New line after response

            # Add assistant response to history
            history.append({"role": "model", "parts": [{"text": full_response}]})

        except Exception as e:
            print(f"\nError: {e}")
            # Remove the failed user message from history
            history = history[:-1]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    # TODO: implement retry loop with exponential backoff
    attempt = 0
    while attempt <= max_retries:
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries:
                raise e
            delay = base_delay * (2**attempt)
            time.sleep(delay)
            attempt += 1


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    # TODO: iterate over prompts, call compare_models, and inject the original "prompt".
    results = []
    for prompt in prompts:
        result = compare_models(prompt)
        result["prompt"] = prompt
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    # TODO: Build and return the formatted table string. Truncate response to 50 chars for clean display.
    lines = []
    lines.append(
        "| Prompt | Model | Response | Latency | Tokens (In/Out) | Cost (USD) |"
    )
    lines.append(
        "|--------|-------|----------|---------|-----------------|------------|"
    )

    for result in results:
        prompt_text = (
            result["prompt"][:30] + "..."
            if len(result["prompt"]) > 30
            else result["prompt"]
        )

        for model_key in ["gpt_sol", "gpt_luna", "gemini_flash"]:
            stats = result[model_key]
            response_text = (
                stats["response"][:50] + "..."
                if len(stats["response"]) > 50
                else stats["response"]
            )
            response_text = response_text.replace("\n", " ")

            lines.append(
                f"| {prompt_text} | {model_key} | {response_text} | "
                f"{stats['latency']:.2f}s | {stats['input_tokens']}/{stats['output_tokens']} | "
                f"${stats['cost']:.6f} |"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 3.8 Flash Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
