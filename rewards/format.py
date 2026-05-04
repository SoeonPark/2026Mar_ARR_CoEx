import re

from .parsing import extract_xml_answer


def open_rs_format_reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """Reward function that checks if the copletion has a format, \\boxed{...}."""
    rewards = []
    pattern = r"\\boxed\{(.*?)\}"
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        response = completion[0]["content"]
        print(f"  >> Completion [{i}]: {response}")
        if re.search(pattern, response, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def int_reward_func(completions: str, answer: str, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(response) for response in responses]
    return [0.5 if response.isdigit() else 0.0 for response in extracted_responses]


def strict_format_reward_func(completions: str, **kwargs) -> list[float]:
    """Reward function that checks if the copletion has a specific format."""
    pattern = r"^<reasoning>\n.*?</reasoning>\n<answer>\n.*?</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions: str, **kwargs) -> list[float]:
    """Reward function that checks if the copletion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>.*?<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions: str, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(content) for content in contents]
