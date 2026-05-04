import re
from typing import Dict, List


def extract_reasoning(solution: str) -> str:
    """Extract reasoning from XML formatted text."""
    reasoning = solution.split("<reasoning>")[-1]
    reasoning = reasoning.split("</reasoning>")[0]
    return reasoning.strip()


def extract_xml_answer(text: str) -> Dict[str, str]:
    """Extract answer from XML formatted text."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> Dict[str, str]:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")


def extract_solutions(prompts: List[str], completions: List[str], answers: List[str]) -> List[str]:
    if not completions:
        return None

    completion = completions[0]

    if isinstance(completion, list) and len(completion) > 0:
        if isinstance(completion[0], dict) and "content" in completion[0]:
            extracted_solution = completion[0]["content"]
        else:
            extracted_solution = completion[0]
    elif isinstance(completion, dict) and "content" in completion:
        extracted_solution = completion["content"]
    else:
        extracted_solution = str(completion)
    return extracted_solution


def boxed_in_answer(completion: str, **kwargs) -> str:
    """Extract the final answer from the completion by looking for \\boxed{...} format."""
    match = re.search(r"\\boxed\s*\{([^}]*)\}", completion, re.DOTALL)
    if match:
        return match.group(1).strip()

    if "\\boxed{" in completion:
        parts = completion.split("\\boxed{")
        answer = parts[-1]
        answer = answer.split("}")[0]
        return answer

    return None
