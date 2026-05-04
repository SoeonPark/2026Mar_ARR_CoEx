from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .parsing import boxed_in_answer


def correctness_reward_func_rule(prompts: list[str], completions: list[str], answer: list[str], **kwargs) -> list[float]:
    """
    Returns:
    - Base: 1.0 for correct answer
    - +0.25: proper </think> tag usage

    Max: 1.5, Min: 0.0
    """
    rewards = []

    print(f"  >> Input Lengths - Prompts: {len(prompts)}, Completions: {len(completions)}, Answers: {len(answer)}")

    for i, (prompt, completion, expected_ans) in enumerate(zip(prompts, completions, answer)):
        response = completion[0]["content"]
        reward = 0.0

        final_answer = expected_ans.strip().split("\n")[-1].strip()
        final_extracted_answer = boxed_in_answer(response)
        answer_correct = final_extracted_answer is not None and final_extracted_answer == final_answer

        if answer_correct:
            reward = 1.0

        has_think_tag = response.count("</think>") == 1
        if has_think_tag:
            reward += 0.5

        rewards.append(reward)

        print(f"  >> Completion [{i}]: {response[:50]}...{response[-50:] if len(response) > 100 else ''}")
        print(f"  >> Expected: {final_answer} | Extracted: {final_extracted_answer}")
        print(f"  >> Correct: {answer_correct} | Think: {has_think_tag}")
        print(f"  >> Reward: {reward}")

    return rewards


def correctness_reward_func(prompts: list[str], completions: list[str], solution: list[str], **kwargs) -> list[float]:
    """
    Returns both rewards and extracted answers from each completion.

    Returns:
        Tuple of (Rewards, Answer_Info):
            - Rewards: list of float rewards
            - Answer_Info: list of dicts with 'extracted_answer', 'gold_answer', 'is_correct', and 'has_think_tag'
    """
    rewards = []
    answer_info = []

    contents = [completion[0]["content"] for completion in completions]

    for i, (content, sol) in enumerate(zip(contents, solution)):
        reward = 0.0

        info = {
            "extracted_answer": None,
            "gold_answer": None,
            "is_correct": False,
            "has_think_tag": False,
        }

        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

        info["gold_answer"] = str(gold_parsed) if gold_parsed else None

        if len(gold_parsed) != 0:
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            info["extracted_answer"] = str(answer_parsed) if answer_parsed else None

            try:
                answer_correct = verify(answer_parsed, gold_parsed)
            except Exception as e:
                print(f"  >> verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                answer_correct = False
        else:
            print(f"  >> Failed to parse gold solution: {sol}")
            rewards.append(1.0)
            info["is_correct"] = True
            answer_info.append(info)
            continue

        if answer_correct:
            reward = 1.0

        has_think_tag = content.count("</think>") == 1
        info["has_think_tag"] = has_think_tag

        if has_think_tag:
            reward += 0.5

        rewards.append(reward)
        answer_info.append(info)

        print(f"  >> Completion [{i}]: {content[:50]}...{content[-50:] if len(content) > 100 else ''}")
        print(f"  >> [{i}] Correct: {answer_correct} | Think: {has_think_tag} | Reward: {reward} | Final Answer: {answer_parsed}\n\n")

    return rewards, answer_info
