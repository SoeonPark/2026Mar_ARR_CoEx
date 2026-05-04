import numpy as np
from evaluate import load
from sentence_transformers import SentenceTransformer, util
from typing import List

from .parsing import extract_solutions


def levenstein_distance(prompts: List[str], completions: List[str], completion_ids: List, **kwargs) -> List[float]:
    if len(completions) < 2:
        return [0.0]

    current_prompt = prompts[0]
    current_completion = completions[0]
    answers = kwargs.get("answer", [""] * len(prompts))
    current_answer = answers[0]

    similarity_scores = []

    for i in range(1, len(completions)):
        compare_completion = completions[i]
        compare_answer = answers[i] if i < len(answers) else ""

        sol1 = extract_solutions([current_prompt], [current_completion], [current_answer])
        sol2 = extract_solutions([prompts[i]], [compare_completion], [compare_answer])

        if not sol1 and not sol2:
            similarity = 1.0
        elif not sol1 or not sol2:
            similarity = 0.0
        else:
            leven = load("character")
            max_len = max(len(sol1), len(sol2))
            try:
                distance = leven.compute(predictions=[sol1], references=[sol2])["character_error_rate"] * max_len
                similarity = 1.0 - (distance / max_len)
            except Exception:
                similarity = 1.0

        similarity_scores.append(similarity)

    return similarity_scores


def get_embedding(model_name: str) -> np.ndarray:
    if not hasattr(get_embedding, "model"):
        get_embedding.model = SentenceTransformer(model_name)
    return get_embedding.model


def bert_score(prompts: List[str], completions: List[str], completion_ids: List[str], **kwargs) -> float:
    if len(completions) < 2:
        return [0.0]

    current_prompt = prompts[0]
    current_completion = completions[0]
    answers = kwargs.get("answer", [""] * len(prompts))
    current_answer = answers[0]

    similarity_scores = []

    model = get_embedding()
    print(f"  >> Loaded embedding model: {get_embedding.model.__class__.__name__.upper()}")

    for i in range(1, len(completions)):
        compare_completion = completions[i]
        compare_answer = answers[i] if i < len(answers) else ""

        sol1 = extract_solutions([current_prompt], [current_completion], [current_answer])
        sol2 = extract_solutions([prompts[i]], [compare_completion], [compare_answer])

        try:
            embeddings = model.encode([sol1, sol2], convert_to_tensor=True)
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        except Exception:
            similarity = 1.0

        similarity_scores.append(similarity)

    return similarity_scores


bleu = load("bleu")


def bleu_score(prompts: List[str], completions: List[str], other_completions: List[str], **kwargs) -> List[float]:
    all_other_completions = []
    for other_comp_i in other_completions:
        sol2 = extract_solutions([prompts[0]], [other_comp_i], [kwargs.get("answer", [""])[0]])
        if not sol2:
            sol2 = ""
        all_other_completions.append(sol2)

    similarity_scores = []
    for i in range(len(completions)):
        sol = extract_solutions([prompts[i]], [completions[i]], [kwargs.get("answer", [""])[i]])

        if not sol:
            similarity_scores.append(1.0)
        else:
            scores = bleu.compute(predictions=[sol], references=[all_other_completions])
            similarity = scores["bleu"]
            similarity_scores.append(1 - similarity)

    return similarity_scores


def one_minus_bleu_score(prompts: List[str], completions: List[str], other_completions: List[str], **kwargs) -> List[float]:
    all_other_solutions = []
    for other_comp_i in other_completions:
        sol2 = extract_solutions([prompts[0]], [other_comp_i], [kwargs.get("answer", [""])[0]])
        all_other_solutions.append(sol2 if sol2 else "")

    similarity_scores = []
    for i in range(len(completions)):
        sol = extract_solutions([prompts[i]], [completions[i]], [kwargs.get("answer", [""])[i]])

        if not sol or not all_other_solutions:
            similarity_scores.append(1.0)
        else:
            individual_bleus = []
            for other_sol in all_other_solutions:
                score = bleu.compute(predictions=[sol], references=[[other_sol]])
                individual_bleus.append(score["bleu"])

            avg_similarity = sum(individual_bleus) / len(individual_bleus)
            similarity_scores.append(1 - avg_similarity)

    return similarity_scores
