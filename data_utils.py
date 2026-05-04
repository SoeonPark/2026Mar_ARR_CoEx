# ./train/data_utils.py

from datasets import load_dataset, Dataset
import pandas as pd
from rewards import extract_hash_answer

from trl import GRPOConfig, GRPOTrainer

import random
import numpy as np
import torch
import os
import json
from typing import List, Dict


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

# System prompt for Math dataset
SYSTEM_PROMPT = (
    "You are a highly intelligent math tutor. "
    "Given a math problem, provide a step-by-step solution leading to the final answer.\n The entire response should be in LaTeX format, and must not exceed 2000 tokens. "
)   

def get_gsm8k_dataset(split: str = 'train') -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    return data.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]}
            ],
            "answer": extract_hash_answer(x["answer"])
        }
    )

def get_hf_math_dataset(split: str = 'test') -> Dataset:
    data = load_dataset("HuggingFaceH4/MATH-500", "all", split=split)
    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user", 
                    "content": SYSTEM_PROMPT + "\n\n" + x["problem"]
                }
            ],
            "answer": x["solution"]
        }
    )

# System prompt for Anker Math dataset
SYS_ANK_MATH_PROMPT = (
    "Produce the final answer with '\\boxed{{}}' around it."
)


def get_anker_math_dataset(split: str = "train") -> Dataset:
    data = load_dataset("ankner/math-500", split=split)
    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": SYSTEM_PROMPT + SYS_ANK_MATH_PROMPT + "\n\n" + x["problem"]
                }
            ],
            "answer": x["solution"]
        }
    )

# SYS_OPENRS_PROMPT = (
#     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, and put your final answer within \\boxed\{\} . The reasoning process and answer areenclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Note that respond by English, NOT use other languages."
#     )

# def get_open_rs_dataset(split: str = "train") -> Dataset:
#     data = load_dataset("knoveleng/open-rs", split=split)
#     return data.map(
#         lambda x: {
#             "prompt": [
#                 {
#                     "role": "user",
#                     "content": SYS_OPENRS_PROMPT + "\n\n" + x["problem"]
#                 }
#             ],
#             "answer": f"{x['solution']}\n{x['answer']}"
#         }
#     )

from datasets import load_dataset, Dataset

SYS_OPENRS_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, "
    "and put your final answer within \\boxed{}. "  # escaped curly braces
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
    "Note that respond by English, NOT use other languages."
)

def get_open_rs_dataset(split: str = "train") -> Dataset:
    data = load_dataset("knoveleng/open-rs", split=split)

    def process_data(x):
        # prompt_messages = [
        #     {
        #         "role": "user",
        #         "content": SYS_OPENRS_PROMPT + "\n\n" + x["problem"]
        #     }
        # ]

        prompt_messages = [
            {"role": "system", "content": SYS_OPENRS_PROMPT},
            {"role": "user",   "content": x["problem"]}
        ]
        
        ans = x['answer']
        
        formatted_response = (
            f"<think>\n{x['solution']}\n</think>\n"
            f"<answer>\n{ans}\n</answer>"
        )

        return {
            "prompt": prompt_messages,      
            "answer": x['answer'], 
            "reference": formatted_response 
        }

    return data.map(process_data)

# System prompt for Code dataset
CODE_SYSTEM_PROMPT = (
    "You are a highly intelligent programming tutor. "
    "Given a programming problem, provide a step-by-step solution leading to the final answer."
)

def get_humaneval_dataset(split: str = 'test') -> Dataset:
    data = load_dataset("openai/humaneval", split=split)
    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": CODE_SYSTEM_PROMPT + "\n\n" + x["ground_truth_diagram_description"]
                }
            ],
            "answer": x["ground_truth_solution"]
        }
    )
