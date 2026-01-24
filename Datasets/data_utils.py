# data_utils.py - Dataset loading and answer extraction

import re
from typing import Optional
from datasets import load_dataset


def load_gsm8k():
    dataset = load_dataset("gsm8k", "main", split="test")
    print(f"Loaded {len(dataset)} test examples")
    return dataset

def load_mmlupro():
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    print(f"Loaded {len(ds)} test examples")
    return ds

def load_strategyqa():
    # Use the ChilleD version which works with newer datasets library
    ds = load_dataset("ChilleD/StrategyQA", split="test")
    print(f"Loaded {len(ds)} test examples")
    return ds

def extract_ground_truth(sample: dict, dataset: str) -> Optional[str]:
    """Extract ground truth based on dataset type."""
    if dataset == "gsm8k":
        match = re.search(r'####\s*([\d,]+)', sample['answer'])
        if match:
            return match.group(1).replace(',', '')
        return None
    
    elif dataset == "mmlupro":
        # MMLU-Pro stores answer as the letter or index
        return sample['answer']  # Usually "A", "B", etc. or index
    
    elif dataset == "strategyqa":
        # StrategyQA is boolean
        return "Yes" if sample['answer'] else "No"
    
    return None


def extract_model_answer(response: str, dataset: str) -> Optional[str]:
    """Extract model answer based on dataset type."""
    
    if dataset == "gsm8k":
        patterns = [
            r'[Tt]he answer is:?\s*\$?([\d,]+)',
            r'[Ff]inal answer:?\s*\$?([\d,]+)',
            r'=\s*\$?([\d,]+)\s*$',
            r'####\s*([\d,]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).replace(',', '')
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            return numbers[-1]
        return None
    
    elif dataset == "mmlupro":
        # Look for letter answer
        patterns = [
            r'[Tt]he answer is:?\s*\(?([A-J])\)?',
            r'[Ff]inal answer:?\s*\(?([A-J])\)?',
            r'\b([A-J])\s*(?:is correct|is the answer)',
            r'(?:correct answer is|answer is)\s*\(?([A-J])\)?',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        # Fallback: find last standalone letter A-J
        letters = re.findall(r'\b([A-J])\b', response)
        if letters:
            return letters[-1].upper()
        return None
    
    elif dataset == "strategyqa":
        # Look for Yes/No
        patterns = [
            r'[Tt]he answer is:?\s*(Yes|No)',
            r'[Ff]inal answer:?\s*(Yes|No)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()
        # Fallback
        if re.search(r'\byes\b', response.lower()):
            return "Yes"
        if re.search(r'\bno\b', response.lower()):
            return "No"
        return None
    
    return None