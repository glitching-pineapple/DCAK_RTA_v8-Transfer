# data_utils.py - Dataset loading and answer extraction

import re
from typing import Optional
from datasets import load_dataset


def load_gsm8k():
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded GSM8K: {len(dataset)} test examples")
    return dataset


def load_mmlupro():
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    print(f"Loaded MMLU-Pro: {len(ds)} test examples")
    return ds


def load_strategyqa():
    ds = load_dataset("ChilleD/StrategyQA", split="test")
    print(f"Loaded StrategyQA: {len(ds)} test examples")
    return ds


def load_medqa():
    """Load MedQA dataset (US medical licensing exam style)."""
    ds = load_dataset("bigbio/med_qa", name="med_qa_en_source", split="test")
    print(f"Loaded MedQA: {len(ds)} test examples")
    return ds


def load_triviaqa():
    """Load TriviaQA dataset."""
    ds = load_dataset("trivia_qa", "rc", split="validation")
    print(f"Loaded TriviaQA: {len(ds)} validation examples")
    return ds


def extract_ground_truth(sample: dict, dataset: str) -> Optional[str]:
    """Extract ground truth based on dataset type."""
    if dataset == "gsm8k":
        match = re.search(r'####\s*([\d,]+)', sample['answer'])
        if match:
            return match.group(1).replace(',', '')
        return None
    
    elif dataset == "mmlupro":
        return sample['answer']
    
    elif dataset == "strategyqa":
        return "Yes" if sample['answer'] else "No"
    
    elif dataset == "medqa":
        if 'answer_idx' in sample:
            return chr(65 + sample['answer_idx'])  # 0->A, 1->B, etc.
        elif 'answer' in sample:
            ans = sample['answer']
            if isinstance(ans, int):
                return chr(65 + ans)
            return str(ans)
        return None
    
    elif dataset == "triviaqa":
        if 'answer' in sample:
            answers = sample['answer']
            if isinstance(answers, dict):
                if 'value' in answers:
                    return answers['value']
                if 'aliases' in answers and answers['aliases']:
                    return answers['aliases'][0]
            elif isinstance(answers, list) and answers:
                return answers[0]
            return str(answers)
        return None
    
    return None


def extract_model_answer(response: str, dataset: str) -> Optional[str]:
    """
    Extract model answer based on dataset type.
    
    Handles common model output patterns including:
    - Clean answers: "Answer: 42"
    - Sentence answers: "Answer: The total is 42 dollars."
    - Markdown bold: "**Answer:** 42"
    - Dollar signs and commas: "Answer: $65,960"
    """
    
    if dataset == "gsm8k":
        # Priority 1: "Answer:" line - extract first number from it (handles sentences)
        # Handles: "Answer: 36", "Answer: The total is 36.", "**Answer:** 36"
        answer_match = re.search(r'\*{0,2}[Aa]nswer\*{0,2}:?\s*(.+?)(?:\n|$)', response)
        if answer_match:
            answer_text = answer_match.group(1)
            num_match = re.search(r'\$?([\d,]+)', answer_text)
            if num_match:
                return num_match.group(1).replace(',', '')
        
        # Priority 2: Common phrasing patterns
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
        
        # Priority 3: Last number in response (fallback)
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            return numbers[-1]
        return None
    
    elif dataset == "mmlupro":
        # Priority 1: "Answer:" line - extract letter
        # Handles: "Answer: B", "Answer: The answer is B.", "**Answer:** B"
        answer_match = re.search(r'\*{0,2}[Aa]nswer\*{0,2}:?\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if answer_match:
            answer_text = answer_match.group(1)
            letter_match = re.search(r'\(?([A-J])\)?', answer_text)
            if letter_match:
                return letter_match.group(1).upper()
        
        # Priority 2: Common phrasing
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
        
        # Priority 3: Last standalone letter
        letters = re.findall(r'\b([A-J])\b', response)
        if letters:
            return letters[-1].upper()
        return None
    
    elif dataset == "strategyqa":
        # Priority 1: "Answer:" line
        answer_match = re.search(r'\*{0,2}[Aa]nswer\*{0,2}:?\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if answer_match:
            answer_text = answer_match.group(1)
            yn_match = re.search(r'\b(Yes|No)\b', answer_text, re.IGNORECASE)
            if yn_match:
                return yn_match.group(1).capitalize()
        
        # Priority 2: Common phrasing
        patterns = [
            r'[Tt]he answer is:?\s*(Yes|No)',
            r'[Ff]inal answer:?\s*(Yes|No)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()
        
        # Priority 3: Fallback
        if re.search(r'\byes\b', response.lower()):
            return "Yes"
        if re.search(r'\bno\b', response.lower()):
            return "No"
        return None
    
    elif dataset == "medqa":
        # Priority 1: "Answer:" line
        answer_match = re.search(r'\*{0,2}[Aa]nswer\*{0,2}:?\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if answer_match:
            answer_text = answer_match.group(1)
            letter_match = re.search(r'\(?([A-E])\)?', answer_text)
            if letter_match:
                return letter_match.group(1).upper()
        
        # Priority 2: Common phrasing
        patterns = [
            r'[Tt]he answer is:?\s*\(?([A-E])\)?',
            r'[Cc]orrect answer:?\s*\(?([A-E])\)?',
            r'\b([A-E])\s*(?:is correct|is the answer)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Priority 3: Last standalone letter A-E
        letters = re.findall(r'\b([A-E])\b', response)
        if letters:
            return letters[-1].upper()
        return None
    
    elif dataset == "triviaqa":
        # Priority 1: "Answer:" line - take everything up to newline or Confidence/Correct
        answer_match = re.search(
            r'\*{0,2}[Aa]nswer\*{0,2}:?\s*(.+?)(?:\n|\*{0,2}[Cc]onfidence|\*{0,2}[Cc]orrect|$)', 
            response
        )
        if answer_match:
            answer = answer_match.group(1).strip().rstrip('.')
            if answer:
                return answer
        
        # Priority 2: Common phrasing
        patterns = [
            r'[Tt]he answer is:?\s*(.+?)(?:\n|$)',
            r'[Ff]inal answer:?\s*(.+?)(?:\n|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip().rstrip('.')
        return None
    
    return None


def check_triviaqa_correct(model_answer: str, sample: dict) -> bool:
    """
    Special correctness check for TriviaQA (multiple acceptable answers).
    """
    if model_answer is None:
        return False
    
    model_lower = model_answer.lower().strip()
    
    acceptable = []
    if 'answer' in sample:
        answers = sample['answer']
        if isinstance(answers, dict):
            if 'value' in answers:
                acceptable.append(answers['value'].lower())
            if 'aliases' in answers:
                acceptable.extend([a.lower() for a in answers['aliases']])
            if 'normalized_aliases' in answers:
                acceptable.extend([a.lower() for a in answers['normalized_aliases']])
        elif isinstance(answers, list):
            acceptable.extend([a.lower() for a in answers])
    
    for acc in acceptable:
        if model_lower == acc or model_lower in acc or acc in model_lower:
            return True
    
    return False
