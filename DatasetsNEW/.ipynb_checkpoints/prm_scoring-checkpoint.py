import pandas as pd
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from tqdm import tqdm

# ============ PRM Functions ============

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


def split_into_steps(full_response):
    """Split response by double newlines."""
    steps = full_response.strip().split("\n\n")
    steps = [s.strip() for s in steps if s.strip()]
    return steps


def get_step_rewards(model, tokenizer, question, steps, system_prompt="Please reason step by step, and put your final answer within \\boxed{}."):
    """Get PRM scores for each step."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
    ]
    
    conversation_str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    input_ids = tokenizer.encode(
        conversation_str, 
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    
    step_sep_id = tokenizer.encode("<extra_0>")[0]
    token_masks = (input_ids == step_sep_id)
    step_rewards = make_step_rewards(outputs[0], token_masks)
    
    return step_rewards[0]  # Return list of scores


# ============ Load Model ============

print("Loading PRM model...")
model_name = "Qwen/Qwen2.5-Math-PRM-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

print("Model loaded!")

# ============ Process Dataset ============

df = pd.read_csv("Steps_Qwen_Instruct_StratQA.csv")

# Split responses into steps
df['steps'] = df['full_response'].apply(split_into_steps)

# Get step rewards for each row
print("Computing step rewards...")
step_rewards_list = []
min_rewards_list = []
mean_rewards_list = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        rewards = get_step_rewards(model, tokenizer, row['question'], row['steps'])
        step_rewards_list.append(rewards)
        min_rewards_list.append(min(rewards) if rewards else None)
        mean_rewards_list.append(sum(rewards)/len(rewards) if rewards else None)
    except Exception as e:
        print(f"Error on row {idx}: {e}")
        step_rewards_list.append([])
        min_rewards_list.append(None)
        mean_rewards_list.append(None)

df['step_rewards'] = step_rewards_list
df['min_step_reward'] = min_rewards_list
df['mean_step_reward'] = mean_rewards_list

# ============ Save Results ============

df.to_csv("PRM_QwenInstruct_StratQA.csv", index=False)
print("Saved!")

# ============ Summary ============

print("\n=== Summary ===")
print(f"Mean of min step rewards: {df['min_step_reward'].mean():.4f}")
print(f"Mean of mean step rewards: {df['mean_step_reward'].mean():.4f}")

# Compare correct vs wrong
print(f"\nCorrect answers - mean step reward: {df[df['is_correct']]['mean_step_reward'].mean():.4f}")
print(f"Wrong answers - mean step reward: {df[~df['is_correct']]['mean_step_reward'].mean():.4f}")