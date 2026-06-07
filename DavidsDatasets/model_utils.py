# model_utils.py - Model and tokenizer loading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import get_model_name, get_model_label, MODEL_VARIANT, MODEL_FAMILY


def get_device():
    """Check GPU availability and return device info."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        n_gpus = torch.cuda.device_count()
        print(f"GPUs available: {n_gpus}")
        for i in range(n_gpus):
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")
    return device


def load_model_and_tokenizer(model_device: str = "cuda:0"):
    """
    Load the model and tokenizer.

    Small models (≤7B) are pinned to a single GPU.
    Large models (>7B, e.g. Qwen3-35B) use device_map='auto' to shard
    across all available GPUs automatically.
    """
    from config import MODEL_FAMILY
    model_name = get_model_name()

    large_model_families = {"qwen3", "gemma4", "gptoss", "llama4scout"}
    use_auto_device_map = MODEL_FAMILY in large_model_families
    device = "auto" if use_auto_device_map else model_device
    print(f"Loading: {model_name} → device_map={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if MODEL_FAMILY in ("qwen3", "gptoss", "llama4scout") else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        use_safetensors=True,
    )
    model.eval()
    print(f"Model loaded successfully: {get_model_label()} on {device}")

    return model, tokenizer


def generate_simple_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    base_suffix: str = "\n\nResponse:",
    enable_thinking=None,
    loop_guard: bool = True,
) -> str:
    """Format a prompt with the chat template (instruct) or a suffix (base) and generate.

    enable_thinking: optional kwarg passed to apply_chat_template for Qwen3-class
    reasoning models. Pass False on short rating/extraction calls so the model
    skips the <think> block and spends the budget on the actual response.
    Tokenizers that don't accept the kwarg silently fall back.

    loop_guard: when True (default), applies no_repeat_ngram_size=3 on base/GPT-OSS
    to prevent countdown-style loops. Pass False for forced-answer calls (max_new_tokens
    ≤ 32 — too short to loop) because the guard checks the ENTIRE input+output sequence
    and bans tokens that would complete any 3-gram already in the dense reasoning context,
    which can leave EOS as the only unbanned option and return an empty string.
    """
    from config import MODEL_VARIANT
    if MODEL_VARIANT == "instruct":
        messages = [{"role": "user", "content": prompt}]
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        try:
            formatted = tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            formatted = tokenizer.apply_chat_template(messages, **template_kwargs)
    else:
        formatted = prompt + base_suffix
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    if loop_guard and (MODEL_VARIANT == "base" or MODEL_FAMILY == "gptoss"):
        gen_kwargs["no_repeat_ngram_size"] = 3
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(
        outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()