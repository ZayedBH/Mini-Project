# This script reapplies the remaining changes from our session
# Run this after the initial commit

import re

# Changes to server.py
server_py_path = "Model/server.py"

with open(server_py_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update build_instruction_prompt function
old_build = '''def build_instruction_prompt(messages):
    # Fine-tuning was done on instruction-style text, not chat-template format.
    user_text = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            user_text = (message.get("content") or "").strip()
            break
    return user_text'''

new_build = '''def build_instruction_prompt(messages, include_history=True):
    # Fine-tuning was done on instruction-style text, not chat-template format.
    user_text = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            user_text = (message.get("content") or "").strip()
            break
    
    # Include relevant conversation history as context
    if include_history and user_text:
        history = conversation_manager.get_history()
        if len(history) > 1:  # More than just the current message
            # Include last 3 exchanges for context (up to 6 messages)
            context_msgs = history[-6:-1]  # Exclude current message
            if context_msgs:
                context = "Previous context:\\n"
                for msg in context_msgs:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    context += f"{role}: {msg['content'][:100]}...\\n" if len(msg['content']) > 100 else f"{role}: {msg['content']}\\n"
                context += f"\\nCurrent request: {user_text}"
                return context
    
    return user_text'''

content = content.replace(old_build, new_build)

# 2. Add extract_python_code_and_explanation function before generate_reply
extract_func = '''def extract_python_code_and_explanation(text: str) -> str:
    """Extract only Python code from response, discard explanations.
    
    Returns only the Python code without any text explanations.
    Removes non-Python languages and preamble text.
    """
    import re
    text = (text or "").strip()
    
    # Look for Python code blocks with markdown formatting
    py_blocks = re.findall(r"```(?:python)?\\n(.*?)\\n```", text, re.DOTALL)
    
    if py_blocks:
        # Return only the first Python code block
        return py_blocks[0].strip()
    
    # No code blocks found, extract code by finding first code marker
    code_markers = ["def ", "class ", "import ", "from ", "for ", "while "]
    earliest_pos = len(text)
    
    for marker in code_markers:
        pos = text.find(marker)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
    
    # If we found a code marker, start from there
    if earliest_pos < len(text):
        code_start = text.rfind('\\n', 0, earliest_pos)
        if code_start == -1:
            code_start = 0
        else:
            code_start += 1
        return text[code_start:].strip()
    
    # Fallback: return if it looks like code
    if is_probable_python_code(text):
        return text.strip()
    
    # Return as-is if no code detected
    return text


'''

# Find the position to insert this function (before generate_reply)
generate_reply_pos = content.find("def generate_reply(")
if generate_reply_pos > 0 and "extract_python_code_and_explanation" not in content:
    content = content[:generate_reply_pos] + extract_func + content[generate_reply_pos:]

# 3. Update generate_reply function
old_gen = '''def generate_reply(tokenizer, model, messages, max_new_tokens=256):
    prompt = build_instruction_prompt(messages)
    if not prompt:
        return "Please enter a message."

    route = route_prompt(prompt)
    if route == "greeting":
        return "Hi! I can help with Python coding tasks."
    if route == "out_of_scope":
        return "Out of Scope"

    model_inputs = tokenizer([prompt], return_tensors="pt")
    model_inputs = {k: v.to(CUDA_DEVICE) for k, v in model_inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = model_inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return format_response_for_ui(decoded)'''

new_gen = '''def generate_reply(tokenizer, model, messages, max_new_tokens=256):
    prompt = build_instruction_prompt(messages)
    if not prompt:
        return "Please enter a message."

    route = route_prompt(prompt)
    if route == "greeting":
        return "Hi! I can help with Python coding tasks."
    if route == "out_of_scope":
        return "Out of Scope"

    # Add subtle instruction to response format
    if route == "valid_intent":
        prompt = f"{prompt}\\n\\nProvide a brief explanation if needed, then the Python solution."

    model_inputs = tokenizer([prompt], return_tensors="pt")
    model_inputs = {k: v.to(CUDA_DEVICE) for k, v in model_inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = model_inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # For valid intents, return only the Python code (no explanations)
    if route == "valid_intent":
        code_only = extract_python_code_and_explanation(decoded)
        result = format_response_for_ui(code_only)
    else:
        result = format_response_for_ui(decoded)
    
    # Save to conversation history
    user_msg = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            user_msg = message.get("content", "")
            break
    
    if user_msg:
        conversation_manager.add_message("user", user_msg)
    conversation_manager.add_message("assistant", result)
    
    return result'''

content = content.replace(old_gen, new_gen)

with open(server_py_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Applied server.py changes")
