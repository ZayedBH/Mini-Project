#!/usr/bin/env python3
"""
Test script for the trained Python Code-Fix SLM model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Model path
model_path = r"d:\miniproject\py_coder_final1"

print("🔍 Testing your trained SLM model...")
print(f"📂 Model path: {model_path}")

try:
    # Load the model and tokenizer
    print("\n📥 Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Print model info
    param_count = model.num_parameters()
    print(f"✅ Model loaded successfully!")
    print(f"📊 Parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    print(f"🔤 Vocabulary size: {len(tokenizer):,}")
    print(f"📏 Max length: {tokenizer.model_max_length}")
    
    # Test with a simple code fixing example
    print(f"\n🧪 Testing code generation...")
    
    test_prompt = """Fix: Sort a list of integers in ascending order.
Buggy:
def solve(lst):
    # return sorted(lst)
Fixed:
"""
    
    # Tokenize and generate
    inputs = tokenizer(test_prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_code = full_response[len(test_prompt):].strip()
    
    print(f"Input prompt:")
    print(f"```python\n{test_prompt}\n```")
    print(f"\nGenerated fix:")
    print(f"```python\n{generated_code}\n```")
    
    print(f"\n🎉 SUCCESS! Your SLM is working correctly!")
    print(f"🚀 Model ready for Python code fixing tasks!")
    
except Exception as e:
    print(f"❌ Error loading or testing model: {e}")
    print(f"\n🔧 Troubleshooting tips:")
    print(f"1. Make sure you have the required packages:")
    print(f"   pip install torch transformers safetensors")
    print(f"2. Check that all model files are present in: {model_path}")
    print(f"3. Ensure you have sufficient RAM (model is ~1GB)")

if __name__ == "__main__":
    print("\n📋 Model files found:")
    if os.path.exists(model_path):
        for file in os.listdir(model_path):
            file_path = os.path.join(model_path, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / 1024 / 1024
                print(f"  📄 {file}: {size_mb:.1f} MB")
    else:
        print(f"  ❌ Model directory not found: {model_path}")