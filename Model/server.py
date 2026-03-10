import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import time
import subprocess
import sys
import threading
from datetime import datetime
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from intent_router import route_prompt


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "py_coder_final1"
INDEX_FILE = ROOT_DIR / "index.html"
CONVERSATIONS_DIR = ROOT_DIR / "conversations"
HOST = "127.0.0.1"
PORT = 8000
CUDA_DEVICE = "cuda:0"


class ConversationManager:
    """Manages persistent conversation history."""
    
    def __init__(self, storage_dir: Path = CONVERSATIONS_DIR):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.current_conversation_id = None
        self.current_history = []
    
    def new_conversation(self) -> str:
        """Start a new conversation and return its ID."""
        self.current_conversation_id = str(uuid.uuid4())[:8]
        self.current_history = []
        return self.current_conversation_id
    
    def load_conversation(self, conversation_id: str) -> bool:
        """Load an existing conversation by ID. Returns True if found."""
        conv_file = self.storage_dir / f"{conversation_id}.json"
        if conv_file.exists():
            try:
                with open(conv_file, 'r') as f:
                    data = json.load(f)
                    self.current_conversation_id = conversation_id
                    self.current_history = data.get("history", [])
                    return True
            except:
                return False
        return False
    
    def add_message(self, role: str, content: str):
        """Add a message to the current conversation."""
        if not self.current_conversation_id:
            self.new_conversation()
        
        self.current_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._save()
    
    def get_history(self) -> list:
        """Get the current conversation history."""
        return self.current_history
    
    def _save(self):
        """Save current conversation to disk."""
        if not self.current_conversation_id:
            return
        
        conv_file = self.storage_dir / f"{self.current_conversation_id}.json"
        with open(conv_file, 'w') as f:
            json.dump({
                "id": self.current_conversation_id,
                "created": datetime.now().isoformat(),
                "history": self.current_history
            }, f, indent=2)


conversation_manager = ConversationManager()


def select_torch_dtype() -> torch.dtype:
    return torch.float16


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this server but no GPU was detected. "
            "Install a CUDA-enabled PyTorch build and run on an NVIDIA GPU."
        )


def load_model_and_tokenizer(model_path: Path):
    require_cuda()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except AttributeError:
        # Some exported tokenizer configs store extra_special_tokens as a list,
        # but newer Transformers expects a dict-like value. Override to proceed.
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            extra_special_tokens=None,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=select_torch_dtype(),
        device_map=CUDA_DEVICE,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    return tokenizer, model


def build_instruction_prompt(messages):
    # Fine-tuning was done on instruction-style text, not chat-template format.
    user_text = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            user_text = (message.get("content") or "").strip()
            break
    return user_text


def is_probable_python_code(text: str) -> bool:
    stripped = text.strip()
    if not stripped or "```" in stripped:
        return False

    code_markers = [
        "def ",
        "class ",
        "import ",
        "from ",
        "return ",
        "for ",
        "while ",
        "if ",
        "elif ",
        "else:",
        "try:",
        "except ",
        "print(",
    ]

    marker_hits = sum(1 for marker in code_markers if marker in stripped)
    has_newlines = "\n" in stripped
    has_indentation = "\n    " in stripped or "\n\t" in stripped

    # Require multiple code signals to avoid wrapping normal prose.
    return (marker_hits >= 2 and has_newlines) or (marker_hits >= 1 and has_indentation)


def format_response_for_ui(text: str) -> str:
    cleaned = (text or "").strip()
    if is_probable_python_code(cleaned):
        return f"```python\n{cleaned}\n```"
    return cleaned


def generate_reply(tokenizer, model, messages, max_new_tokens=256):
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
    return format_response_for_ui(decoded)


class ChatHandler(BaseHTTPRequestHandler):
    tokenizer = None
    model = None

    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status_code, html_text):
        body = html_text.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ["/", "/index.html"]:
            if not INDEX_FILE.exists():
                self._send_html(404, "index.html not found")
                return

            self._send_html(200, INDEX_FILE.read_text(encoding="utf-8"))
            return

        if self.path == "/api/health":
            model_device = str(getattr(self.model, "device", "unknown"))
            self._send_json(
                200,
                {
                    "ok": True,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "model_device": model_device,
                },
            )
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/chat":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_data = self.rfile.read(content_length)
            data = json.loads(raw_data.decode("utf-8"))

            messages = data.get("messages", [])
            if not isinstance(messages, list) or len(messages) == 0:
                self._send_json(400, {"error": "messages must be a non-empty list"})
                return

            max_new_tokens = int(data.get("max_new_tokens", 512))
            max_new_tokens = max(1, min(max_new_tokens, 2048))

            answer = generate_reply(
                self.tokenizer,
                self.model,
                messages,
                max_new_tokens=max_new_tokens,
            )

            self._send_json(200, {"response": answer})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    require_cuda()
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Loading model from: {MODEL_DIR}")
    tokenizer, model = load_model_and_tokenizer(MODEL_DIR)
    ChatHandler.tokenizer = tokenizer
    ChatHandler.model = model

    server = ThreadingHTTPServer((HOST, PORT), ChatHandler)
    print(f"Server running at http://{HOST}:{PORT}")
    print("Open that URL in your browser to use the chat UI.")
    server.serve_forever()


if __name__ == "__main__":
    main()
