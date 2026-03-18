import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import time
import subprocess
import sys
import threading
from datetime import datetime
import uuid
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from intent_router import route_prompt, FOLLOW_UP_RE, UNSUPPORTED_LANGUAGE_RE
from supabase_manager import SupabaseManager

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "py_coder_final1"
INDEX_FILE = ROOT_DIR / "index.html"
CONVERSATIONS_DIR = ROOT_DIR / "conversations"
HOST = "127.0.0.1"
PORT = 8000
UNSUPPORTED_LANGUAGE_RESPONSE = (
    "I can only help with Python. Please ask your coding question in Python."
)

# Auto-detect GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available, using CPU")

# Initialize Supabase
db = SupabaseManager()


class SupabaseConversationManager:
    """Manages conversation history using Supabase."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.current_conversation_id = None
        self.current_history = []
    
    def new_conversation(self) -> str:
        """Start a new conversation and return its ID."""
        self.current_conversation_id = db.create_conversation(self.user_id)
        self.current_history = []
        return self.current_conversation_id
    
    def load_conversation(self, conversation_id: str) -> bool:
        """Load an existing conversation by ID. Returns True if found."""
        # Verify conversation exists and belongs to this user
        conversation = db.get_conversation(conversation_id, self.user_id)
        if not conversation:
            return False
        
        # Load messages for this conversation (can be empty for new convos)
        messages = db.get_messages(conversation_id)
        self.current_conversation_id = conversation_id
        self.current_history = [
            {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"]
            }
            for msg in messages
        ]
        return True
    
    def add_message(self, role: str, content: str):
        """Add a message to the current conversation."""
        if not self.current_conversation_id:
            self.new_conversation()
        
        db.add_message(self.current_conversation_id, role, content)
        
        self.current_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_history(self) -> list:
        """Get the current conversation history."""
        return self.current_history


def select_torch_dtype() -> torch.dtype:
    # Use float32 for stability (GPUs may have issues with float16 on some models)
    return torch.float32


def require_cuda():
    pass  # CPU mode — no GPU required


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

    try:
        print(f"Loading model with device={DEVICE}, dtype={select_torch_dtype()}")
        
        # Try GPU loading with device_map first if CUDA available
        if DEVICE == "cuda":
            try:
                print("Attempting GPU load with device_map='auto'...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=select_torch_dtype(),
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                print(f"Model loaded successfully with device_map. Model device: {next(model.parameters()).device}")
            except Exception as gpu_error:
                print(f"GPU loading with device_map failed: {gpu_error}")
                print("Falling back to standard CPU loading...")
                raise gpu_error  # Re-raise to go to except block
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=select_torch_dtype(),
            ).to(DEVICE)
            print(f"Model loaded successfully. Model device: {next(model.parameters()).device}")
            
    except Exception as e:
        print(f"Error loading model with {DEVICE}: {e}")
        print("Loading on CPU as fallback...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=False,
            ).to("cpu")
            print(f"Model loaded on CPU. Model device: {next(model.parameters()).device}")
        except Exception as e2:
            print(f"Fallback loading also failed: {e2}")
            raise

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    return tokenizer, model


def build_instruction_prompt(messages):
    # Fine-tuning was done on instruction-style text, not chat-template format.
    # Build context from previous messages
    context = []
    has_previous_code = False
    
    for msg in messages[:-1]:  # All messages except the last one
        role = msg.get("role", "user")
        text = (msg.get("content") or "").strip()
        if text:
            # For assistant code responses, include the full code as context
            if role == "assistant" and "```" in text:
                # Extract code from code block
                code_match = text.split("```python\n")
                if len(code_match) > 1:
                    code = code_match[1].split("```")[0].strip()
                    context.append(f"[PREVIOUS CODE]\n{code}\n[END CODE]")
                    has_previous_code = True
            elif role == "user":
                # For user messages, include the question
                context.append(f"Question: {text}")
    
    # Get the latest user question
    user_text = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            user_text = (message.get("content") or "").strip()
            break
    
    # Build the final prompt
    if context:
        # If asking about previous code, make it very clear
        if has_previous_code and any(phrase in user_text.lower() for phrase in 
            ["explain", "why", "how", "what is", "what does", "make it", "change", 
             "modify", "improve", "can you", "help", "understand"]):
            full_prompt = "\n".join(context) + f"\n\nExplain the code above. Current request: {user_text}"
        else:
            full_prompt = "\n".join(context) + f"\nNew request: {user_text}"
    else:
        full_prompt = user_text
    
    return full_prompt


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
    if not cleaned:
        return ""

    # Strip training artefacts the model appends after its output.
    for suffix in ("\nEnd.", "\nEnd", "End.", "\n###", "\nBuggy:", "\nFix:"):
        if suffix in cleaned:
            cleaned = cleaned[:cleaned.index(suffix)].strip()
    
    # Recover common truncation case: model emits an opening fence without closing it.
    if cleaned.startswith("```") and cleaned.count("```") == 1:
        candidate = cleaned[3:].strip()
        if candidate.startswith("python"):
            candidate = candidate[6:].strip()
        elif candidate.startswith("py"):
            candidate = candidate[2:].strip()
        if is_probable_python_code(candidate):
            return f"```python\n{candidate}\n```"

    # Check if there's a code block in the response
    if "```" in cleaned:
        # Extract ONLY the code from code block, ignore everything else
        try:
            # Find code block
            start = cleaned.find("```")
            end = cleaned.find("```", start + 3)
            
            if start != -1 and end != -1 and end > start:
                # Extract the code between backticks
                code_block = cleaned[start + 3:end].strip()
                
                # Remove language identifier if present (e.g., "python\n")
                if code_block.startswith("python"):
                    code_block = code_block[6:].strip()
                elif code_block.startswith("py"):
                    code_block = code_block[2:].strip()
                
                code_block = code_block.strip()
                
                # If we got valid code, return ONLY the code block
                if code_block and is_probable_python_code(code_block):
                    return f"```python\n{code_block}\n```"
        except Exception as e:
            print(f"Error extracting code block: {e}")
    
    # If whole response looks like code, wrap it
    if is_probable_python_code(cleaned):
        return f"```python\n{cleaned}\n```"
    
    # For explanations/text responses, return as-is (no code)
    return cleaned


def _build_fix_prompt(instruction: str, buggy_code: str = "") -> str:
    """Build a prompt in the exact format the model was trained on."""
    return f"Fix: {instruction}\nBuggy:\n{buggy_code}\nFixed:\n"


def _looks_truncated_output(text: str) -> bool:
    s = (text or "").rstrip()
    if not s:
        return False

    # Incomplete markdown fence.
    if s.count("```") % 2 == 1:
        return True

    # Clear trailing operator / separator patterns.
    trailing_fragments = (
        "+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", "<=", ">=",
        "and", "or", "not", "(", "[", "{", ",", "\\",
    )
    if any(s.endswith(fragment) for fragment in trailing_fragments):
        return True

    # Unbalanced brackets are a strong truncation signal for code output.
    if s.count("(") > s.count(")"):
        return True
    if s.count("[") > s.count("]"):
        return True
    if s.count("{") > s.count("}"):
        return True

    return False


def _generate_segment(tokenizer, model, prompt_text: str, max_new_tokens: int) -> str:
    model_inputs = tokenizer([prompt_text], return_tensors="pt")
    model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
    eos_id = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=48,
            do_sample=False,
            repetition_penalty=1.02,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = model_inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_reply(tokenizer, model, messages, max_new_tokens=512):
    try:
        # Extract latest user message for routing.
        latest_user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                latest_user_text = (msg.get("content") or "").strip()
                break

        if not latest_user_text:
            return "Please enter a message."

        # Defensive scope guard: refuse non-Python language requests early.
        if UNSUPPORTED_LANGUAGE_RE.search(latest_user_text):
            return UNSUPPORTED_LANGUAGE_RESPONSE

        route = route_prompt(latest_user_text)
        if route == "greeting":
            return "Hi! I can help with Python coding tasks."
        if route == "unsupported_language":
            return UNSUPPORTED_LANGUAGE_RESPONSE

        # Extract the last code block from prior assistant turn (if any).
        # Cap at 1500 chars to avoid bloating the prompt beyond the model's useful range.
        prior_code = ""
        for msg in reversed(messages[:-1]):
            if msg.get("role") == "assistant":
                content = msg.get("content") or ""
                if "```python" in content:
                    parts = content.split("```python\n")
                    if len(parts) > 1:
                        prior_code = parts[1].split("```")[0].strip()[:1500]
                break

        # If the prompt references prior output, always use context path.
        is_follow_up = bool(prior_code and FOLLOW_UP_RE.search(latest_user_text))

        # Detect explain requests — reframe as "add comments" since the model
        # was only trained on code generation, not text explanations.
        EXPLAIN_RE = re.compile(
            r'\b(explain|what does|what is|how does|why does|describe)\b',
            re.IGNORECASE
        )
        is_explain = bool(EXPLAIN_RE.search(latest_user_text))

        if not is_follow_up and route == "valid_intent":
            # Fresh self-contained code generation request — no prior context needed.
            prompt = _build_fix_prompt(latest_user_text)
        elif prior_code:
            if is_explain:
                # Reframe explain as adding inline comments — within the model's capability.
                instruction = "add detailed inline comments to explain what each part of this code does"
            else:
                instruction = latest_user_text
            prompt = _build_fix_prompt(instruction, prior_code)
        else:
            # No prior code available — treat as fresh request.
            prompt = _build_fix_prompt(latest_user_text)

        decoded = _generate_segment(tokenizer, model, prompt, max_new_tokens=max_new_tokens)

        # If generation stops in the middle of code, ask the model for continuation
        # using the already generated text as prefix.
        merged = decoded
        for _ in range(2):
            if not _looks_truncated_output(merged):
                break
            continuation_seed = prompt + merged
            continuation = _generate_segment(tokenizer, model, continuation_seed, max_new_tokens=192)
            if not continuation:
                break
            if continuation in merged:
                break
            merged = (merged + "\n" + continuation).strip()

        return format_response_for_ui(merged)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generating response: {type(e).__name__}: {str(e)[:200]}"


class ChatHandler(BaseHTTPRequestHandler):
    tokenizer = None
    model = None

    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
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
            # Get actual device from model parameters
            if self.model:
                try:
                    # Try to get devices of all parameters
                    devices = set()
                    for param in list(self.model.parameters())[:10]:  # Check first 10 params
                        devices.add(str(param.device))
                    if devices:
                        model_device = ", ".join(sorted(devices))
                    else:
                        model_device = "unknown"
                except Exception:
                    model_device = "unknown"
            else:
                model_device = "unknown"
            
            self._send_json(
                200,
                {
                    "ok": True,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "configured_device": DEVICE,
                    "model_devices": model_device,
                },
            )
            return

        # Login endpoint - returns test user
        if self.path == "/api/login":
            user = db.get_user_by_email("test@example.com")
            if user:
                self._send_json(200, {
                    "success": True,
                    "user": {
                        "id": user["id"],
                        "email": user["email"],
                        "username": user["username"]
                    }
                })
            else:
                self._send_json(400, {"error": "Test user not found. Run setup_db.py first."})
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/api/chat":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_data = self.rfile.read(content_length)
                data = json.loads(raw_data.decode("utf-8"))

                # Get user ID and conversation ID from request
                user_id = data.get("user_id")
                conversation_id = data.get("conversation_id")
                messages = data.get("messages", [])

                if not user_id:
                    self._send_json(400, {"error": "user_id is required"})
                    return

                if not isinstance(messages, list) or len(messages) == 0:
                    self._send_json(400, {"error": "messages must be a non-empty list"})
                    return

                # Verify user exists
                user = db.get_user_by_id(user_id)
                if not user:
                    self._send_json(401, {"error": "User not found"})
                    return

                # Initialize conversation manager for this user
                conv_manager = SupabaseConversationManager(user_id)

                # Load conversation if provided, otherwise create new one.
                # If the provided ID no longer exists (e.g. after a server restart),
                # silently create a fresh conversation instead of returning 404.
                if conversation_id:
                    if not conv_manager.load_conversation(conversation_id):
                        conversation_id = conv_manager.new_conversation()
                else:
                    conversation_id = conv_manager.new_conversation()

                max_new_tokens = int(data.get("max_new_tokens", 512))
                max_new_tokens = max(1, min(max_new_tokens, 2048))

                answer = generate_reply(
                    self.tokenizer,
                    self.model,
                    messages,
                    max_new_tokens=max_new_tokens,
                )

                # Save user message and AI response to database
                conv_manager.add_message("user", messages[-1]["content"])
                conv_manager.add_message("assistant", answer)

                self._send_json(200, {
                    "response": answer,
                    "conversation_id": conversation_id
                })
            except Exception as exc:
                import traceback
                traceback.print_exc()
                try:
                    self._send_json(500, {"error": str(exc)[:500]})
                except Exception:
                    pass
            return

        # New conversation endpoint
        if self.path == "/api/new-conversation":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_data = self.rfile.read(content_length) if content_length > 0 else b"{}"
                data = json.loads(raw_data.decode("utf-8")) if raw_data else {}

                user_id = data.get("user_id")
                if not user_id:
                    self._send_json(400, {"error": "user_id is required"})
                    return

                user = db.get_user_by_id(user_id)
                if not user:
                    self._send_json(401, {"error": "User not found"})
                    return

                conv_manager = SupabaseConversationManager(user_id)
                conversation_id = conv_manager.new_conversation()

                self._send_json(200, {
                    "conversation_id": conversation_id,
                    "user_id": user_id
                })
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        # Get conversations endpoint
        if self.path == "/api/conversations":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_data = self.rfile.read(content_length) if content_length > 0 else b"{}"
                data = json.loads(raw_data.decode("utf-8")) if raw_data else {}

                user_id = data.get("user_id")
                if not user_id:
                    self._send_json(400, {"error": "user_id is required"})
                    return

                user = db.get_user_by_id(user_id)
                if not user:
                    self._send_json(401, {"error": "User not found"})
                    return

                conversations = db.get_conversations(user_id)
                self._send_json(200, {
                    "conversations": conversations
                })
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        # Get messages endpoint
        if self.path == "/api/messages":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_data = self.rfile.read(content_length) if content_length > 0 else b"{}"
                data = json.loads(raw_data.decode("utf-8")) if raw_data else {}

                user_id = data.get("user_id")
                conversation_id = data.get("conversation_id")
                if not user_id or not conversation_id:
                    self._send_json(400, {"error": "user_id and conversation_id are required"})
                    return

                # Verify user exists
                user = db.get_user_by_id(user_id)
                if not user:
                    self._send_json(401, {"error": "User not found"})
                    return

                # Verify conversation belongs to user
                conversation = db.get_conversation(conversation_id, user_id)
                if not conversation:
                    self._send_json(404, {"error": "Conversation not found"})
                    return

                # Get messages
                messages = db.get_messages(conversation_id)
                self._send_json(200, {
                    "messages": messages
                })
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        self._send_json(404, {"error": "Not found"})


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    print(f"Running on device: {DEVICE}")
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
