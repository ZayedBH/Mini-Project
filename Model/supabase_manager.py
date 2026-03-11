import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime
import uuid

# Load environment variables from parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

class SupabaseManager:
    def __init__(self):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
    
    # ===== Users =====
    def create_user(self, email: str, username: str) -> dict:
        """Create a new user."""
        try:
            response = self.supabase.table("users").insert({
                "email": email,
                "username": username
            }).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error creating user: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> dict:
        """Get user by email."""
        try:
            response = self.supabase.table("users").select("*").eq("email", email).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error getting user: {e}")
            return None
    
    def get_user_by_id(self, user_id: str) -> dict:
        """Get user by ID."""
        try:
            response = self.supabase.table("users").select("*").eq("id", user_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error getting user: {e}")
            return None
    
    # ===== Conversations =====
    def create_conversation(self, user_id: str) -> str:
        """Create a new conversation for a user."""
        try:
            conv_id = str(uuid.uuid4())[:8]
            response = self.supabase.table("conversations").insert({
                "id": conv_id,
                "user_id": user_id
            }).execute()
            return conv_id
        except Exception as e:
            print(f"Error creating conversation: {e}")
            return None
    
    def get_conversations(self, user_id: str) -> list:
        """Get all conversations for a user."""
        try:
            response = self.supabase.table("conversations").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
            return response.data if response.data else []
        except Exception as e:
            print(f"Error getting conversations: {e}")
            return []
    
    def get_conversation(self, conversation_id: str, user_id: str) -> dict:
        """Get a specific conversation by ID (verify it belongs to user)."""
        try:
            response = self.supabase.table("conversations").select("*").eq("id", conversation_id).eq("user_id", user_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error getting conversation: {e}")
            return None
    
    # ===== Messages =====
    def add_message(self, conversation_id: str, role: str, content: str) -> bool:
        """Add a message to a conversation."""
        try:
            self.supabase.table("messages").insert({
                "conversation_id": conversation_id,
                "role": role,
                "content": content
            }).execute()
            return True
        except Exception as e:
            print(f"Error adding message: {e}")
            return False
    
    def get_messages(self, conversation_id: str) -> list:
        """Get all messages in a conversation."""
        try:
            response = self.supabase.table("messages").select("*").eq("conversation_id", conversation_id).order("timestamp", desc=False).execute()
            return response.data if response.data else []
        except Exception as e:
            print(f"Error getting messages: {e}")
            return []
