"""
Setup local SQLite database with test user.
"""
from local_db import LocalConversationManager
from pathlib import Path

def setup_local_db():
    db = LocalConversationManager(Path(__file__).parent / "conversations.db")
    
    print("Setting up local SQLite database...")
    
    # Check if test user already exists
    user = db.get_user_by_email("test@example.com")
    
    if user:
        print("✓ Test user already exists!")
        print(f"  Email: {user['email']}")
        print(f"  Username: {user['username']}")
        print(f"  User ID: {user['id']}")
    else:
        user = db.create_user("test@example.com", "testuser")
        if user:
            print("✓ Test user created successfully!")
            print(f"  Email: {user['email']}")
            print(f"  Username: {user['username']}")
            print(f"  User ID: {user['id']}")
        else:
            print("✗ Failed to create test user")
            return
    
    # Save test user info
    user_info = f"""Local SQLite Database Setup Complete
=====================================
Email: {user['email']}
Username: {user['username']}
User ID: {user['id']}

Use this user_id when testing the chat API.
Database file: conversations.db

Example API call:
curl -X POST http://127.0.0.1:8000/api/login
"""
    
    with open(Path(__file__).parent / "LOCAL_USER.txt", "w") as f:
        f.write(user_info)
    
    print(f"\n✓ Setup complete!")
    print(f"✓ Database file: conversations.db")
    print(f"✓ Test user info saved to LOCAL_USER.txt")

if __name__ == "__main__":
    setup_local_db()
