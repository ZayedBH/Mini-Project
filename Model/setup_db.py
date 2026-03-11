"""
Setup script - Run this AFTER creating tables in Supabase.

STEP 1: Create tables in Supabase dashboard
- Go to: https://app.supabase.com/project/awpzyrjgufjftzkxvazd/sql
- Copy and paste the entire contents of supabase_migrations.sql
- Run it

STEP 2: Run this script to create test user
- python setup_db.py

The test user will be: 
  Email: test@example.com
  Username: testuser
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

def setup_test_user():
    from supabase_manager import SupabaseManager
    
    db = SupabaseManager()
    
    print("Attempting to create test user...")
    
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
            print("  Make sure:")
            print("  1. Tables are created in Supabase (run supabase_migrations.sql)")
            print("  2. .env file has correct SUPABASE_URL and SUPABASE_KEY")
            return
    
    # Save test user info
    user_info = f"""Test User Information
======================
Email: {user['email']}
Username: {user['username']}
User ID: {user['id']}

Use this user_id when testing the chat API.
"""
    
    with open(Path(__file__).parent / "TEST_USER.txt", "w") as f:
        f.write(user_info)
    
    print(f"\n✓ Test user info saved to TEST_USER.txt")

if __name__ == "__main__":
    setup_test_user()
