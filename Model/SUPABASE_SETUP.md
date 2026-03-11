# Supabase Setup Guide

Follow these steps to set up user authentication and chat storage:

## Step 1: Create Tables in Supabase Dashboard

1. Open your Supabase project: https://app.supabase.com/project/awpzyrjgufjftzkxvazd/sql
2. Click "New Query" 
3. Copy the entire contents of `supabase_migrations.sql` (in this folder)
4. Paste it into the SQL editor
5. Click "Run" or press `Ctrl+Enter`

You should see success messages for all table creations.

## Step 2: Create Test User

In the Model folder, run:

```powershell
python setup_db.py
```

Expected output:
```
Attempting to create test user...
✓ Test user created successfully!
  Email: test@example.com
  Username: testuser
  User ID: [some-uuid]

✓ Test user info saved to TEST_USER.txt
```

If you see an error about tables not existing, go back to Step 1 and make sure the SQL ran successfully.

## Step 3: Start the Server

```powershell
cd Model
python server.py
```

You should see:
```
All keys matched successfully
Loading model from: ...
Server running at http://127.0.0.1:8000
```

## Step 4: Test the API

### Login (Get User ID)
```bash
curl http://127.0.0.1:8000/api/login
```

Response:
```json
{
  "success": true,
  "user": {
    "id": "your-user-id-here",
    "email": "test@example.com",
    "username": "testuser"
  }
}
```

Save the `user.id` - you'll need it for the next step.

### Start a New Chat
```bash
curl -X POST http://127.0.0.1:8000/api/new-conversation \
  -H "Content-Type: application/json" \
  -d '{"user_id": "your-user-id-here"}'
```

Response:
```json
{
  "conversation_id": "abc12345",
  "user_id": "your-user-id-here"
}
```

### Send a Message
```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "your-user-id-here",
    "conversation_id": "abc12345",
    "messages": [
      {"role": "user", "content": "how do i sort a list in python?"}
    ]
  }'
```

Response:
```json
{
  "response": "def sort_list(lst):\n    return sorted(lst)",
  "conversation_id": "abc12345"
}
```

### Get All Conversations
```bash
curl -X POST http://127.0.0.1:8000/api/conversations \
  -H "Content-Type: application/json" \
  -d '{"user_id": "your-user-id-here"}'
```

## Troubleshooting

**Error: "Test user not found"**
- Run `python setup_db.py` first

**Error: "User not found"**
- Make sure you're using the correct user_id from `/api/login`

**Error: "Conversation not found"**
- Make sure you created a conversation first with `/api/new-conversation`

**Error: "SUPABASE_URL and SUPABASE_KEY must be set"**
- Check your `.env` file in the parent directory has both values

## Files Created

- `supabase_manager.py` - Database connection and operations
- `setup_db.py` - Script to create test user
- `supabase_migrations.sql` - SQL to create tables
- `TEST_USER.txt` - Test user credentials (created after setup_db.py)
