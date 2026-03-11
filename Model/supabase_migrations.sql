-- Supabase Database Setup
-- Run this SQL in the Supabase SQL Editor to set up all tables

-- Create users table
CREATE TABLE IF NOT EXISTS public.users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  username TEXT UNIQUE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create conversations table
CREATE TABLE IF NOT EXISTS public.conversations (
  id TEXT PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create messages table
CREATE TABLE IF NOT EXISTS public.messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id TEXT NOT NULL REFERENCES public.conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (to avoid errors)
DROP POLICY IF EXISTS "Allow public access to users" ON public.users;
DROP POLICY IF EXISTS "Allow public to insert users" ON public.users;
DROP POLICY IF EXISTS "Allow public access to conversations" ON public.conversations;
DROP POLICY IF EXISTS "Allow public to insert conversations" ON public.conversations;
DROP POLICY IF EXISTS "Allow public access to messages" ON public.messages;
DROP POLICY IF EXISTS "Allow public to insert messages" ON public.messages;

-- Create RLS policies (allowing public access for testing)
CREATE POLICY "Allow public access to users" ON public.users FOR SELECT USING (true);
CREATE POLICY "Allow public to insert users" ON public.users FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public to update users" ON public.users FOR UPDATE USING (true) WITH CHECK (true);

CREATE POLICY "Allow public access to conversations" ON public.conversations FOR SELECT USING (true);
CREATE POLICY "Allow public to insert conversations" ON public.conversations FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public to update conversations" ON public.conversations FOR UPDATE USING (true) WITH CHECK (true);

CREATE POLICY "Allow public access to messages" ON public.messages FOR SELECT USING (true);
CREATE POLICY "Allow public to insert messages" ON public.messages FOR INSERT WITH CHECK (true);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON public.conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON public.messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON public.messages(timestamp);
