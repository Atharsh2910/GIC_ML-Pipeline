-- Run this in Supabase SQL Editor before using the Python client.
-- https://supabase.com/dashboard/project/_/sql

create extension if not exists "pgcrypto";

create table if not exists public.gigshield_decisions (
    id uuid primary key default gen_random_uuid(),
    trace_id text not null unique,
    worker_id bigint,
    decision text not null,
    confidence double precision default 0,
    payout_amount double precision default 0,
    payload_json jsonb default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_gigshield_decisions_worker on public.gigshield_decisions (worker_id);
create index if not exists idx_gigshield_decisions_created on public.gigshield_decisions (created_at desc);

create table if not exists public.gigshield_rag_queries (
    id bigserial primary key,
    trace_id text,
    query_type text,
    snippet text,
    created_at timestamptz not null default now()
);

create index if not exists idx_gigshield_rag_trace on public.gigshield_rag_queries (trace_id);

create table if not exists public.gigshield_agent_events (
    id bigserial primary key,
    trace_id text not null,
    agent_name text not null,
    event_type text,
    payload jsonb default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_gigshield_events_trace on public.gigshield_agent_events (trace_id);

alter table public.gigshield_decisions enable row level security;
alter table public.gigshield_rag_queries enable row level security;
alter table public.gigshield_agent_events enable row level security;

-- Service role bypasses RLS; for anon/authenticated clients add policies as needed.
-- Example (optional): allow service role only — typically you use service_role key from backend only.
