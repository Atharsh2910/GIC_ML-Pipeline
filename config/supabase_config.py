"""Supabase table names and env configuration (no secrets here)."""

TABLE_DECISIONS = "gigshield_decisions"
TABLE_RAG_QUERIES = "gigshield_rag_queries"
TABLE_AGENT_EVENTS = "gigshield_agent_events"
TABLE_WORKERS = "gigshield_workers"

ENV_URL = "SUPABASE_URL"
ENV_SERVICE_KEY = "SUPABASE_SERVICE_ROLE_KEY"
ENV_ANON_KEY = "SUPABASE_ANON_KEY"  # optional; prefer service role for server writes
