-- Worker snapshots from CSV (bulk load via scripts/populate_supabase_pinecone.py)

create table if not exists public.gigshield_workers (
    worker_id bigint primary key,
    record jsonb not null,
    ingested_at timestamptz not null default now()
);

create index if not exists idx_gigshield_workers_city on public.gigshield_workers ((record->>'city'));
create index if not exists idx_gigshield_workers_disruption on public.gigshield_workers ((record->>'disruption_type'));
create index if not exists idx_gigshield_workers_slab on public.gigshield_workers ((record->>'selected_slab'));

alter table public.gigshield_workers enable row level security;
