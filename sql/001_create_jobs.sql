-- JobMatch: jobs table + vector search setup
-- Safe to re-run: uses IF NOT EXISTS / OR REPLACE throughout.

-- 1. Enable pgvector in the dedicated "extensions" schema (Supabase convention).
create extension if not exists vector
  with schema extensions;

-- 2. Jobs table: one row per scraped/ingested job posting.
create table if not exists public.jobs (
  id bigint generated always as identity primary key,
  title text not null,                     -- job title
  company text,                            -- hiring company (optional)
  description text not null,               -- full posting text, embedded for RAG
  url text unique,                         -- original posting link; UNIQUE prevents re-inserting duplicates
  embedding extensions.vector(768),        -- intfloat/multilingual-e5-base output (768 dims)
  created_at timestamptz not null default now()
);

-- 3. HNSW index for fast approximate nearest-neighbor search using cosine distance.
--    Matches the similarity metric used in match_jobs() below.
create index if not exists jobs_embedding_hnsw_idx
  on public.jobs
  using hnsw (embedding extensions.vector_cosine_ops);

-- 4. RPC function for similarity search, callable from Python via
--    supabase.rpc("match_jobs", {"query_embedding": [...], "match_count": 5})
--    Returns the closest jobs ranked by cosine similarity (1 - cosine distance).
create or replace function public.match_jobs(
  query_embedding extensions.vector(768),
  match_count int default 5
)
returns table (
  id bigint,
  title text,
  company text,
  url text,
  similarity float
)
language sql
stable
as $$
  select
    jobs.id,
    jobs.title,
    jobs.company,
    jobs.url,
    1 - (jobs.embedding <=> query_embedding) as similarity
  from public.jobs as jobs
  where jobs.embedding is not null
  order by jobs.embedding <=> query_embedding
  limit match_count;
$$;

-- 5. Row Level Security: enabled for defense-in-depth, but no policies are
--    defined because this table is only ever accessed by the JobMatch backend
--    using the Supabase service role key, which bypasses RLS entirely.
--    If a client-facing (anon/authenticated) role ever needs direct access,
--    add explicit SELECT/INSERT policies here first.
alter table public.jobs enable row level security;

comment on table public.jobs is
  'Job postings with embeddings for RAG matching. Accessed only via the backend service role key, which bypasses RLS; no RLS policies are defined.';
