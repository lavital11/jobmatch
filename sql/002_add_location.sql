-- JobMatch: add location to jobs table
-- Safe to re-run: uses ADD COLUMN IF NOT EXISTS.

alter table public.jobs
  add column if not exists location text;
