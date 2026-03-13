-- Drop existing view/table if it exists
DROP VIEW IF EXISTS market_tokens;
DROP TABLE IF EXISTS market_tokens;

BEGIN TRANSACTION;

-- Create market_tokens as a TABLE (materialized for performance)
-- This is populated from markets view, which derives from events
-- Run this script to refresh the table when events data is updated
CREATE TABLE market_tokens (
    market_id   TEXT    NOT NULL,
    token_id    TEXT    NOT NULL,
    outcome     TEXT    NOT NULL,
    token_index INTEGER NOT NULL,
    is_active   INTEGER NOT NULL DEFAULT 1,  -- Active flag for filtering in get_pricing
    PRIMARY KEY (market_id, token_index)
);

-- Populate market_tokens from markets view
INSERT INTO market_tokens (market_id, token_id, outcome, token_index, is_active)
WITH
  token_ids AS (
    SELECT
      market_id,
      json_each.key AS token_index,
      json_each.value AS token_id
    FROM markets, json_each(markets.clob_token_ids)
    WHERE markets.clob_token_ids IS NOT NULL
      AND json_valid(markets.clob_token_ids)
  ),
  outcomes AS (
    SELECT
      market_id,
      json_each.key AS outcome_index,
      json_each.value AS outcome
    FROM markets, json_each(markets.outcomes_json)
    WHERE markets.outcomes_json IS NOT NULL
      AND json_valid(markets.outcomes_json)
  )
SELECT
  t.market_id,
  t.token_id,
  o.outcome,
  CAST(t.token_index AS INTEGER) AS token_index,
  -- Active flag for filtering in get_pricing
  -- A market is considered active for pricing if:
  --   1. Not closed (closed = 0 or NULL)
  --   2. Active flag is true (active = 1)
  --   3. Not resolved (uma_resolution_status is NULL or empty)
  CASE
    WHEN COALESCE(m.closed, 0) = 0
     AND COALESCE(m.active, 0) = 1
     AND (m.uma_resolution_status IS NULL OR m.uma_resolution_status = '')
    THEN 1
    ELSE 0
  END AS is_active
FROM token_ids t
INNER JOIN outcomes o
  ON t.market_id = o.market_id
  AND t.token_index = o.outcome_index
INNER JOIN markets m
  ON t.market_id = m.market_id;

-- Create indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_token_id ON market_tokens(token_id);
CREATE INDEX IF NOT EXISTS idx_market_id ON market_tokens(market_id);
CREATE INDEX IF NOT EXISTS idx_is_active ON market_tokens(is_active);

COMMIT;

-- Show summary statistics
SELECT 'Total tokens:' AS metric, COUNT(*) AS count FROM market_tokens
UNION ALL
SELECT 'Active tokens:' AS metric, COUNT(*) AS count FROM market_tokens WHERE is_active = 1
UNION ALL
SELECT 'Inactive tokens:' AS metric, COUNT(*) AS count FROM market_tokens WHERE is_active = 0;
