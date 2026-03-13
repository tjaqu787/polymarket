-- Show count before changes
SELECT 'Market tokens count BEFORE:' AS status, COUNT(*) AS n_tokens
FROM market_tokens
WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE name='market_tokens')
UNION ALL
SELECT 'Market tokens count BEFORE:' AS status, 0 AS n_tokens
WHERE NOT EXISTS (SELECT 1 FROM sqlite_master WHERE name='market_tokens');

-- Drop existing table/view if it exists
DROP TABLE IF EXISTS market_tokens;
DROP VIEW IF EXISTS market_tokens;

-- Create market_tokens as a VIEW that derives from markets.clob_token_ids
-- This will automatically update when markets (which derives from events) is updated
CREATE VIEW market_tokens AS
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

-- Note: Can't directly index a view in SQLite
-- The underlying events table is already indexed for better performance

-- Show count after changes
SELECT 'Market tokens count AFTER:' AS status, COUNT(*) AS n_tokens FROM market_tokens;

-- Show active vs inactive counts
SELECT 'Active tokens:' AS status, COUNT(*) AS n_tokens FROM market_tokens WHERE is_active = 1
UNION ALL
SELECT 'Inactive tokens:' AS status, COUNT(*) AS n_tokens FROM market_tokens WHERE is_active = 0;
