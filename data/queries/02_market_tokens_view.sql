-- Drop existing view/table if it exists
DROP VIEW IF EXISTS market_tokens;
DROP TABLE IF EXISTS market_tokens;

-- Create market_tokens as a VIEW referencing markets
-- This view automatically reflects changes in the markets table
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
