
-- Drop existing view if it exists
DROP VIEW IF EXISTS bets_for_timing_view;

-- Create the timing view with only necessary columns
-- This view filters markets to timing-related questions and adds semantic grouping
CREATE VIEW bets_for_timing_view AS
SELECT
  -- Market identifiers
  m.market_id,
  m.event_id,
  m.market_slug,

  -- Semantic grouping (from our normalization)
  smg.semantic_group_id,
  smg.canonical_slug,
  smg.actor,

  -- Token information
  mt.token_id,
  mt.outcome,
  mt.token_index,

  -- Market content
  m.question,
  m.category,

  -- Event metadata
  e.slug AS event_slug,
  e.title AS event_title,

  -- Dates
  SUBSTR(m.end_date, 1, 10) AS resolution_date,
  m.end_date,
  m.closed_time,

  -- Status flags
  m.active,
  m.closed,
  m.archived,

  -- Resolution data
  m.uma_resolution_status,
  m.outcome_prices_json,
  m.outcomes_json,

  -- Market metrics
  m.volume_num,
  m.liquidity_num,

  -- Legacy grouping (for backward compatibility)
  m.event_id AS market_group

FROM markets m
INNER JOIN market_tokens mt ON m.market_id = mt.market_id
LEFT JOIN semantic_market_groups smg ON m.market_id = smg.market_id
INNER JOIN events e ON m.event_id = e.id

WHERE
  -- Filter to timing-related questions
  (
    lower(m.question) LIKE '% by %'
    OR lower(m.question) LIKE '% before %'
    OR lower(m.question) LIKE '% no later than %'
    OR lower(m.question) LIKE '% until %'
  )

  -- Exclude non-timing patterns
  AND lower(m.question) NOT LIKE '% by more than %'
  AND lower(m.question) NOT LIKE '% by at least %'

  -- Exclude sports
  AND lower(m.question) NOT LIKE '%nba%'
  AND lower(m.question) NOT LIKE '%nfl%'
  AND lower(m.question) NOT LIKE '%mlb%'

  -- Exclude price/financial predictions
  AND lower(m.question) NOT LIKE '%all-time high%'
  AND lower(m.question) NOT LIKE '%points%'
  AND lower(m.question) NOT LIKE '%eth%'
  AND lower(m.question) NOT LIKE '%$%'
  AND lower(m.question) NOT LIKE '%usd%'
  AND lower(m.question) NOT LIKE '%market cap%'
  AND lower(m.question) NOT LIKE '%mcap%'

  -- Exclude other non-event categories
  AND lower(m.question) NOT LIKE '%covid%'
  AND lower(m.question) NOT LIKE '%tweet %'
  AND lower(m.question) NOT LIKE '%candidate win%'
  AND lower(m.question) NOT LIKE '% win %'
  AND lower(m.question) NOT LIKE '%rcp%'
  AND lower(m.question) NOT LIKE '% case %'
  AND lower(m.question) NOT LIKE '% cases %'
  AND lower(m.question) NOT LIKE '% epstein %'

  -- Filter to "No" outcomes only (better for hazard rate modeling)
  AND mt.outcome = 'No'
;

-- Create index on semantic_group_id for faster queries
CREATE INDEX IF NOT EXISTS idx_bets_timing_semantic_group
ON semantic_market_groups(semantic_group_id);
