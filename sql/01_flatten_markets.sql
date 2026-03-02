PRAGMA foreign_keys = ON;

BEGIN TRANSACTION;

DROP TABLE IF EXISTS markets;

CREATE TABLE markets (
  market_id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL,

  question TEXT,
  market_slug TEXT,
  created_at TEXT,
  updated_at TEXT,
  end_date TEXT,
  closed_time TEXT,
  active INTEGER,
  closed INTEGER,

  volume_num REAL,
  liquidity_num REAL,
  best_bid REAL,
  best_ask REAL,

  outcomes_json TEXT,
  outcome_prices_json TEXT,

  uma_resolution_status TEXT,
  updated_by INTEGER,

  FOREIGN KEY(event_id) REFERENCES events(id)
);

INSERT INTO markets (
  market_id, event_id,
  question, market_slug,
  created_at, updated_at, end_date, closed_time,
  active, closed,
  volume_num, liquidity_num, best_bid, best_ask,
  outcomes_json, outcome_prices_json,
  uma_resolution_status, updated_by
)
SELECT
  json_extract(m.value, '$.id')                         AS market_id,
  e.id                                                 AS event_id,

  json_extract(m.value, '$.question')                  AS question,
  json_extract(m.value, '$.slug')                      AS market_slug,

  json_extract(m.value, '$.createdAt')                 AS created_at,
  json_extract(m.value, '$.updatedAt')                 AS updated_at,
  json_extract(m.value, '$.endDate')                   AS end_date,
  json_extract(m.value, '$.closedTime')                AS closed_time,

  CAST(json_extract(m.value, '$.active') AS INTEGER)   AS active,
  CAST(json_extract(m.value, '$.closed') AS INTEGER)   AS closed,

  CAST(json_extract(m.value, '$.volumeNum') AS REAL)   AS volume_num,
  CAST(json_extract(m.value, '$.liquidityNum') AS REAL)AS liquidity_num,
  CAST(json_extract(m.value, '$.bestBid') AS REAL)     AS best_bid,
  CAST(json_extract(m.value, '$.bestAsk') AS REAL)     AS best_ask,

  json_extract(m.value, '$.outcomes')                  AS outcomes_json,
  json_extract(m.value, '$.outcomePrices')             AS outcome_prices_json,

  json_extract(m.value, '$.umaResolutionStatus')       AS uma_resolution_status,
  CAST(json_extract(m.value, '$.updatedBy') AS INTEGER)AS updated_by

FROM events e
JOIN json_each(e.markets) m
WHERE e.markets IS NOT NULL
  AND e.markets != ''
  AND json_valid(e.markets)
  AND json_extract(m.value, '$.id') IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_markets_event_id ON markets(event_id);
CREATE INDEX IF NOT EXISTS idx_markets_end_date ON markets(end_date);

-- Sanity checks (run these before COMMIT)
-- SELECT COUNT(*) AS n_markets FROM markets;

-- If happy:
-- COMMIT;

-- If something looks wrong:
-- ROLLBACK;