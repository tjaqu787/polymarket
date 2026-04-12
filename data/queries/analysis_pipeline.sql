-- Step 1: Join timing markets with price history
DROP VIEW IF EXISTS timing_prices_long;

CREATE VIEW timing_prices_long AS
SELECT
    b.*,
    p.token_id,
    p.outcome,
    p.ts,
    p.date AS price_date,
    p.price,
    (julianday(b.end_date) - julianday(p.date)) AS days_to_deadline
FROM bets_for_timing_view b
JOIN price_history p
    ON b.market_id = p.market_id
WHERE (julianday(b.end_date) - julianday(p.date)) > 0;

-- Step 2: Filter to binary "Yes" outcomes
DROP VIEW IF EXISTS timing_prices_yes;

CREATE VIEW timing_prices_yes AS
SELECT *
FROM timing_prices_long
WHERE lower(outcome) = 'yes';

-- Step 3: Build model input table (fixed horizons)
DROP TABLE IF EXISTS timing_model_input;

CREATE TABLE timing_model_input AS
WITH targets(label, target_days) AS (
    VALUES
        ('p30', 30.0),
        ('p7',   7.0),
        ('p1',   1.0)
),
ranked AS (
    SELECT
        y.market_id,
        y.event_id,
        y.question,
        y.end_date,
        y.token_id,
        y.price_date,
        y.ts,
        y.price,
        y.days_to_deadline,
        t.label,
        t.target_days,
        ROW_NUMBER() OVER (
            PARTITION BY y.market_id, t.label
            ORDER BY ABS(y.days_to_deadline - t.target_days), y.ts DESC
        ) AS rn
    FROM timing_prices_yes y
    CROSS JOIN targets t
),
picked AS (
    SELECT *
    FROM ranked
    WHERE rn = 1
)
SELECT
    market_id,
    MAX(event_id) AS event_id,
    MAX(question) AS question,
    MAX(end_date) AS end_date,

    MAX(CASE WHEN label = 'p30' THEN price END) AS price_30d,
    MAX(CASE WHEN label = 'p30' THEN days_to_deadline END) AS days_30d,

    MAX(CASE WHEN label = 'p7' THEN price END) AS price_7d,
    MAX(CASE WHEN label = 'p7' THEN days_to_deadline END) AS days_7d,

    MAX(CASE WHEN label = 'p1' THEN price END) AS price_1d,
    MAX(CASE WHEN label = 'p1' THEN days_to_deadline END) AS days_1d

FROM picked
GROUP BY market_id;

-- Step 4: Analysis table with derived quantities
DROP VIEW IF EXISTS timing_model_analysis;

CREATE VIEW timing_model_analysis AS
SELECT
    *,
    (price_7d - price_30d) AS delta_30_to_7,
    (price_1d - price_7d)  AS delta_7_to_1,
    (price_1d - price_30d) AS delta_30_to_1,

    CASE
        WHEN price_30d > 0 AND price_30d < 1 AND days_30d > 0
        THEN -LOG(1.0 - price_30d) / days_30d
    END AS lambda_30d,

    CASE
        WHEN price_7d > 0 AND price_7d < 1 AND days_7d > 0
        THEN -LOG(1.0 - price_7d) / days_7d
    END AS lambda_7d,

    CASE
        WHEN price_1d > 0 AND price_1d < 1 AND days_1d > 0
        THEN -LOG(1.0 - price_1d) / days_1d
    END AS lambda_1d

FROM timing_model_input;

-- Building the model-input table using Tyrell’s canonical join path (token_id)
DROP VIEW IF EXISTS timing_prices_long_bft;

CREATE VIEW timing_prices_long_bft AS
SELECT
    b.market_id,
    b.event_id,
    b.question,
    b.end_date,
    b.market_group,
    b.category,
    b.event_slug,
    b.event_title,
    b.resolution_date,

    b.token_id,
    b.outcome AS bft_outcome,

    ph.ts,
    ph.date AS price_date,
    ph.price,
    (julianday(b.end_date) - julianday(ph.date)) AS days_to_deadline
FROM bets_for_timing_view b
JOIN price_history ph
    ON b.token_id = ph.token_id
WHERE lower(b.outcome) = 'yes'
  AND lower(ph.outcome) = 'yes'
  AND (julianday(b.end_date) - julianday(ph.date)) > 0;

-- Creating the wide model-input table at fixed horizons (30d, 7d, 1d)
DROP TABLE IF EXISTS timing_model_input_bft;

CREATE TABLE timing_model_input_bft AS
WITH targets(label, target_days) AS (
    VALUES
        ('p30', 30.0),
        ('p7',   7.0),
        ('p1',   1.0)
),
ranked AS (
    SELECT
        y.market_id,
        y.event_id,
        y.question,
        y.end_date,
        y.category,
        y.market_group,
        y.event_slug,
        y.event_title,
        y.resolution_date,
        y.token_id,

        y.price_date,
        y.ts,
        y.price,
        y.days_to_deadline,

        t.label,
        t.target_days,

        ROW_NUMBER() OVER (
            PARTITION BY y.token_id, t.label
            ORDER BY ABS(y.days_to_deadline - t.target_days), y.ts DESC
        ) AS rn
    FROM timing_prices_long_bft y
    CROSS JOIN targets t
),
picked AS (
    SELECT *
    FROM ranked
    WHERE rn = 1
)
SELECT
    token_id,
    MAX(market_id) AS market_id,
    MAX(event_id) AS event_id,
    MAX(question) AS question,
    MAX(end_date) AS end_date,
    MAX(category) AS category,
    MAX(market_group) AS market_group,
    MAX(event_slug) AS event_slug,
    MAX(event_title) AS event_title,
    MAX(resolution_date) AS resolution_date,

    MAX(CASE WHEN label = 'p30' THEN price END) AS price_30d,
    MAX(CASE WHEN label = 'p30' THEN days_to_deadline END) AS days_30d,

    MAX(CASE WHEN label = 'p7' THEN price END) AS price_7d,
    MAX(CASE WHEN label = 'p7' THEN days_to_deadline END) AS days_7d,

    MAX(CASE WHEN label = 'p1' THEN price END) AS price_1d,
    MAX(CASE WHEN label = 'p1' THEN days_to_deadline END) AS days_1d

FROM picked
GROUP BY token_id;


-- MARCH 25 JS:
DROP VIEW IF EXISTS timing_prices_long_bft;

CREATE VIEW timing_prices_long_bft AS
SELECT
  b.market_id,
  b.event_id,
  b.question,
  b.end_date,
  b.market_group,
  b.category,
  b.event_slug,
  b.event_title,
  b.resolution_date,
  b.token_id,
  ph.ts,
  ph.date AS price_date,
  ph.price,
  (julianday(b.end_date) - julianday(ph.date)) AS days_to_deadline
FROM bets_for_timing_view b
JOIN price_history ph

----------------------------------------
SELECT COUNT(*) AS n_rows FROM timing_prices_long_bft;
SELECT * FROM timing_prices_long_bft LIMIT 10;

----------------------------------------
DROP TABLE IF EXISTS timing_model_input_bft;

CREATE TABLE timing_model_input_bft AS
WITH targets(label, target_days) AS (
  VALUES ('p30', 30.0), ('p7', 7.0), ('p1', 1.0)
),
ranked AS (
  SELECT
    y.token_id,
    y.market_id,
    y.event_id,
    y.event_slug,
    y.category,
    y.question,
    y.end_date,
    y.price_date,
--------------------
SELECT COUNT(*) AS n_markets FROM timing_model_input_bft;

SELECT
  SUM(CASE WHEN price_30d IS NULL THEN 1 ELSE 0 END) AS missing_30d,
  SUM(CASE WHEN price_7d  IS NULL THEN 1 ELSE 0 END) AS missing_7d,
  SUM(CASE WHEN price_1d  IS NULL THEN 1 ELSE 0 END) AS missing_1d,
  COUNT(*) AS total
FROM timing_model_input_bft;

SELECT * FROM timing_model_input_bft LIMIT 10;
    
DROP VIEW IF EXISTS timing_model_analysis_bft;

CREATE VIEW timing_model_analysis_bft AS
SELECT
  *,
  (price_7d - price_30d) AS delta_30_to_7,
  (price_1d - price_7d)  AS delta_7_to_1,
  (price_1d - price_30d) AS delta_30_to_1,

  CASE
    WHEN price_30d > 0 AND price_30d < 1 AND days_30d > 0
    THEN -LOG(1.0 - price_30d) / days_30d
  END AS lambda_30d,

  CASE
    WHEN price_7d > 0 AND price_7d < 1 AND days_7d > 0
    THEN -LOG(1.0 - price_7d) / days_7d
  END AS lambda_7d,

------------------------
-- Mean movement toward deadline
SELECT
  AVG(price_30d) AS avg_p30,
  AVG(price_7d)  AS avg_p7,
  AVG(price_1d)  AS avg_p1,
  AVG(delta_30_to_1) AS avg_delta_30_to_1,
  AVG(delta_30_to_7) AS avg_delta_30_to_7,
  AVG(delta_7_to_1)  AS avg_delta_7_to_1
FROM timing_model_analysis_bft;

-- How often do markets move up vs down?
SELECT
  SUM(CASE WHEN delta_30_to_1 > 0 THEN 1 ELSE 0 END) AS n_up,
  SUM(CASE WHEN delta_30_to_1 < 0 THEN 1 ELSE 0 END) AS n_down,
  SUM(CASE WHEN delta_30_to_1 = 0 THEN 1 ELSE 0 END) AS n_flat,
  COUNT(*) AS total
FROM timing_model_analysis_bft
WHERE delta_30_to_1 IS NOT NULL;

-- Mean implied rates (term structure)
SELECT
  AVG(lambda_30d) AS avg_lambda_30d,
  AVG(lambda_7d)  AS avg_lambda_7d,
  AVG(lambda_1d)  AS avg_lambda_1d
FROM timing_model_analysis_bft;
------------------

SELECT * FROM timing_model_analysis_bft;

SELECT name, type
FROM sqlite_master
WHERE type IN ('table','view')
  AND (name LIKE '%snap%' OR name LIKE '%timing%' OR name LIKE '%model%')
ORDER BY type, name;
-----
SELECT
  ts,
  datetime(ts, 'unixepoch') AS ts_as_utc
FROM price_history
LIMIT 5;

---
SELECT name, type
FROM pragma_table_info('markets')
WHERE lower(name) LIKE '%liquid%'
   OR lower(name) LIKE '%volume%';
---
DROP TABLE IF EXISTS timing_model_input_bft_cov;

CREATE TABLE timing_model_input_bft_cov AS
WITH
targets(label, target_days) AS (
  VALUES
    ('p30', 30.0),
    ('p7',   7.0),
    ('p1',   1.0)
),

ranked AS (
  SELECT
    y.*,
    t.label,
    t.target_days,
    ABS(y.days_to_deadline - t.target_days) AS abs_diff,
    ROW_NUMBER() OVER (
      PARTITION BY y.token_id, t.label
      ORDER BY ABS(y.days_to_deadline - t.target_days), y.ts DESC
    ) AS rn
  FROM timing_prices_long_bft y
  CROSS JOIN targets t
),

picked AS (
  SELECT *
  FROM ranked
  WHERE rn = 1
),

activity AS (
  SELECT
    p.token_id,
    p.label,
    COUNT(ph2.ts) AS n_updates_24h
  FROM picked p
  LEFT JOIN price_history ph2
    ON ph2.token_id = p.token_id
   AND lower(ph2.outcome) = 'yes'
   -- If your ts is milliseconds, replace 86400 with (86400*1000)
   AND ph2.ts BETWEEN (p.ts - 86400) AND p.ts
  GROUP BY p.token_id, p.label
)

SELECT
  MAX(p.market_id)        AS market_id,
  MAX(p.event_id)         AS event_id,
  MAX(p.event_slug)       AS event_slug,
  MAX(p.market_question)  AS market_question,
  MAX(p.end_date)         AS end_date,
  p.token_id              AS token_id,

  -- snapshot timestamps per horizon (needed for auditing and future covariates)
  MAX(CASE WHEN p.label='p30' THEN p.ts END) AS ts_30d,
  MAX(CASE WHEN p.label='p7'  THEN p.ts END) AS ts_7d,
  MAX(CASE WHEN p.label='p1'  THEN p.ts END) AS ts_1d,

  -- prices per horizon
  MAX(CASE WHEN p.label='p30' THEN p.price END) AS price_30d,
  MAX(CASE WHEN p.label='p7'  THEN p.price END) AS price_7d,
  MAX(CASE WHEN p.label='p1'  THEN p.price END) AS price_1d,

  -- time-to-deadline per horizon
  MAX(CASE WHEN p.label='p30' THEN p.days_to_deadline END) AS days_30d,
  MAX(CASE WHEN p.label='p7'  THEN p.days_to_deadline END) AS days_7d,
  MAX(CASE WHEN p.label='p1'  THEN p.days_to_deadline END) AS days_1d,

  -- activity covariate per horizon
  MAX(CASE WHEN p.label='p30' THEN a.n_updates_24h END) AS n_updates_24h_30d,
  MAX(CASE WHEN p.label='p7'  THEN a.n_updates_24h END) AS n_updates_24h_7d,
  MAX(CASE WHEN p.label='p1'  THEN a.n_updates_24h END) AS n_updates_24h_1d

FROM picked p
LEFT JOIN activity a
  ON a.token_id = p.token_id
 AND a.label    = p.label
GROUP BY p.token_id;

-----
DROP TABLE IF EXISTS timing_model_input_bft_cov;

CREATE TABLE timing_model_input_bft_cov AS
WITH
targets(label, target_days) AS (
  VALUES
    ('p30', 30.0),
    ('p7',   7.0),
    ('p1',   1.0)
),

ranked AS (
  SELECT
    y.market_id,
    y.event_id,
    y.event_slug,
    y.token_id,
    y.ts AS snap_ts,
    y.price,
    y.days_to_deadline,
    t.label,
    t.target_days,
    ABS(y.days_to_deadline - t.target_days) AS abs_diff,
    ROW_NUMBER() OVER (
      PARTITION BY y.token_id, t.label
      ORDER BY ABS(y.days_to_deadline - t.target_days), y.ts DESC
    ) AS rn
  FROM timing_prices_long_bft y
  CROSS JOIN targets t
),

picked AS (
  SELECT *
  FROM ranked
  WHERE rn = 1
),

activity AS (
  SELECT
    p.token_id,
    p.label,
    COUNT(ph2.ts) AS n_updates_24h
  FROM picked p
  LEFT JOIN price_history ph2
    ON ph2.token_id = p.token_id
   AND lower(ph2.outcome) = 'yes'
   AND ph2.ts BETWEEN (p.snap_ts - 86400) AND p.snap_ts
  GROUP BY p.token_id, p.label
),

picked_plus AS (
  SELECT
    p.*,
    COALESCE(a.n_updates_24h, 0) AS n_updates_24h,
    m.question      AS market_question,
    m.end_date      AS end_date,
    m.liquidity_num AS liquidity_num,
    m.volume_24hr   AS volume_24hr
  FROM picked p
  LEFT JOIN activity a
    ON a.token_id = p.token_id
   AND a.label    = p.label
  LEFT JOIN markets m
    ON m.market_id = p.market_id
)

SELECT
  market_id,
  MAX(event_id)        AS event_id,
  MAX(event_slug)      AS event_slug,
  MAX(market_question) AS market_question,
  MAX(end_date)        AS end_date,
  token_id,

  MAX(CASE WHEN label='p30' THEN snap_ts END) AS ts_30d,
  MAX(CASE WHEN label='p7'  THEN snap_ts END) AS ts_7d,
  MAX(CASE WHEN label='p1'  THEN snap_ts END) AS ts_1d,

  MAX(CASE WHEN label='p30' THEN price END) AS price_30d,
  MAX(CASE WHEN label='p7'  THEN price END) AS price_7d,
  MAX(CASE WHEN label='p1'  THEN price END) AS price_1d,

  MAX(CASE WHEN label='p30' THEN days_to_deadline END) AS days_30d,
  MAX(CASE WHEN label='p7'  THEN days_to_deadline END) AS days_7d,
  MAX(CASE WHEN label='p1'  THEN days_to_deadline END) AS days_1d,

  MAX(CASE WHEN label='p30' THEN n_updates_24h END) AS n_updates_24h_30d,
  MAX(CASE WHEN label='p7'  THEN n_updates_24h END) AS n_updates_24h_7d,
  MAX(CASE WHEN label='p1'  THEN n_updates_24h END) AS n_updates_24h_1d,

  MAX(liquidity_num) AS liquidity_num,
  MAX(volume_24hr)   AS volume_24hr

FROM picked_plus
GROUP BY token_id;

----
SELECT
  SUM(liquidity_num IS NULL) AS missing_liquidity,
  SUM(volume_24hr   IS NULL) AS missing_volume,
  MIN(liquidity_num) AS min_liquidity,
  MAX(liquidity_num) AS max_liquidity,
  MIN(volume_24hr)   AS min_volume,
  MAX(volume_24hr)   AS max_volume
FROM timing_model_input_bft_cov;


----
DROP TABLE IF EXISTS timing_model_input_bft_cov2;

CREATE TABLE timing_model_input_bft_cov2 AS
WITH
targets(label, target_days) AS (
  VALUES
    ('p30', 30.0),
    ('p7',   7.0),
    ('p1',   1.0)
),

ranked AS (
  SELECT
    y.market_id,
    y.event_id,
    y.event_slug,
    y.question AS market_question,
    y.end_date,
    y.token_id,
    y.ts AS snap_ts,
    y.price,
    y.days_to_deadline,
    t.label,
    t.target_days,
    ABS(y.days_to_deadline - t.target_days) AS abs_diff,
    ROW_NUMBER() OVER (
      PARTITION BY y.token_id, t.label
      ORDER BY ABS(y.days_to_deadline - t.target_days), y.ts DESC
    ) AS rn
  FROM timing_prices_long_bft y
  CROSS JOIN targets t
),

picked AS (
  SELECT *
  FROM ranked
  WHERE rn = 1
),

activity AS (
  SELECT
    p.token_id,
    p.label,
    COUNT(ph2.ts) AS n_obs_24h,
    MIN(ph2.price) AS min_price_24h,
    MAX(ph2.price) AS max_price_24h,
    MAX(ph2.price) - MIN(ph2.price) AS price_range_24h
  FROM picked p
  LEFT JOIN price_history ph2
    ON ph2.token_id = p.token_id
   AND lower(ph2.outcome) = 'yes'
   AND ph2.ts BETWEEN (p.snap_ts - 86400) AND p.snap_ts
  GROUP BY p.token_id, p.label
),

picked_plus AS (
  SELECT
    p.*,
    COALESCE(a.n_obs_24h, 0) AS n_obs_24h,
    COALESCE(a.min_price_24h, p.price) AS min_price_24h,
    COALESCE(a.max_price_24h, p.price) AS max_price_24h,
    COALESCE(a.price_range_24h, 0.0) AS price_range_24h,
    m.liquidity_num,
    m.volume_24hr
  FROM picked p
  LEFT JOIN activity a
    ON a.token_id = p.token_id
   AND a.label    = p.label
  LEFT JOIN markets m
    ON m.market_id = p.market_id
)

SELECT
  market_id,
  MAX(event_id)        AS event_id,
  MAX(event_slug)      AS event_slug,
  MAX(market_question) AS market_question,
  MAX(end_date)        AS end_date,
  token_id,

  MAX(CASE WHEN label='p30' THEN snap_ts END) AS ts_30d,
  MAX(CASE WHEN label='p7'  THEN snap_ts END) AS ts_7d,
  MAX(CASE WHEN label='p1'  THEN snap_ts END) AS ts_1d,

  MAX(CASE WHEN label='p30' THEN price END) AS price_30d,
  MAX(CASE WHEN label='p7'  THEN price END) AS price_7d,
  MAX(CASE WHEN label='p1'  THEN price END) AS price_1d,

  MAX(CASE WHEN label='p30' THEN days_to_deadline END) AS days_30d,
  MAX(CASE WHEN label='p7'  THEN days_to_deadline END) AS days_7d,
  MAX(CASE WHEN label='p1'  THEN days_to_deadline END) AS days_1d,

  MAX(CASE WHEN label='p30' THEN n_obs_24h END)       AS n_obs_24h_30d,
  MAX(CASE WHEN label='p7'  THEN n_obs_24h END)       AS n_obs_24h_7d,
  MAX(CASE WHEN label='p1'  THEN n_obs_24h END)       AS n_obs_24h_1d,

  MAX(CASE WHEN label='p30' THEN price_range_24h END) AS price_range_24h_30d,
  MAX(CASE WHEN label='p7'  THEN price_range_24h END) AS price_range_24h_7d,
  MAX(CASE WHEN label='p1'  THEN price_range_24h END) AS price_range_24h_1d,

  MAX(liquidity_num) AS liquidity_num,
  MAX(volume_24hr)   AS volume_24hr

FROM picked_plus
GROUP BY token_id;

-----
SELECT COUNT(*) AS n_rows
FROM timing_model_input_bft_cov2;

SELECT
  MIN(price_range_24h_30d) AS min_30d,
  MAX(price_range_24h_30d) AS max_30d,
  AVG(price_range_24h_30d) AS mean_30d,
  MIN(price_range_24h_7d)  AS min_7d,
  MAX(price_range_24h_7d)  AS max_7d,
  AVG(price_range_24h_7d)  AS mean_7d,
  MIN(price_range_24h_1d)  AS min_1d,
  MAX(price_range_24h_1d)  AS max_1d,
  AVG(price_range_24h_1d)  AS mean_1d
FROM timing_model_input_bft_cov2;
