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