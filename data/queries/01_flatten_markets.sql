PRAGMA foreign_keys = ON;

-- Show count before changes
SELECT 'Markets count BEFORE:' AS status, COUNT(*) AS n_markets FROM markets WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='markets')
UNION ALL
SELECT 'Markets count BEFORE:' AS status, COUNT(*) AS n_markets FROM markets WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE type='view' AND name='markets')
UNION ALL
SELECT 'Markets count BEFORE:' AS status, 0 AS n_markets WHERE NOT EXISTS (SELECT 1 FROM sqlite_master WHERE name='markets');

BEGIN TRANSACTION;

-- Drop the old table if it exists (converting to view)
DROP TABLE IF EXISTS markets;
DROP VIEW IF EXISTS markets;

-- Create markets as a VIEW that derives from events.markets JSON
-- This will automatically update when events are updated
CREATE VIEW IF NOT EXISTS markets AS
SELECT
  json_extract(m.value, '$.id')                                    AS market_id,
  e.id                                                             AS event_id,

  -- Basic info
  json_extract(m.value, '$.question')                             AS question,
  json_extract(m.value, '$.slug')                                 AS market_slug,
  json_extract(m.value, '$.description')                          AS description,
  json_extract(m.value, '$.category')                             AS category,
  json_extract(m.value, '$.marketType')                           AS market_type,

  -- Dates
  json_extract(m.value, '$.createdAt')                            AS created_at,
  json_extract(m.value, '$.updatedAt')                            AS updated_at,
  json_extract(m.value, '$.startDate')                            AS start_date,
  json_extract(m.value, '$.endDate')                              AS end_date,
  json_extract(m.value, '$.closedTime')                           AS closed_time,
  json_extract(m.value, '$.endDateIso')                           AS end_date_iso,
  json_extract(m.value, '$.startDateIso')                         AS start_date_iso,

  -- Status flags
  CAST(json_extract(m.value, '$.active') AS INTEGER)              AS active,
  CAST(json_extract(m.value, '$.closed') AS INTEGER)              AS closed,
  CAST(json_extract(m.value, '$.archived') AS INTEGER)            AS archived,
  CAST(json_extract(m.value, '$.restricted') AS INTEGER)          AS restricted,
  CAST(json_extract(m.value, '$.wideFormat') AS INTEGER)          AS wide_format,
  CAST(json_extract(m.value, '$.new') AS INTEGER)                 AS new,
  CAST(json_extract(m.value, '$.sentDiscord') AS INTEGER)         AS sent_discord,
  CAST(json_extract(m.value, '$.featured') AS INTEGER)            AS featured,
  CAST(json_extract(m.value, '$.approved') AS INTEGER)            AS approved,
  CAST(json_extract(m.value, '$.ready') AS INTEGER)               AS ready,
  CAST(json_extract(m.value, '$.funded') AS INTEGER)              AS funded,
  CAST(json_extract(m.value, '$.cyom') AS INTEGER)                AS cyom,
  CAST(json_extract(m.value, '$.fpmmLive') AS INTEGER)            AS fpmm_live,
  CAST(json_extract(m.value, '$.clearBookOnStart') AS INTEGER)    AS clear_book_on_start,
  CAST(json_extract(m.value, '$.manualActivation') AS INTEGER)    AS manual_activation,
  CAST(json_extract(m.value, '$.negRiskOther') AS INTEGER)        AS neg_risk_other,
  CAST(json_extract(m.value, '$.pendingDeployment') AS INTEGER)   AS pending_deployment,
  CAST(json_extract(m.value, '$.deploying') AS INTEGER)           AS deploying,
  CAST(json_extract(m.value, '$.hasReviewedDates') AS INTEGER)    AS has_reviewed_dates,
  CAST(json_extract(m.value, '$.readyForCron') AS INTEGER)        AS ready_for_cron,

  -- Financial data
  CAST(json_extract(m.value, '$.volumeNum') AS REAL)              AS volume_num,
  CAST(json_extract(m.value, '$.liquidityNum') AS REAL)           AS liquidity_num,
  CAST(json_extract(m.value, '$.volume') AS REAL)                 AS volume,
  CAST(json_extract(m.value, '$.liquidity') AS REAL)              AS liquidity,
  CAST(json_extract(m.value, '$.bestBid') AS REAL)                AS best_bid,
  CAST(json_extract(m.value, '$.bestAsk') AS REAL)                AS best_ask,
  CAST(json_extract(m.value, '$.spread') AS REAL)                 AS spread,
  CAST(json_extract(m.value, '$.lastTradePrice') AS REAL)         AS last_trade_price,

  -- Volume metrics
  CAST(json_extract(m.value, '$.volume24hr') AS REAL)             AS volume_24hr,
  CAST(json_extract(m.value, '$.volume1wk') AS REAL)              AS volume_1wk,
  CAST(json_extract(m.value, '$.volume1mo') AS REAL)              AS volume_1mo,
  CAST(json_extract(m.value, '$.volume1yr') AS REAL)              AS volume_1yr,
  CAST(json_extract(m.value, '$.volume1wkAmm') AS REAL)           AS volume_1wk_amm,
  CAST(json_extract(m.value, '$.volume1moAmm') AS REAL)           AS volume_1mo_amm,
  CAST(json_extract(m.value, '$.volume1yrAmm') AS REAL)           AS volume_1yr_amm,
  CAST(json_extract(m.value, '$.volume1wkClob') AS REAL)          AS volume_1wk_clob,
  CAST(json_extract(m.value, '$.volume1moClob') AS REAL)          AS volume_1mo_clob,
  CAST(json_extract(m.value, '$.volume1yrClob') AS REAL)          AS volume_1yr_clob,

  -- Price changes
  CAST(json_extract(m.value, '$.oneDayPriceChange') AS REAL)      AS one_day_price_change,
  CAST(json_extract(m.value, '$.oneHourPriceChange') AS REAL)     AS one_hour_price_change,
  CAST(json_extract(m.value, '$.oneWeekPriceChange') AS REAL)     AS one_week_price_change,
  CAST(json_extract(m.value, '$.oneMonthPriceChange') AS REAL)    AS one_month_price_change,
  CAST(json_extract(m.value, '$.oneYearPriceChange') AS REAL)     AS one_year_price_change,

  -- Outcomes
  json_extract(m.value, '$.outcomes')                             AS outcomes_json,
  json_extract(m.value, '$.outcomePrices')                        AS outcome_prices_json,

  -- Resolution
  json_extract(m.value, '$.umaResolutionStatus')                  AS uma_resolution_status,
  json_extract(m.value, '$.umaResolutionStatuses')                AS uma_resolution_statuses,
  json_extract(m.value, '$.resolutionSource')                     AS resolution_source,
  json_extract(m.value, '$.resolvedBy')                           AS resolved_by,

  -- Technical IDs
  json_extract(m.value, '$.conditionId')                          AS condition_id,
  json_extract(m.value, '$.marketMakerAddress')                   AS market_maker_address,
  json_extract(m.value, '$.clobTokenIds')                         AS clob_token_ids,

  -- Fees and rewards
  json_extract(m.value, '$.fee')                                  AS fee,
  CAST(json_extract(m.value, '$.rewardsMinSize') AS REAL)         AS rewards_min_size,
  CAST(json_extract(m.value, '$.rewardsMaxSpread') AS REAL)       AS rewards_max_spread,
  CAST(json_extract(m.value, '$.competitive') AS REAL)            AS competitive,

  -- Feature flags
  CAST(json_extract(m.value, '$.pagerDutyNotificationEnabled') AS INTEGER) AS pager_duty_notification_enabled,
  CAST(json_extract(m.value, '$.rfqEnabled') AS INTEGER)          AS rfq_enabled,
  CAST(json_extract(m.value, '$.holdingRewardsEnabled') AS INTEGER) AS holding_rewards_enabled,
  CAST(json_extract(m.value, '$.feesEnabled') AS INTEGER)         AS fees_enabled,
  CAST(json_extract(m.value, '$.requiresTranslation') AS INTEGER) AS requires_translation,

  -- Media
  json_extract(m.value, '$.image')                                AS image,
  json_extract(m.value, '$.icon')                                 AS icon,
  json_extract(m.value, '$.twitterCardLocation')                  AS twitter_card_location,
  json_extract(m.value, '$.twitterCardLastRefreshed')             AS twitter_card_last_refreshed,

  -- Misc
  json_extract(m.value, '$.submitted_by')                         AS submitted_by,
  json_extract(m.value, '$.creator')                              AS creator,
  CAST(json_extract(m.value, '$.updatedBy') AS INTEGER)           AS updated_by,
  json_extract(m.value, '$.feeType')                              AS fee_type

FROM events e
JOIN json_each(e.markets) m
WHERE e.markets IS NOT NULL
  AND e.markets != ''
  AND json_valid(e.markets)
  AND json_extract(m.value, '$.id') IS NOT NULL;

-- Create indexes on the events table to improve view performance
CREATE INDEX IF NOT EXISTS idx_events_id ON events(id);
CREATE INDEX IF NOT EXISTS idx_events_markets ON events(markets) WHERE markets IS NOT NULL;

-- Commit the transaction
COMMIT;

-- Show count after changes
SELECT 'Markets count AFTER:' AS status, COUNT(*) AS n_markets FROM markets;
