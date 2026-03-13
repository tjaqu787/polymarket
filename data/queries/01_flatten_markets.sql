PRAGMA foreign_keys = ON;

BEGIN TRANSACTION;

-- Create comprehensive markets table with all available fields
CREATE TABLE IF NOT EXISTS markets (
  market_id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL,

  -- Basic info
  question TEXT,
  market_slug TEXT,
  description TEXT,
  category TEXT,
  market_type TEXT,

  -- Dates
  created_at TEXT,
  updated_at TEXT,
  start_date TEXT,
  end_date TEXT,
  closed_time TEXT,
  end_date_iso TEXT,
  start_date_iso TEXT,

  -- Status flags
  active INTEGER,
  closed INTEGER,
  archived INTEGER,
  restricted INTEGER,
  wide_format INTEGER,
  new INTEGER,
  sent_discord INTEGER,
  featured INTEGER,
  approved INTEGER,
  ready INTEGER,
  funded INTEGER,
  cyom INTEGER,
  fpmm_live INTEGER,
  clear_book_on_start INTEGER,
  manual_activation INTEGER,
  neg_risk_other INTEGER,
  pending_deployment INTEGER,
  deploying INTEGER,
  has_reviewed_dates INTEGER,
  ready_for_cron INTEGER,

  -- Financial data
  volume_num REAL,
  liquidity_num REAL,
  volume REAL,
  liquidity REAL,
  best_bid REAL,
  best_ask REAL,
  spread REAL,
  last_trade_price REAL,

  -- Volume metrics
  volume_24hr REAL,
  volume_1wk REAL,
  volume_1mo REAL,
  volume_1yr REAL,
  volume_1wk_amm REAL,
  volume_1mo_amm REAL,
  volume_1yr_amm REAL,
  volume_1wk_clob REAL,
  volume_1mo_clob REAL,
  volume_1yr_clob REAL,

  -- Price changes
  one_day_price_change REAL,
  one_hour_price_change REAL,
  one_week_price_change REAL,
  one_month_price_change REAL,
  one_year_price_change REAL,

  -- Outcomes
  outcomes_json TEXT,
  outcome_prices_json TEXT,

  -- Resolution
  uma_resolution_status TEXT,
  uma_resolution_statuses TEXT,
  resolution_source TEXT,
  resolved_by TEXT,

  -- Technical IDs
  condition_id TEXT,
  market_maker_address TEXT,
  clob_token_ids TEXT,

  -- Fees and rewards
  fee TEXT,
  rewards_min_size REAL,
  rewards_max_spread REAL,
  competitive REAL,

  -- Feature flags
  pager_duty_notification_enabled INTEGER,
  rfq_enabled INTEGER,
  holding_rewards_enabled INTEGER,
  fees_enabled INTEGER,
  requires_translation INTEGER,

  -- Media
  image TEXT,
  icon TEXT,
  twitter_card_location TEXT,
  twitter_card_last_refreshed TEXT,

  -- Misc
  submitted_by TEXT,
  creator TEXT,
  updated_by INTEGER,
  fee_type TEXT,

  FOREIGN KEY(event_id) REFERENCES events(id)
);

-- Insert or replace all markets with all fields
INSERT OR REPLACE INTO markets (
  market_id, event_id,
  question, market_slug, description, category, market_type,
  created_at, updated_at, start_date, end_date, closed_time, end_date_iso, start_date_iso,
  active, closed, archived, restricted, wide_format, new, sent_discord, featured,
  approved, ready, funded, cyom, fpmm_live, clear_book_on_start, manual_activation,
  neg_risk_other, pending_deployment, deploying, has_reviewed_dates, ready_for_cron,
  volume_num, liquidity_num, volume, liquidity, best_bid, best_ask, spread, last_trade_price,
  volume_24hr, volume_1wk, volume_1mo, volume_1yr,
  volume_1wk_amm, volume_1mo_amm, volume_1yr_amm,
  volume_1wk_clob, volume_1mo_clob, volume_1yr_clob,
  one_day_price_change, one_hour_price_change, one_week_price_change,
  one_month_price_change, one_year_price_change,
  outcomes_json, outcome_prices_json,
  uma_resolution_status, uma_resolution_statuses, resolution_source, resolved_by,
  condition_id, market_maker_address, clob_token_ids,
  fee, rewards_min_size, rewards_max_spread, competitive,
  pager_duty_notification_enabled, rfq_enabled, holding_rewards_enabled,
  fees_enabled, requires_translation,
  image, icon, twitter_card_location, twitter_card_last_refreshed,
  submitted_by, creator, updated_by, fee_type
)
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

-- Create indexes for commonly queried fields
CREATE INDEX IF NOT EXISTS idx_markets_event_id ON markets(event_id);
CREATE INDEX IF NOT EXISTS idx_markets_end_date ON markets(end_date);
CREATE INDEX IF NOT EXISTS idx_markets_category ON markets(category);
CREATE INDEX IF NOT EXISTS idx_markets_active ON markets(active);
CREATE INDEX IF NOT EXISTS idx_markets_closed ON markets(closed);
CREATE INDEX IF NOT EXISTS idx_markets_market_type ON markets(market_type);

-- Sanity check
SELECT COUNT(*) AS n_markets FROM markets;

-- Commit the transaction
COMMIT;
