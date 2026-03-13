CREATE VIEW bets_for_timing_view AS
SELECT
  filtered_markets.*,
  mt.token_id,
  mt.outcome,
  mt.token_index,
  e.slug AS event_slug,
  e.title AS event_title,
  filtered_markets.event_id AS market_group,
  SUBSTR(filtered_markets.end_date, 1, 10) AS resolution_date
FROM (
  SELECT *
  FROM markets
  WHERE (lower(question) LIKE '% by %'
      OR lower(question) LIKE '% before %'
      OR lower(question) LIKE '% no later than %'
      OR lower(question) LIKE '% until %')
    AND lower(question) NOT LIKE '% by more than %'
    AND lower(question) NOT LIKE '%nba%'
    AND lower(question) NOT LIKE '%nfl%'
    AND lower(question) NOT LIKE '%mlb%'
    AND lower(question) NOT LIKE '%all-time high%'
    AND lower(question) NOT LIKE '%points%'
    AND lower(question) NOT LIKE '% by at least %'
    AND lower(question) NOT LIKE '%eth%'
    AND lower(question) NOT LIKE '%$%'
    AND lower(question) NOT LIKE '%covid%'
    AND lower(question) NOT LIKE '%tweet %'
    AND lower(question) NOT LIKE '%market cap%'
    AND lower(question) NOT LIKE '%mcap%'
    AND lower(question) NOT LIKE '%usd%'
    AND lower(question) NOT LIKE '%candidate win%'
    AND lower(question) NOT LIKE '% win %'
    AND lower(question) NOT LIKE '%rcp%'
    AND lower(question) NOT LIKE '% case %'
    AND lower(question) NOT LIKE '% cases %'
) filtered_markets
INNER JOIN market_tokens mt ON filtered_markets.market_id = mt.market_id
INNER JOIN events e ON filtered_markets.event_id = e.id
;