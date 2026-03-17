DROP VIEW IF EXISTS timing_markets_base;

CREATE VIEW timing_markets_base AS
SELECT
    m.market_id,
    m.event_id,
    m.market_slug,
    m.question,
    m.end_date,
    COALESCE(NULLIF(m.market_slug,''), m.question) AS text_for_tokens
FROM markets m
WHERE
    (
        lower(m.question) LIKE '% by %'
        OR lower(m.question) LIKE '% before %'
        OR lower(m.question) LIKE '% no later than %'
        OR lower(m.question) LIKE '% until %'
    )
    AND lower(m.question) NOT LIKE '% by more than %'
    AND lower(m.question) NOT LIKE '% by at least %'
;