-- First pass: deadline-ish markets
SELECT COUNT(*) AS n_deadline_like
FROM markets
WHERE lower(question) LIKE '% by %'
   OR lower(question) LIKE '% before %'
   OR lower(question) LIKE '% no later than %'
   OR lower(question) LIKE '% until %';

-- Exclude sports phrasing like "win by more than"
SELECT COUNT(*) AS n_deadline_like_cleaned
FROM markets
WHERE (lower(question) LIKE '% by %'
    OR lower(question) LIKE '% before %'
    OR lower(question) LIKE '% no later than %'
    OR lower(question) LIKE '% until %')
  AND lower(question) NOT LIKE '% by more than %'
  AND lower(question) NOT LIKE '% by at least %';

-- Show examples
SELECT market_id, end_date, substr(question,1,140) AS q
FROM markets
WHERE (lower(question) LIKE '% by %'
    OR lower(question) LIKE '% before %'
    OR lower(question) LIKE '% no later than %'
    OR lower(question) LIKE '% until %')
  AND lower(question) NOT LIKE '% by more than %'
  AND lower(question) NOT LIKE '% by at least %'
LIMIT 30;