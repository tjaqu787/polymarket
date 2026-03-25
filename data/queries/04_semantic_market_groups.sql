-- Create semantic market groups table
-- This table stores the normalized/canonical version of market slugs
-- and assigns semantic group IDs for clustering similar markets

-- Drop existing table if it exists
DROP TABLE IF EXISTS semantic_market_groups;

-- Create the semantic groups table
CREATE TABLE semantic_market_groups (
    market_id TEXT PRIMARY KEY,
    canonical_slug TEXT NOT NULL,
    actor TEXT,
    semantic_group_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient lookups
CREATE INDEX idx_semantic_group_id ON semantic_market_groups(semantic_group_id);
CREATE INDEX idx_canonical_slug ON semantic_market_groups(canonical_slug);
CREATE INDEX idx_actor ON semantic_market_groups(actor);

-- Note: This table is populated by data/build_semantic_groups.py
