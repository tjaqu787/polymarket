"""
Slug Normalization Utility for Semantic Market Grouping

This module provides functions to normalize market slugs by stripping timing patterns,
extracting actors/countries, and generating semantic group identifiers.

Used to cluster semantically similar markets that ask about the same event at different times.
"""

import re
from typing import Optional, Tuple


def normalize_slug(slug: str) -> str:
    """
    Normalize a market slug by stripping timing patterns.

    Removes suffixes like:
    - -by-march-31, -by-december-31-2026, -by-2027, -by-july, -by-eoy
    - -before-july, -before-2027, -before-july-12-2025
    - -until-march-31
    - -on-or-before-june-30-2022
    - -no-later-than-march-31
    - Numeric variant suffixes like -1, -2, -3

    Args:
        slug: Original market slug

    Returns:
        Canonical slug with timing patterns removed

    Examples:
        >>> normalize_slug("will-a-nuclear-weapon-detonate-by-march-31")
        'will-a-nuclear-weapon-detonate'
        >>> normalize_slug("nuclear-weapon-detonation-by-june-30-2025")
        'nuclear-weapon-detonation'
        >>> normalize_slug("explosion-at-zaporizhzhia-nuclear-plant-by-sep-30")
        'explosion-at-zaporizhzhia-nuclear-plant'
    """
    if not slug:
        return slug

    # Define timing patterns to strip (in order of specificity)
    patterns = [
        # Complex date patterns first (more specific)
        r'-on-or-before-[a-z]+-\d{1,2}-\d{2,4}$',        # -on-or-before-june-30-2022
        r'-no-later-than-[a-z]+-\d{1,2}-\d{2,4}$',       # -no-later-than-march-31-2026
        r'-by-[a-z]+-\d{1,2}-\d{2,4}$',                  # -by-march-31-2026
        r'-before-[a-z]+-\d{1,2}-\d{2,4}$',              # -before-july-12-2025
        r'-until-[a-z]+-\d{1,2}-\d{2,4}$',               # -until-march-31-2025

        # Date patterns without year (month-day)
        r'-on-or-before-[a-z]+-\d{1,2}$',                # -on-or-before-june-30
        r'-no-later-than-[a-z]+-\d{1,2}$',               # -no-later-than-march-31
        r'-by-[a-z]+-\d{1,2}$',                          # -by-march-31
        r'-before-[a-z]+-\d{1,2}$',                      # -before-july-12
        r'-until-[a-z]+-\d{1,2}$',                       # -until-march-31

        # Month or year only
        r'-by-[a-z]+-[a-z]+$',                            # -by-end-of-year
        r'-by-[a-z]+$',                                   # -by-july, -by-eoy, -by-friday
        r'-before-[a-z]+$',                               # -before-july, -before-october
        r'-until-[a-z]+$',                                # -until-december
        r'-by-\d{2,4}$',                                  # -by-2027, -by-27
        r'-before-\d{2,4}$',                              # -before-2027

        # Day of week patterns
        r'-by-(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$',
        r'-before-(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$',

        # Numeric variant suffixes - be VERY aggressive
        # Match any trailing hyphens followed by numbers, including long chains like -123-456-789
        r'(-\d+)+$',                                      # -1, -2, -299, -123-456-789-... (any numeric suffixes)
    ]

    canonical = slug.lower().strip()

    # Apply patterns iteratively until no more changes
    # This handles cases like "by-march-31-2026-123" where we need multiple passes
    max_iterations = 10
    for iteration in range(max_iterations):
        old_canonical = canonical
        for pattern in patterns:
            canonical = re.sub(pattern, '', canonical)

        # If nothing changed, we're done
        if canonical == old_canonical:
            break

    return canonical


def extract_actor(canonical_slug: str, question: str) -> Optional[str]:
    """
    Extract country/actor/entity from slug or question if present.

    This is used to split semantically similar markets that involve different actors.
    For example: "Will Russia test nuclear weapon" vs "Will US test nuclear weapon"

    Args:
        canonical_slug: Normalized slug (from normalize_slug)
        question: Full question text

    Returns:
        Actor identifier (lowercase) or None if no specific actor detected

    Examples:
        >>> extract_actor("will-russia-test-nuclear-weapon", "Will Russia test a nuclear weapon?")
        'russia'
        >>> extract_actor("will-the-us-test-nuclear-weapon", "Will the U.S. test a nuclear weapon?")
        'us'
        >>> extract_actor("nuclear-weapon-detonation", "Will a nuclear weapon detonate?")
        None
    """
    # Define actors/countries/entities to detect
    # Order matters: check more specific patterns first
    actors = [
        ('united-states', 'us'),          # Normalize to 'us'
        ('usa', 'us'),                     # Normalize to 'us'
        ('u-s', 'us'),                     # Normalize to 'us'
        ('the-us', 'us'),                  # Normalize to 'us'
        ('north-korea', 'north-korea'),
        ('south-korea', 'south-korea'),
        ('russia', 'russia'),
        ('china', 'china'),
        ('iran', 'iran'),
        ('israel', 'israel'),
        ('ukraine', 'ukraine'),
        ('trump', 'trump'),
        ('biden', 'biden'),
        ('putin', 'putin'),
        ('gop', 'gop'),
        ('republicans', 'republicans'),
        ('democrats', 'democrats'),
        ('nato', 'nato'),
        ('un', 'un'),
    ]

    # Check both slug and question
    text_to_search = f"{canonical_slug} {question.lower()}"

    for pattern, normalized_actor in actors:
        # Use word boundaries to avoid partial matches
        # e.g., don't match "use" when looking for "us"
        if re.search(rf'\b{re.escape(pattern)}\b', text_to_search):
            return normalized_actor

    return None


def generate_semantic_group_id(canonical_slug: str,
                               actor: Optional[str],
                               event_slug: Optional[str]) -> str:
    """
    Generate a unique semantic group identifier.

    The semantic group ID determines which markets will be clustered together.

    Strategy:
    - Use canonical_slug as base (primary clustering key)
    - Append actor if present to split groups by actor
    - event_slug is used as input for canonical_slug but not directly in grouping

    Args:
        canonical_slug: Normalized market slug
        actor: Extracted actor/country (or None)
        event_slug: Polymarket's event slug (for reference, not used directly)

    Returns:
        Semantic group identifier

    Examples:
        >>> generate_semantic_group_id("nuclear-weapon-detonation", None, "nuclear-weapon-detonation-by")
        'nuclear-weapon-detonation'
        >>> generate_semantic_group_id("will-russia-test-nuclear-weapon", "russia", "russia-nuclear-test-by")
        'will-russia-test-nuclear-weapon:russia'
        >>> generate_semantic_group_id("will-us-test-nuclear-weapon", "us", "us-nuclear-test-by")
        'will-us-test-nuclear-weapon:us'
    """
    # Use canonical_slug as the primary base for grouping
    # This ensures semantically similar markets cluster together
    base = canonical_slug

    # If actor is present, append to create distinct groups
    # This splits "Russia tests nuke" from "US tests nuke"
    if actor:
        return f"{base}:{actor}"

    return base


def normalize_and_group(market_slug: str,
                        question: str,
                        event_slug: Optional[str] = None) -> Tuple[str, Optional[str], str]:
    """
    Convenience function to perform all normalization and grouping steps.

    Args:
        market_slug: Original market slug
        question: Market question text
        event_slug: Polymarket's event slug (optional)

    Returns:
        Tuple of (canonical_slug, actor, semantic_group_id)

    Example:
        >>> normalize_and_group(
        ...     "will-russia-test-a-nuclear-weapon-by-november-30-2025",
        ...     "Will Russia test a nuclear weapon by November 30 2025?",
        ...     "russia-nuclear-test-by"
        ... )
        ('will-russia-test-a-nuclear-weapon', 'russia', 'russia-nuclear-test-by:russia')
    """
    canonical_slug = normalize_slug(market_slug)
    actor = extract_actor(canonical_slug, question)
    semantic_group_id = generate_semantic_group_id(canonical_slug, actor, event_slug)

    return canonical_slug, actor, semantic_group_id


if __name__ == "__main__":
    # Test cases
    print("Testing slug normalization:")
    print()

    test_cases = [
        ("will-a-nuclear-weapon-detonate-by-march-31", "Will a nuclear weapon detonate by March 31?", "will-a-nuclear-weapon-detonate-by"),
        ("nuclear-weapon-detonation-by-june-30-2025", "Nuclear weapon detonation by June 30?", "nuclear-weapon-detonation-by"),
        ("will-russia-test-a-nuclear-weapon-by-november-30-2025", "Will Russia test a nuclear weapon by November 30 2025?", "russia-nuclear-test-by"),
        ("will-the-us-test-a-nuclear-weapon-by-december-31-2025", "Will the U.S. test a nuclear weapon by December 31 2025?", "us-nuclear-test-by"),
        ("explosion-at-zaporizhzhia-nuclear-plant-by-july-12", "Explosion at Zaporizhzhia nuclear plant by July 12?", "explosion-at-zaporizhzhia-nuclear-plant-by-july-12"),
        ("gpt-5-released-by-december-31", "Will GPT-5 be released by December 31?", "when-will-gpt-5-be-released"),
    ]

    for market_slug, question, event_slug in test_cases:
        canonical, actor, group_id = normalize_and_group(market_slug, question, event_slug)
        print(f"Market: {market_slug}")
        print(f"  → Canonical: {canonical}")
        print(f"  → Actor: {actor}")
        print(f"  → Group ID: {group_id}")
        print()
