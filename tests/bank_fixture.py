"""A small, valid seed bank built in code, for composition tests (not dataset seeds)."""

from __future__ import annotations

from generation.config import GenerationConfig
from generation.schema import (
    DOMAIN_ABBR,
    ZONE_PREFIX,
    Domain,
    Filler,
    HNCategory,
    SeedSegment,
    Zone,
)

LEXICONS = {
    "PERSON": [f"Name{i}" for i in range(40)],
    "PROJECT": ["Bluefin", "Kestrel", "Marigold"],
    "DATE_REL": ["next Tuesday", "end of sprint"],
    "NUMBER": ["twelve", "40"],
}
FILLERS = (
    Filler(filler_id="fill-001", text="Yeah."),
    Filler(filler_id="fill-002", text="Sorry, go ahead."),
    Filler(filler_id="fill-003", text="Can everyone hear me?", tag="opening"),
    Filler(filler_id="fill-004", text="Thanks, everyone.", tag="closing_chatter"),
)
DOMAINS = (Domain.ENGINEERING, Domain.SALES)


def _sentences(zone: Zone, family: int, variant: int, domain: Domain) -> list[dict]:
    tag = domain.value
    lead = f"Family {family} variant {variant} opens the {zone.value} part about {{PROJECT}}."
    if zone == Zone.ACTION_ITEM:
        key = {
            "speaker": "S1",
            "text": f"{{S2|owner}} does item {family}.{variant} ({tag}) by {{DATE_REL|deadline}}.",
            "importance": 1,
            "zone": "action_item",
        }
    else:
        key = {
            "speaker": "S1",
            "text": f"Key point {family}.{variant} ({tag}): we keep {{NUMBER}} things moving.",
            "importance": 1,
            "zone": zone.value,
        }
    return [
        {
            "speaker": "S1",
            "text": lead,
            "importance": 0,
            "zone": "discussion" if zone in (Zone.DECISION, Zone.ACTION_ITEM) else zone.value,
        },
        {
            "speaker": "S2",
            "text": f"Right, I follow the reasoning for number {family}.{variant} ({tag}) here.",
            "importance": 0,
            "zone": "discussion" if zone in (Zone.DECISION, Zone.ACTION_ITEM) else zone.value,
        },
        key,
        {
            "speaker": "S2",
            "text": f"Okay, agreed on {family}.{variant} ({tag}), let's move along.",
            "importance": 0,
            "zone": zone.value,
        },
    ]


def make_segments(families_per_cell: int = 10) -> list[SeedSegment]:
    """Normal families for every zone in two domains, valid for all meeting types."""
    segments = []
    for zone in Zone:
        for domain in DOMAINS:
            for k in range(1, families_per_cell + 1):
                family = f"{ZONE_PREFIX[zone]}-standup-{DOMAIN_ABBR[domain]}-{k:03d}"
                for v in (1, 2):
                    segments.append(
                        SeedSegment.model_validate(
                            {
                                "segment_id": f"{family}-v{v}",
                                "family_id": family,
                                "primary_zone": zone,
                                "meeting_types": [
                                    "standup",
                                    "quick_sync",
                                    "project_review",
                                    "stakeholder_update",
                                    "retrospective",
                                ],
                                "domain": domain,
                                "hard_negative": False,
                                "hn_category": None,
                                "confidence": 1.0,
                                "n_speakers": 2,
                                "sentences": _sentences(zone, k, v, domain),
                            }
                        )
                    )
    return segments


def make_hard_negatives(families: int = 40) -> list[SeedSegment]:
    """Hard-negative families (tentative ideas), all importance 0."""
    out = []
    for k in range(1, families + 1):
        family = f"hn-{HNCategory.TENTATIVE_IDEA.value}-{k:03d}"
        for v in (1, 2):
            out.append(
                SeedSegment.model_validate(
                    {
                        "segment_id": f"{family}-v{v}",
                        "family_id": family,
                        "primary_zone": "discussion",
                        "meeting_types": [
                            "standup",
                            "quick_sync",
                            "project_review",
                            "stakeholder_update",
                            "retrospective",
                        ],
                        "domain": DOMAINS[k % 2],
                        "hard_negative": True,
                        "hn_category": "tentative_idea",
                        "confidence": 0.7,
                        "n_speakers": 2,
                        "sentences": [
                            {
                                "speaker": "S1",
                                "text": f"Maybe someone could look at idea {k}.{v} at some point.",
                                "importance": 0,
                                "zone": "discussion",
                            },
                            {
                                "speaker": "S2",
                                "text": f"Could be, though nobody has time for {k}.{v} right now.",
                                "importance": 0,
                                "zone": "discussion",
                            },
                        ],
                    }
                )
            )
    return out


def small_config(cfg: GenerationConfig) -> GenerationConfig:
    """Real config with word ranges the small fixture bank can reach."""
    raw = cfg.model_dump(mode="json")
    for mt in raw["meeting_types"].values():
        mt["words"] = [80, 260]
        mt["participants"] = [3, 4]
    raw["volumes"]["synthetic_notes_per_type"] = 10
    raw["volumes"]["hard_negative_notes_per_type"] = 4
    raw["composition"]["max_segment_reuse"] = 1000
    return GenerationConfig.model_validate(raw)
