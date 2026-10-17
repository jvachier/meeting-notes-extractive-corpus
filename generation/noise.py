"""ASR-style rendering of clean sentences (PLAN §4.2 step 6).

``sentence_text_asr`` is lowercased, has no punctuation, and with probability ``p_disfl``
gets one disfluency: a filler token ("um", "uh") or a one-word repetition. Rendering never
touches labels: it maps one clean string to one ASR string.
"""

from __future__ import annotations

import random
import re

from generation.config import NoiseConfig

_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_KEEP_RE = re.compile(r"[^\w\s']")
_EDGE_APOSTROPHE_RE = re.compile(r"(?<!\w)'|'(?!\w)")


def asr_text(text: str) -> str:
    """Lowercase, spell out ``%`` and decimals, drop punctuation and symbols.

    Apostrophes inside words are kept ("don't"), as most ASR systems do.
    """
    out = text.lower().replace("%", " percent")
    out = _THOUSANDS_RE.sub("", out)
    out = _DECIMAL_RE.sub(" point ", out)
    out = out.replace("’", "'")
    out = _KEEP_RE.sub(" ", out)
    out = _EDGE_APOSTROPHE_RE.sub(" ", out)
    return " ".join(out.replace("_", " ").split())


def add_disfluency(asr: str, rng: random.Random, cfg: NoiseConfig) -> str:
    """Insert one disfluency into an ASR string (a filler token or a repeated word)."""
    words = asr.split()
    if not words:
        return asr
    position = rng.randrange(len(words))
    if rng.random() < cfg.p_repetition:
        words.insert(position, words[position])
    else:
        words.insert(position, rng.choice(cfg.disfluency_tokens))
    return " ".join(words)


def render_asr(text: str, rng: random.Random, cfg: NoiseConfig, *, disfluent: bool = True) -> str:
    """Full ASR rendering: ``asr_text`` plus, with probability ``p_disfl``, one disfluency.

    ``disfluent=False`` skips the disfluency (used for AMI, which is already spontaneous
    speech). The random draw is made either way, so the stream stays aligned.
    """
    out = asr_text(text)
    roll = rng.random()
    if disfluent and roll < cfg.p_disfl:
        out = add_disfluency(out, rng, cfg)
    return out
