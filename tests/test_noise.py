"""Tests for generation.noise."""

from __future__ import annotations

import random
import re

from generation.noise import add_disfluency, asr_text, render_asr


def test_asr_text_lowercases_and_strips_punctuation():
    assert asr_text("Okay, decision: we hold the release!") == "okay decision we hold the release"
    assert asr_text("We're at 12% and $40,000; costs are $1.2 million.") == (
        "we're at 12 percent and 40000 costs are 1 point 2 million"
    )
    assert asr_text("'Quoted' trial-to-paid, isn't it?") == "quoted trial to paid isn't it"


def test_asr_has_no_uppercase_or_punctuation(cfg):
    rng = random.Random(0)
    for text in ["Hello, World!", "Ship it by Friday (maybe).", "Done… finally?"]:
        out = render_asr(text, rng, cfg.noise)
        assert out == out.lower()
        assert not re.search(r"[^\w\s']", out)


def test_disfluency_adds_exactly_one_token(cfg):
    rng = random.Random(3)
    for _ in range(50):
        out = add_disfluency("we ship on friday", rng, cfg.noise)
        assert len(out.split()) == 5


def test_render_is_deterministic_and_rate_is_close(cfg):
    texts = [f"Sentence number {i} is here." for i in range(2000)]
    a = [render_asr(t, random.Random(f"x{i}"), cfg.noise) for i, t in enumerate(texts)]
    b = [render_asr(t, random.Random(f"x{i}"), cfg.noise) for i, t in enumerate(texts)]
    assert a == b
    changed = sum(len(x.split()) != len(asr_text(t).split()) for x, t in zip(a, texts, strict=True))
    assert abs(changed / len(texts) - cfg.noise.p_disfl) < 0.03


def test_no_disfluency_when_disabled(cfg):
    rng = random.Random(0)
    for _ in range(100):
        assert render_asr("Um so we ship.", rng, cfg.noise, disfluent=False) == "um so we ship"
