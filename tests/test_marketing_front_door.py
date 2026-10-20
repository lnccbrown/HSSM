"""Keep public marketing routes accurate, reachable and licence-safe."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_front_door_links_every_required_public_route() -> None:
    """The documentation homepage exposes the complete public journey."""
    home = (ROOT / "docs/index.md").read_text(encoding="utf-8")
    for target in (
        "getting_started/getting_started.ipynb",
        "#citation",
        "https://github.com/lnccbrown/HSSM/discussions",
        "use-and-licensing.md",
        "community/content-suggestions.md",
    ):
        assert target in home


def test_campaign_and_project_pages_are_in_navigation() -> None:
    """Every new campaign and project page is reachable through MkDocs."""
    serialized = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for target in (
        "campaigns/index.md",
        "campaigns/releases.md",
        "campaigns/methods.md",
        "campaigns/decision-support.md",
        "use-and-licensing.md",
        "community/content-suggestions.md",
        "reference/marketing-experiments.md",
    ):
        assert target in serialized
        assert (ROOT / "docs" / target).is_file()


def test_licensing_copy_does_not_mislabel_hssm_or_imply_permission() -> None:
    """Public copy preserves the licence and evaluation boundaries."""
    text = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "docs/index.md",
            "docs/use-and-licensing.md",
            "docs/campaigns/decision-support.md",
        )
    )
    text = " ".join(text.split())
    assert "publicly available scientific software" in text
    assert "not represented as OSI open source" in text
    assert "not legal advice" in text
    assert "does not promise a commercial licence" in text


def test_content_suggestion_form_is_public_safe_intake_not_approval() -> None:
    """The issue form rejects sensitive input and grants no authority."""
    form = yaml.safe_load(
        (ROOT / ".github/ISSUE_TEMPLATE/content-suggestion.yml").read_text(
            encoding="utf-8"
        )
    )
    body = str(form["body"])
    assert "patient-level data" in body
    assert "private conversations" in body
    assert "does not grant publication" in body
