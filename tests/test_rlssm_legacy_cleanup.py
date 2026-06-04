from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_rldm_artifacts_are_removed():
    legacy_paths = [
        ROOT / "src" / "hssm" / "likelihoods" / "rldm.py",
        ROOT / "src" / "hssm" / "likelihoods" / "rldm_optimized.py",
        ROOT / "tests" / "test_rldm_likelihood.py",
        ROOT / "docs" / "tutorials" / "rlssm_tutorial.ipynb",
        ROOT / "docs" / "tutorials" / "rlssm_rlwm_model.ipynb",
        ROOT / "docs" / "tutorials" / "add_custom_rlssm_model.ipynb",
    ]

    assert [path.relative_to(ROOT) for path in legacy_paths if path.exists()] == []


def test_mkdocs_no_longer_references_legacy_rlssm_tutorials():
    mkdocs = (ROOT / "mkdocs.yml").read_text()

    assert "tutorials/rlssm_tutorial.ipynb" not in mkdocs
    assert "tutorials/rlssm_rlwm_model.ipynb" not in mkdocs
    assert "tutorials/add_custom_rlssm_model.ipynb" not in mkdocs
    assert "tutorials/rlssm_quickstart.ipynb" in mkdocs
