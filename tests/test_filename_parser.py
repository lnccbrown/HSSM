from pathlib import Path

import pytest

from award.utils import get_document_code_description

@pytest.mark.parametrize(
    "path, expected",
    [
        ("1234_4568_name.pdf", ("1234_4568", "name")),
        ("1234_5678.doc", ("1234_5678", None)),
        ("My file description.docx", (None, "My file description")),
    ],
)
def test_get_document_code_description(path, expected):
    assert get_document_code_description(path) == expected
