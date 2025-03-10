"""Document ingestion module."""

import hashlib
import json
from pathlib import Path
from pprint import pp

import meilisearch
import pymupdf

from award.utils import find_files_with_extensions, get_document_code_description


def hash_string(input_string: str) -> str:
    """Generate a SHA-256 hash for the given input string.

    Args:
        input_string (str): The string to hash.

    Returns:
        str: The resulting SHA-256 hash as a hexadecimal string.

    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(input_string.encode("utf-8"))
    return sha256_hash.hexdigest()


def get_document_meta(file_path: Path | str) -> dict:
    """Extract metadata from a document.

    Args:
        file_path (Path | str): The path to the document file.

    Returns:
        dict: A dictionary containing the document's code number,
        description, and text, one field per page.

    """
    file_path = Path(file_path)
    code, description = get_document_code_description(file_path)

    try:
        document = pymupdf.open(file_path)
    except pymupdf.FileDataError:
        return dict(id=code, description=description, pages={})
    pages = {i: t.get_text() for (i, t) in enumerate(document, 1)}
    return dict(id=code, description=description, pages=pages)


def dir2index(directory: Path, output: Path = None) -> dict:
    """Convert a directory of documents to an index compatible with Meilisearch."""
    directory = Path(directory)
    files = find_files_with_extensions(directory)
    d = [get_document_meta(file) for file in files]

    if output:
        with open(output, "w") as file:
            json.dump(d, file, indent=4)

    return d


if __name__ == "__main__":
    pdf = Path(
        "assets/testfiles/009126_00038255_3_Brown University Requisition No. 04199154 Contract Draft_RN_03.29.23.pdf"
    )
    directory = Path("assets/testfiles")

    index = dir2index(directory, output="index.json")

    client = meilisearch.Client("http://localhost:7700", "aSampleMasterKey")

    json_file = open("index.json", encoding="utf-8")
    docs = json.load(json_file)
    # client.index('pdfs1').add_documents(docs)
    client.index("pdfs1").add_documents(index)

    pp(client.index("pdfs1").search("funambulo"))
    pp(client.index("pdfs1").search("foobarbaz"))
