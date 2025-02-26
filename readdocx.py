# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

import argparse
# from pprint import pp
from pathlib import Path
import json

import textract
# from nltk.stem import WordNetLemmatizer

from tokenize_ import get_tokens
from search import get_keywords


def read_doc(file_path):
    text = textract.process(file_path).decode("utf-8")
    return text

def filter_tokens_in_keywords(tokens, keywords):
    return [t for t in tokens if t[0].lower() in keywords]

def main():
    description = "Check inclusion of keywords in MS-word files."
    "Output is a json file where the keys are the file names and the values are keywords found in the file along with the location of the found keywords."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("dir", type=str, help="Path to the DOCX files directory.")
    parser.add_argument(
        "--keywords",
        type=Path,
        help="Path to the file containing keywords to search for.",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output json file.",
        required=True,
    )
    args = parser.parse_args()

    keywords = get_keywords(args.keywords)

    results = {
        file_path.name: filter_tokens_in_keywords(get_tokens(read_doc(file_path)), keywords)
        for file_path in Path(args.dir).glob("*.docx")
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
