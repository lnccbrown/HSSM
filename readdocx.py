# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

import argparse
from pprint import pp
from tokenize_ import get_tokens
import textract


def read_doc(file_path):
    text = textract.process(file_path).decode("utf-8")
    return text


def main():
    parser = argparse.ArgumentParser(description="Extract text from a DOCX file.")
    parser.add_argument("file_path", type=str, help="Path to the DOCX file.")
    args = parser.parse_args()

    txt = read_doc(args.file_path)
    tokens = list(get_tokens(txt))
    pp(tokens[:10])
    return tokens


if __name__ == "__main__":
    main()
