# /// script
# requires-python = ">=3.11"
# dependencies = ["enchant"]
# ///

import enchant


def correct_text(text):
    d = enchant.Dict("en_US")
    words = text.split()
    corrected_words = [
        d.suggest(word)[0] if not d.check(word) and d.suggest(word) else word
        for word in words
    ]
    return " ".join(corrected_words)


if __name__ == "__main__":

    def main() -> None:
        txt = "BROWN Office of Sponeored Projects"
        print(correct_text(txt))  # Output: "example"

    main()
