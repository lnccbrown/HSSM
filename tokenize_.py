from enchant.tokenize import get_tokenizer
from spellcheck import correct_text


def get_tokens(text: str) -> list[str]:
    tknzr = get_tokenizer("en_US")
    return tknzr(text)


def main() -> None:
    text = "Brwn Unversity"
    tokens = get_tokens(text)

    # spell check tokens
    out = [(correct_text(t), i) for t, i in tokens]

    print(out)
    return out


if __name__ == "__main__":
    main()
