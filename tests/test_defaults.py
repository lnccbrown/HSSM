from hssm import show_defaults


def test_show_defaults():
    print(show_defaults("ddm", None))
    print(show_defaults("ddm", "analytical"))
