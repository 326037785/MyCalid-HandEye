"""
VSCode F5 entry point.

This file intentionally keeps a simple main() + selectable test block style,
instead of pytest.
"""

from handeye_sim.test_blocks.test1_eye_in_hand import test1


def main() -> None:
    # Select the test block here.
    # Later you can add test2(), test3(), etc. and switch the call below.
    test1()


if __name__ == "__main__":
    main()
