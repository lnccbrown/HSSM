"""Safe wrapper for running pytest in CI to prevent command injection."""

import os
import shlex
import subprocess
import sys


def main():
    """Run pytest safely using shlex to prevent command injection."""
    # Retrieve dynamic arguments from environment variable to avoid shell interpolation
    test_args_str = os.environ.get("HSSM_TEST_ARGS", "")

    # Split safely using shell-like syntax
    extra_args = shlex.split(test_args_str)

    # Build the base command
    cmd = ["uv", "run", "pytest"]

    # Add any fixed arguments passed to this script
    cmd.extend(sys.argv[1:])

    # Append the dynamic arguments
    cmd.extend(extra_args)

    # Execute without a shell
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
