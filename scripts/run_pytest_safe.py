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
    try:
        extra_args = shlex.split(test_args_str)
    except ValueError as e:
        print(f"Error parsing HSSM_TEST_ARGS: {e}", file=sys.stderr)
        sys.exit(1)

    # Build the base command
    cmd = ["uv", "run", "pytest"]

    # Append the dynamic arguments first so the script's fixed arguments
    # (below) always take precedence and cannot be overridden by callers.
    cmd.extend(extra_args)

    # Add any fixed arguments passed to this script
    cmd.extend(sys.argv[1:])

    # Execute without a shell
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
