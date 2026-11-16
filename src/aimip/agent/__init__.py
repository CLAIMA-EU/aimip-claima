"""Optional AIMIP chat entrypoint; importing the CLI needs no agent dependencies."""

import os


def disable_tracing() -> None:
    """Remove inherited tracing configuration before importing LangChain."""
    for key in list(os.environ):
        if key.startswith(("LANGCHAIN_", "LANGSMITH_")):
            os.environ.pop(key, None)


disable_tracing()


def main(argv=None) -> int:
    try:
        from .chat import main as chat_main
        return chat_main(argv)
    except ImportError:
        import sys
        print("aimip chat requires the optional agent dependencies; install container/requirements-agent.txt", file=sys.stderr)
        return 2
