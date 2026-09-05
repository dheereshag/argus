"""
Root CLI entrypoint script forwarding execution to app.main.

Usage:
    uv run python main.py <path_to_image>
"""

from app.main import main

if __name__ == "__main__":
    main()
