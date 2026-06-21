"""Thin wrapper so the pipeline can be run as a script:

    .venv/Scripts/python scripts/run.py --n 5000

Equivalent to `python -m amphion.run`.
"""

from amphion.run import main

if __name__ == "__main__":
    main()
