"""Allow `python -m agentdispatch` from the repository root.

This path needs no install and no `.pth` hook, which matters on machines where
something keeps re-applying macOS's UF_HIDDEN flag to site-packages files —
Python 3.13 skips hidden `.pth` files, which silently breaks editable installs.
"""

from .cli import main

raise SystemExit(main())
