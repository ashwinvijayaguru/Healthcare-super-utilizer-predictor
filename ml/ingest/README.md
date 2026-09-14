# Real-data adapters

Each adapter maps a public dataset onto the feature contract in `ml/config.py`,
returning a DataFrame with exactly `FEATURE_NAMES` plus `is_super_utilizer`.
Once an adapter returns that shape, `ml/train.py --data <file>` needs no changes.

Column names in MEPS and DE-SynPUF carry a **year suffix** (`ERTOT21`, `TOTEXP22`).
Every adapter takes a `year` argument and builds names from it; verify against the
codebook for the release you download before trusting a run.
