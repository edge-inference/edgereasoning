# Third-Party Dependencies

This directory contains external repositories used by the project.

## token2metrics

The `token2metrics` repository is fetched automatically during `make setup`.

Manual fetch/update (if needed):

```bash
# From the project root
bash scripts/fetch_token2metrics.sh
```

Optional flags for the fetch script:

```bash
# Skip editable install into current env
INSTALL_EDITABLE=0 bash scripts/fetch_token2metrics.sh

# Skip running token2metrics/setup.py (requirements install)
RUN_T2M_SETUP=0 bash scripts/fetch_token2metrics.sh

# Pin to a specific branch/tag/commit
PIN=v1.0.0 bash scripts/fetch_token2metrics.sh
```

The script clones or updates `third_party/token2metrics` from `https://github.com/bkubwimana/token2metrics.git` and can install it editable into the current Python environment.


