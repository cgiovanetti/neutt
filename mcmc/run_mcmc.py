"""Cobaya launcher with the pandas/cobaya compatibility shim. Usage: python run_mcmc.py <yaml>."""
import sys

import pandas as pd
pd.options.future.infer_string = False  # cobaya version quirk

from cobaya.run import run  # noqa: E402

if __name__ == "__main__":
    run(sys.argv[1])
