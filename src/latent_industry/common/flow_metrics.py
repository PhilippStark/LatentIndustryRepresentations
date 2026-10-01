import numpy as np
import pandas as pd


def symmetric_flow_phi(
    flows: pd.DataFrame, src_col: str, tgt_col: str, val_col: str
) -> pd.DataFrame:
    """Symmetric relatedness from directed flows between codes (Neffke-Henning style).

      phi(i,j) = (f(i->j) + f(j->i)) / (out(i) + in(j) + out(j) + in(i))

    where f(i->j) is the flow value from i to j, out(i) is the total outflow
    from i, and in(j) is the total inflow to j. Used for worker-transition
    counts (labor relatedness) and IO flow values (IO relatedness) alike.

    Parameters
    ----------
    flows : pd.DataFrame
        Columns ``[src_col, tgt_col, val_col]``, one row per directed
        (src, tgt) pair with its aggregated flow value.

    Returns
    -------
    pd.DataFrame
        Columns ``[src_col, tgt_col, "phi"]``.
    """
    f = flows[[src_col, tgt_col, val_col]].copy()

    out_agg = flows.groupby(src_col)[val_col].sum()  # indexed by code
    in_agg = flows.groupby(tgt_col)[val_col].sum()  # indexed by code

    f = f.merge(out_agg.rename("out_i"), left_on=src_col, right_index=True, how="left")
    f = f.merge(in_agg.rename("in_j"), left_on=tgt_col, right_index=True, how="left")
    f = f.merge(out_agg.rename("out_j"), left_on=tgt_col, right_index=True, how="left")
    f = f.merge(in_agg.rename("in_i"), left_on=src_col, right_index=True, how="left")

    rev = flows[[src_col, tgt_col, val_col]].rename(
        columns={src_col: tgt_col, tgt_col: src_col, val_col: "rev_val"}
    )
    f = f.merge(rev, on=[src_col, tgt_col], how="left")
    f["rev_val"] = f["rev_val"].fillna(0)

    numerator = f[val_col] + f["rev_val"]
    denominator = (
        f["out_i"].fillna(0) + f["in_j"].fillna(0) + f["out_j"].fillna(0) + f["in_i"].fillna(0)
    )
    f["phi"] = np.where(denominator > 0, numerator / denominator, 0.0)

    return f[[src_col, tgt_col, "phi"]]
