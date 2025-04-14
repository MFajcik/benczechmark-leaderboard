from typing import Sequence
import numpy as np
from statsmodels.stats.multitest import multipletests

def correct_pvals_for_model(
    pvals: Sequence[float],
    fdr_alpha: float = 0.05,
    fdr_correction_method: str = 'fdr_bh'
) -> np.ndarray:
    """
    Adjusts a list of p-values for multiple hypothesis testing using a specified
    False Discovery Rate (FDR) correction method.

    Parameters:
    ----------
    pvals : Sequence[float]
        A sequence (e.g., list, array) of p-values corresponding to duels of single model.
        if leaderboard contains N models, the length of pvals sequence should be N-1

    fdr_alpha : float, optional
        The desired false discovery rate level (default is 0.05).
    fdr_correction_method : str, optional
        The FDR correction method to use. Supported methods include:
        - 'fdr_bh' (Benjamini-Hochberg)
        - 'fdr_by' (Benjamini-Yekutieli)
        - 'fdr_tsbh' (Two-stage Benjamini-Hochberg)
        - 'fdr_tsbky' (Two-stage Benjamini-Krieger-Yekutieli)
        (default is 'fdr_bh').

    Returns:
    -------
    np.ndarray
        An array of p-values adjusted for multiple comparisons.
    """
    _, corrected_pvals, _, _ = multipletests(pvals, alpha=fdr_alpha, method=fdr_correction_method)
    return corrected_pvals
