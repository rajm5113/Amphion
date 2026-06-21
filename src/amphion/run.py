"""Command-line entry point for the assembled pipeline.

    python -m amphion.run --n 5000 --out reports/shortlist.csv
"""

from __future__ import annotations

import argparse

from .pipeline import run


def main():
    ap = argparse.ArgumentParser(description="Amphion: rank novel AMP candidates into a shortlist.")
    ap.add_argument("--n", type=int, default=None, help="limit to the first N candidates")
    ap.add_argument("--candidates", default=None, help="path to candidates.csv (default data/interim/)")
    ap.add_argument("--out", default=None, help="output shortlist.csv path (default reports/)")
    args = ap.parse_args()

    df = run(n=args.n, candidates_csv=args.candidates, out_csv=args.out)
    print(f"Shortlist: {len(df)} candidates passed all gates.")
    if len(df):
        print("Top 5:")
        for _, r in df.head(5).iterrows():
            print(f"  {r['sequence']:<40s} rank={r['rank_score']:.3f}  amp={r['amp_prob']:.2f}  "
                  f"MIC={r['pred_mic_uM']:.1f}uM  hemo={r['hemolytic_prob']:.2f}")


if __name__ == "__main__":
    main()
