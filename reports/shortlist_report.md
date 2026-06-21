# Amphion shortlist — novel, predicted-active, predicted-safe candidates

**Pipeline:** generate → validity → novelty → activity → toxicity → rank (config-driven).

| Stage | Count |
|---|---:|
| Valid unique candidates scored | 5,000 |
| Passed activity gate (amp_prob ≥ 0.5) | 4,234 |
| Passed safety gate (hemolytic_prob ≤ 0.5) | 3,287 |
| Passed novelty gate (identity < 0.6) | 1,321 |
| **Shortlist (passed ALL gates)** | **467** |

## Top candidates
Ranked by the composite score (weights from `config.yaml`:
amp 0.3, potency 0.3, safety 0.25,
novelty 0.1, synth 0.05). `amp_prob` shown with its
ensemble uncertainty (±). Full provenance for every candidate is in `shortlist.csv`.

| # | sequence | rank | amp_prob | pred MIC (µM) | hemolytic | novelty | synth |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `TAVVKVKVKVSDKLAKGPVSIGPSIVVNHAHEALKKL` | 0.811 | 0.98±0.02 | 1.6 | 0.26 | 0.50 | 0.90 |
| 2 | `KAALHCFIAKKPKKVRCK` | 0.810 | 0.99±0.02 | 1.1 | 0.33 | 0.43 | 1.00 |
| 3 | `MLAKIISKIILTLKKLLPKLKPSAIPAKT` | 0.809 | 0.97±0.03 | 1.5 | 0.26 | 0.42 | 1.00 |
| 4 | `QRRFHRLVKFEKHFKRKFHRKLH` | 0.806 | 0.99±0.02 | 2.2 | 0.20 | 0.41 | 0.99 |
| 5 | `AAGRLRVQVRKLKKKIRDFLVPQIK` | 0.805 | 0.99±0.02 | 1.2 | 0.34 | 0.43 | 1.00 |
| 6 | `CVGHKVIPVLVRIKRKC` | 0.802 | 0.99±0.02 | 2.1 | 0.24 | 0.44 | 1.00 |
| 7 | `GRPNPVPNGNPNGPRPPYNPGNPGYPGRPPPFPRPRPPFG` | 0.802 | 0.89±0.08 | 0.5 | 0.40 | 0.42 | 0.85 |
| 8 | `SILTGTKTLAKKLASTILGKKKRKAG` | 0.798 | 0.99±0.02 | 1.4 | 0.33 | 0.41 | 1.00 |
| 9 | `NGVQPKYTGHGWHFRPRGPQHQKAWWATGAWAWVWAAW` | 0.798 | 0.99±0.02 | 1.0 | 0.40 | 0.47 | 0.88 |
| 10 | `CGGWTRRCWSFRSGGWWLRKFLRRKIRGNRGPRWRGSGR` | 0.797 | 0.96±0.05 | 1.2 | 0.34 | 0.51 | 0.78 |
| 11 | `TAEERAAEGNVLSPGKVLVVAVSWVWFPKGKWKVKW` | 0.795 | 0.95±0.05 | 1.5 | 0.31 | 0.50 | 0.91 |
| 12 | `RLGNVLTPILRAIVRIIRKTARANEKK` | 0.794 | 0.98±0.02 | 1.2 | 0.39 | 0.47 | 1.00 |
| 13 | `GIWKTIKSMAKGVLKALAEKVANKLKKKAQKNPPGWDVIGTGA` | 0.793 | 0.99±0.02 | 0.9 | 0.40 | 0.42 | 0.80 |
| 14 | `MGAAIKAGLGKIGKTFAKGGARQGIKAIAIDWLGRKAGNWKEIGEGLNAK` | 0.793 | 0.99±0.02 | 1.1 | 0.37 | 0.45 | 0.70 |
| 15 | `VVDKLTSFPTFAKPAKATFKVHSAKFKVKVFKGVLKVFHVEAASLSS` | 0.791 | 0.99±0.02 | 0.7 | 0.48 | 0.48 | 0.74 |
| 16 | `SIGRKIRTKYRKTVIKSIKRWKKRLK` | 0.791 | 0.94±0.04 | 1.6 | 0.26 | 0.41 | 0.92 |
| 17 | `GFGCPFNNANCAHVLSGVKANGKVGYGANPKDFEFKFEKL` | 0.789 | 0.89±0.09 | 0.8 | 0.37 | 0.48 | 0.85 |
| 18 | `RKSCARVCLSRKTFRCLVTSNVLSTLWLRTGCKDVEVRCKDEC` | 0.789 | 0.97±0.02 | 0.8 | 0.41 | 0.53 | 0.51 |
| 19 | `QGGVGFSTVAGVKWLSKWLLKKKW` | 0.788 | 0.99±0.02 | 1.5 | 0.37 | 0.44 | 1.00 |
| 20 | `FNVKALSKLLKQLKKIVGKAAGLAVKLTNKLPAAK` | 0.788 | 0.99±0.02 | 1.2 | 0.39 | 0.42 | 0.93 |

## How to read this
- **rank_score** — weighted blend of the columns to its right; higher = a better all-round candidate.
- **amp_prob ± unc** — calibrated probability the peptide is antimicrobial, with ensemble uncertainty.
- **pred MIC (µM)** — predicted best potency; lower = more potent (regressor is the weakest model — treat as a coarse guide).
- **hemolytic** — predicted probability it harms human red blood cells; lower = safer.
- **novelty** — 1 − (max identity to any known sequence); higher = more original.
- **synth** — heuristic ease-of-synthesis (length / charge / cysteine / hydrophobicity).

> **These are computational predictions with uncertainty, not validated hits.** No
> software confirms a real bacterial kill or human-cell safety — that requires a wet
> lab (Stage 3, out of scope). This shortlist is a prioritized starting point for
> experimental validation, nothing more.
