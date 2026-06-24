# Amphion shortlist — novel, predicted-active, predicted-safe candidates

**Pipeline:** generate → validity → novelty → activity → toxicity → rank (config-driven).

| Stage | Count |
|---|---:|
| Valid unique candidates scored | 5,000 |
| Passed activity gate (amp_prob ≥ 0.5) | 4,355 |
| Passed safety gate (hemolytic_prob ≤ 0.5) | 3,155 |
| Passed novelty gate (identity < 0.6) | 1,321 |
| **Shortlist (passed ALL gates)** | **496** |

## Top candidates
Ranked by the composite score (weights from `config.yaml`:
amp 0.3, potency 0.3, safety 0.25,
novelty 0.1, synth 0.05). `amp_prob` shown with its
ensemble uncertainty (±). Full provenance for every candidate is in `shortlist.csv`.

| # | sequence | rank | amp_prob | pred MIC (µM) | hemolytic | novelty | synth |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `FLGGLIKKARPVWVPRVVPAVKKFIKK` | 0.825 | 0.99±0.01 | 1.0 | 0.21 | 0.40 | 1.00 |
| 2 | `FLGVVFKKVPSVFGAVGKTVGKFWIKGAREGIFEYGKQINYID` | 0.817 | 0.99±0.01 | 0.3 | 0.47 | 0.46 | 0.80 |
| 3 | `NGLQSIVGSAIGKGIPELVHGILSAGPKLAEHL` | 0.817 | 0.99±0.01 | 1.6 | 0.13 | 0.41 | 0.95 |
| 4 | `STIGKLEKAVKGAIGGLRAIRRHL` | 0.817 | 0.99±0.01 | 1.9 | 0.12 | 0.43 | 1.00 |
| 5 | `SILTGTKTLAKKLASTILGKKKRKAG` | 0.812 | 0.99±0.01 | 1.7 | 0.15 | 0.41 | 1.00 |
| 6 | `LRAKKAFKKARVYPNYVRIPLRG` | 0.812 | 0.96±0.04 | 1.0 | 0.23 | 0.45 | 1.00 |
| 7 | `GAWDVIKKVVPAVAGVVKEGGKNIKKKK` | 0.810 | 0.99±0.01 | 2.1 | 0.11 | 0.41 | 1.00 |
| 8 | `GRPNPVPNGNPNGPRPPYNPGNPGYPGRPPPFPRPRPPFG` | 0.808 | 0.93±0.13 | 0.8 | 0.21 | 0.42 | 0.85 |
| 9 | `FLGKVVKGAIKWVPAFSRRYPAIRYMR` | 0.804 | 0.96±0.04 | 1.0 | 0.26 | 0.40 | 1.00 |
| 10 | `CVGHKVIPVLVRIKRKC` | 0.803 | 0.97±0.04 | 2.4 | 0.10 | 0.44 | 1.00 |
| 11 | `GFMSKVWNAARKVGNKVAPAVANVMAEKAMGLIKKS` | 0.802 | 0.99±0.01 | 1.2 | 0.26 | 0.46 | 0.91 |
| 12 | `GKNRNKAICVSIGACLPAWKVCKL` | 0.802 | 0.95±0.04 | 1.2 | 0.21 | 0.46 | 0.90 |
| 13 | `NGVQPKYTGHGWHFRPRGPQHQKAWWATGAWAWVWAAW` | 0.800 | 0.96±0.04 | 1.1 | 0.25 | 0.47 | 0.88 |
| 14 | `KAHHGIHVGLPQLFLGIHLIRKGGIHIGFIHKIGNSIQGD` | 0.798 | 0.99±0.01 | 0.6 | 0.43 | 0.50 | 0.85 |
| 15 | `KIAHKAGKMATKVLPAVVDVLKGCKS` | 0.797 | 0.99±0.01 | 3.3 | 0.08 | 0.43 | 1.00 |
| 16 | `GLPGVVSKVVSKFRQTVKRKFAN` | 0.796 | 0.99±0.01 | 3.3 | 0.09 | 0.44 | 1.00 |
| 17 | `GIIKTIVSKIKSTGQQAKLGVTNVLDQAKCKIDGC` | 0.795 | 0.98±0.03 | 2.5 | 0.11 | 0.40 | 0.93 |
| 18 | `HTALHMLARLPRRLKSTATK` | 0.795 | 0.95±0.04 | 2.7 | 0.09 | 0.45 | 1.00 |
| 19 | `RPAFKAIAQGVLRHAGNAIQRIANEIW` | 0.794 | 0.97±0.03 | 2.1 | 0.19 | 0.51 | 1.00 |
| 20 | `GNFRKAGTKFRKGALEYGGAALEKLGEKLRQKAQKAINAHP` | 0.790 | 0.99±0.01 | 1.3 | 0.30 | 0.50 | 0.83 |

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
