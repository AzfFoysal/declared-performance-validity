# Declared-Performance Validity

Code for the working paper *When the World Changes but the Model Doesn't: Distribution Shift as a Blind Spot in AI Compliance, and a Statistical Framework for Declared-Performance Validity*.

AI regulation asks providers to declare how accurate a system is (EU AI Act Art. 15(3)) and re-assesses the system when the system is modified (Art. 3(23), 43(4)). When the deployment population drifts instead, the declared figure can become false with no modification at all. The only route left is corrective action once a provider has "reason to consider" non-conformity (Art. 20), and nothing says what evidence counts. This repository turns that phrase into a test.

## The idea in one paragraph

A declaration is valid until a pre-registered sequential test on randomly audited labels rejects the hypothesis that deployed error stays within the declared error plus a tolerance. The test is a "testing by betting" monitor (Timans et al., UAI 2025), so the chance of ever raising a false alarm is at most delta over the whole deployment. Labels are expensive, so a label-free accuracy estimate (ATC, Garg et al., ICLR 2022) decides how often to audit, never whether the declaration has failed. Items are always chosen for audit uniformly at random, which keeps the guarantee intact.

## Repository layout

| Path | What it is |
| --- | --- |
| `monitor.py` | Betting risk monitor (Tier 2), ATC estimator (Tier 1), audit policy, stream runner |
| `simulate.py` | Stylised classifier and scenario definitions used by the simulations |
| `experiments.py` | All simulation experiments in the paper (main table, sensitivity, label delay, effect size, false-alarm level, trajectories) |
| `case_digits.py` | Real-classifier study: neural network on UCI handwritten digits under sensor noise, blur, camera shift, gradual noise and a labeling-convention change |
| `figures.py` | Paper figures from the results files |
| `case_study.py` | Runs the monitor on real model outputs (`outputs.npz`) |
| `thesis_integration/export_outputs.py` | Exports 27-class aligned predictions from the plant-disease thesis model (PlantVillage to PlantDoc) |
| `results/simulation_results.csv` | Output of `simulate.py` |
| `tests/` | Checks of the false-alarm bound, detection, ATC, label savings, class mapping and masking |
| `docs/literature_log.xlsx` | Annotated literature log (20 papers, legal sources, reading queue) |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q          # 7 tests, about 20 seconds
python experiments.py        # about 3 minutes on 2 cores; writes results/*.csv
python case_digits.py        # about 6 minutes; writes results/digits*.{csv,json}
python figures.py            # writes paper/figures/*.pdf
```

## Simulation results

Declared accuracy 96.3%, tolerance 2 points (violation below 94.3%), delta = 0.05, 20,000-item streams, change at item 5,000. Each cell: detection rate within 15,000 items of the change · median delay in items · mean labels used.

| Scenario | Fixed 1% audits | Two-tier 1% to 10% | Fixed 10% audits |
| --- | --- | --- | --- |
| No shift, false alarms | 0.0% · 200 labels | 0.0% · 240 labels | 0.0% · 2,000 labels |
| On the boundary (94.3%), false alarms | 0.0% | 1.3% | 2.0% |
| Abrupt covariate shift to 78% | 99% · 6,532 · 118 | 100% · 878 · 130 | 100% · 894 · 596 |
| Gradual covariate shift to 78% | 56% · 10,936 · 188 | 100% · 4,301 · 545 | 100% · 4,506 · 1,157 |
| Mild covariate shift to 92.5% | 2% · 11,557 · 200 | 48% · 8,621 · 1,216 | 52% · 8,190 · 1,646 |
| Concept drift to 78% | 100% · 6,224 · 116 | 99% · 5,720 · 130 | 100% · 929 · 596 |

Under covariate shift the two-tier design matches the 10% policy's delay with 22% to 47% of its labels. Under concept drift, where confidence does not change, the label-free tier is blind and detection relies on the 1% random-audit floor, which is why that floor must stay above zero.

## Real-classifier results (results/digits.csv)

Declared accuracy 97.2% (clean test set), violation level 95.2%, 300 runs per condition. No policy raised a false alarm in any condition. Median items to invalidation (mean labels):

| Condition | Deployed | Fixed 1% | Two-tier, relative trigger | Fixed 10% |
| --- | --- | --- | --- | --- |
| Sensor noise | 79.8% | 6,494 (118) | 885 (128) | 918 (593) |
| Focus drift | 84.9% | 9,046 (153) | 1,282 (172) | 1,354 (642) |
| Camera shift, 1 px | 46.5% | 1,801 (69) | 412 (78) | 306 (531) |
| Gradual noise | 66.3% | 8,053 (161) | 3,191 (392) | 3,131 (1,114) |
| Labeling convention (concept drift) | 77.7% | 5,591 (105) | 4,902 (118) | 881 (586) |

The relative trigger compares the label-free estimate with its own reading on clean test data; an absolute trigger fired constantly on clean data because the estimator read 2 points low, which cost labels but never produced a false finding.

## Plant-disease case study (in progress)

On the machine that holds the trained model and data, from the root of the thesis repository:

```bash
python thesis_integration/export_outputs.py --model hybrid \
    --weights checkpoints/hybrid_full_seed42.pth --data_dir data --out outputs.npz
python case_study.py outputs.npz
```

The export keeps only the 27 classes shared by PlantVillage and PlantDoc in every split and masks the other 11 before the argmax, so the declared (lab) and deployed (field) accuracies are measured on the same label set. It stops with a clear error if any class name fails to match.

## Key references

- Timans, Verma, Nalisnick, Naesseth. On Continuous Monitoring of Risk Violations under Unknown Shift. UAI 2025.
- Garg, Balakrishnan, Lipton, Neyshabur, Sedghi. Leveraging Unlabeled Data to Predict Out-of-Distribution Performance. ICLR 2022.
- Waudby-Smith, Ramdas. Estimating means of bounded random variables by betting. JRSS-B 2024.
- Regulation (EU) 2024/1689 (AI Act), Articles 3(23), 13, 15, 20, 43(4), 72.

## Author

Md Abu Zehad Foysal, Troy University. Contact: md.zehad.foysal@gmail.com
