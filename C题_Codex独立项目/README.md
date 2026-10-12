# 2025 MCM Problem C: Olympic Medal Modeling

This is an independent study project using the five supplied official CSV inputs. It includes runnable Python code, results, validation records and an editable report.

## Read the project

- [Chinese technical explanation](report/中文技术说明.md)
- [English report PDF](report/solution.pdf)
- [Editable LaTeX source](report/solution.tex)
- [English Markdown](report/solution.md)
- [2028 prediction table](outputs/tables/predictions_2028.csv)
- [Claim/evidence mapping](outputs/validation/claim_evidence_map.md)

The PDF was exported by ReportLab. The recorded native TeX compilation failed; it is not a verified compilation of the LaTeX source.

## Run locally

Use Python 3.12 and create your own virtual environment. The local environment and caches are not distributed.

```sh
python -m venv .venv
# Activate the environment using the command appropriate for your operating system.
python -m pip install -r requirements-lock.txt
python -u -X utf8 scripts/run_all.py
python -X utf8 scripts/build_report.py
python -X utf8 scripts/verify_report.py
```

The package retains the existing source inputs and computed results. Packaging and privacy checks do not constitute a new modeling run.

## Interpretation

Historical weighted baselines performed best on point-error checks; the ensemble adds scenario and first-medal analysis. The 2024 checks are retrospective because development included that edition. The 2028 event, participation and medal-budget settings are assumptions. Coach interventions are labeled scenarios, not identified causal effects. Consult the technical explanation and validation records for limitations.
