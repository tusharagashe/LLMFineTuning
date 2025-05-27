# Monoclonal Antibody Trial Recovery Dataset

## Overview

This repository provides a curated dataset of monoclonal antibody (mAb) clinical trials that experienced early **termination, withdrawal, or suspension**, and were subsequently **resubmitted or redesigned** and reached **completed status**. This dataset is designed to support research in clinical trial optimization, AI-guided trial design, and failure-driven drug development.

The resulting CSV (`monoclonal_antibody_trial_pairs.csv`) includes:
- NCT IDs of failed and completed trials
- Matching interventions and conditions
- Key design changes
- Clinical outcomes

## Objective

To identify real-world examples of iterative learning in drug development, particularly where **monoclonal antibody trials failed and were later restructured into completed studies**, allowing:
- Comparison of protocol revisions
- Training of LLM/RAG agents for clinical design
- Historical risk analysis of mAb pipelines

## Data Sources

- [ClinicalTrials.gov](https://clinicaltrials.gov/data-api/api)

## Methodology

### 1. Define Inclusion Criteria

- **Therapeutic Class**: Trials investigating **monoclonal antibodies (mAbs)** only.
- **Failure Definition**: A trial was considered "failed" if it was **Terminated**, **Withdrawn**, or **Suspended** per ClinicalTrials.gov status labels. This reflects trials that were **stopped early before planned completion**, regardless of the scientific merit of results.
  - *Note: Trials that completed but failed to meet endpoints (i.e., “negative results”) are not considered failed in this dataset.*
- **Success Definition**: Recruitment status of `Completed`.
- **Required**: Both trials must have valid **NCT IDs** and be accessible via ClinicalTrials.gov.

### 2. Query via ClinicalTrials.gov

**Search Strategy (Failure Trials):**
```
Other terms:           Monoclonal Antibody
Recruitment Statuses : TERMINATED, WITHDRAWN, SUSPENDED
```

**Extracted Fields:**
- `NCTId`
- `BriefTitle`
- `Condition`
- `Sponsor`
- `InterventionName`
- `OverallStatus`

### 3. Candidate Matching Logic (Failed → Completed)

For each failed trial:
- Search for **completed trials** sharing:
  - The same or similar `InterventionName` (e.g., Tanezumab)
  - Same `SponsorName`
  - Matching or overlapping `Condition`

**Search Strategy (Completed Trials):**
```
Other terms:           Monoclonal Antibody
Recruitment Statuses : COMPLETED
```

### 4. Manual Validation Criteria

Each proposed pair was reviewed manually to confirm:
- Continuity in therapeutic program (e.g., Phase II to Phase III)
- Clear design changes (e.g., dose, population, endpoint)
- Evidence in public records (e.g., FDA re-approvals, publications)

**Exclusions:**
- Non-mAb drugs (e.g., Remdesivir, Tenecteplase)
- Trials lacking a direct lineage (different disease, different sponsor)
- Trials without NCT identifiers for both failure and completion

### 5. Dataset Construction

Each matched pair includes:

| Column                    | Description                                                              |
|---------------------------|--------------------------------------------------------------------------|
| `Monoclonal Antibody`     | Drug or combination used                                                 |
| `Failed Trial NCT ID`     | NCT ID of the withdrawn/terminated trial                                 |
| `Completed Trial NCT ID`  | NCT ID of the resubmitted/completed trial                                |
| `Condition`               | Disease/indication targeted                                              |
| `Key Changes`             | Summary of design changes between failed and successful trials           |
| `Outcome`                 | Final clinical result or regulatory decision                             |

The dataset is saved as `monoclonal_antibody_trial_pairs.csv`.
