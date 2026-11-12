# JobPilot

JobPilot is a Python prototype for tracking job applications and extracting
structured information from public job postings with Gemini. It also includes a
local-snapshot evaluation pipeline for measuring extraction quality against
human-reviewed annotations.

> **Prototype status:** The tracker and extraction workflows are small
experiments, not a production service. Evaluation data and source snapshots are
private and are not included in this repository.

## What it does

- Reads a Google Sheet and turns a natural-language application update into a
  proposed new row. It checks for a possible duplicate by company and role,
  previews the row, and requires explicit confirmation before appending. It
  does not update existing rows.
- Retrieves public job posting pages, extracts readable text and
  `JobPosting` JSON-LD when available, and asks Gemini to return structured
  fields validated against a Pydantic schema.
- Runs an evaluation against saved local snapshots, without accessing the
  production Google Sheet or re-fetching websites.

The tracker workflow is interactive and can append to the configured sheet.
The `extract` and `eval` commands do not access the sheet. Use a test sheet if
trying the tracker workflow.

## Architecture

```text
app.py
  ├─ tracker flow ── Gemini application parsing ── duplicate check ── confirmed append to Google Sheets
  ├─ extract flow ── public-page retrieval ── Gemini structured extraction ── Pydantic validation
  └─ eval flow ──── saved snapshots ── Gemini extraction ── field/record metrics ── CSV and JSON

src/scraper.py      Public HTTP(S) retrieval, redirect checks, HTML and JobPosting JSON-LD parsing
src/extractor.py    Gemini request, JSON response handling, token metadata capture
src/schemas.py      Pydantic JobPosting schema and constrained categories
src/evaluation.py   Annotation validation, snapshot workflow, scoring, and result export
tests/              unittest coverage for scraper, extractor, evaluation, and CLI isolation
```

### Technology stack

- Python
- Google Gemini through the `google-genai` SDK
- Pydantic for structured output validation
- Requests and Beautiful Soup for HTTP and HTML parsing
- Google Sheets API through `google-api-python-client` and `google-auth-oauthlib`
- `python-dotenv` for local environment configuration
- Python `unittest` for automated tests

## Setup

Use Python 3.10 or later. Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a local `.env` file:

```dotenv
GEMINI_API_KEY=your-gemini-api-key
# Optional. Extraction defaults to gemini-3.1-flash-lite.
GEMINI_MODEL=gemini-3.1-flash-lite
GOOGLE_SHEET_ID=your-test-spreadsheet-id
```

The interactive tracker additionally requires a Google service-account key at
`service-account.json`, with the Google Sheets API enabled and the target sheet
shared with that service account. **Never commit `.env`,
`service-account.json`, API keys, or real spreadsheet IDs.** Use a test sheet
when experimenting with the append workflow.

Run the interactive tracker:

```bash
python app.py
```

Run public posting extraction without touching the spreadsheet:

```bash
python app.py extract 'https://careers.example.com/jobs/123'
```

This retrieval supports public static HTTP(S) pages. It does not render
JavaScript, authenticate, bypass CAPTCHAs, or circumvent access controls. If
automatic retrieval fails in the interactive extraction flow, a user can paste
the job description instead.

## Evaluation methodology

Evaluation is performed with saved posting snapshots and an annotation CSV.
Every annotation row must match a snapshot by posting ID and split. Failed
retrievals, missing snapshots, and metadata mismatches are separated from
Gemini extraction errors. A local snapshot rerun does not retrieve the website
again. It still sends the snapshot text to Gemini.

The 12 scored fields are company, job title, location, work mode, salary amount,
salary currency, salary period, compensation details, other benefits,
application deadline, contract duration, and employment type. The supplied
`job_url` is excluded from primary accuracy and reported separately as URL
preservation.

Annotation and metric conventions:

- A blank value is unannotated and excluded from scoring. `__MISSING__` means
  the reviewer checked the source and confirmed the field was absent. For
  categorical fields, `unknown` is scored as missing.
- Salary amount is numeric only; currency and pay period are separate.
  Financial compensation details (such as bonuses, pension, and leave) and
  non-financial benefits (such as wellbeing and professional development) are
  separate list fields.
- Scalar exact matching applies formatting normalization only; it does not
  award semantic partial credit. List scoring is deterministic, order
  insensitive, and based on exact normalized items (Unicode NFKC, case-folding,
  whitespace collapse, punctuation treated as spaces, duplicate items removed).
  List-item precision, recall, and F1 are micro-averaged.
- Field-presence precision, recall, and F1 measure whether a value was
  predicted, even if its content is wrong. The report separately includes
  present-value accuracy and accuracy for genuinely missing fields.
- Record exact match requires every annotated scored field in that posting to
  match. Error categories distinguish unsupported predictions, omissions,
  incorrect values, and partial-value mismatches. Human annotation mismatch
  flags are reported separately.
- Website retrieval time is the original collection time saved in the snapshot;
  local snapshot load time and Gemini inference time are separate. Total
  extraction time combines original retrieval and inference.

To collect a snapshot:

```bash
python app.py eval collect \
  --posting-id job-01 \
  --split development \
  --url 'https://careers.example.com/jobs/123' \
  --snapshot-dir data/snapshots
```

Prepare a version-2 annotation CSV with the required columns and reviewed gold
values. List values must be JSON arrays, for example
`["annual bonus","pension plan"]`. Then run an evaluation into a new output
directory:

```bash
python app.py eval run \
  --annotations data/evaluation/annotations.csv \
  --snapshot-dir data/snapshots \
  --output-dir data/evaluation/baseline
```

The command reports aggregate, development, and test-set metrics and writes
predictions and evaluation results as CSV and JSON. Do not tune against the
held-out test set. Evaluation output directories are not overwritten unless
`--overwrite` is explicitly supplied. Use your own appropriately authorized
annotations and snapshots; the private benchmark dataset is not published.

## Baseline benchmark (November 2026)

The baseline used the existing extraction prompt and Gemini 3.1 Flash-Lite on
**19 approved postings** with **228 human-verified field annotations**. The
annotations were AI-assisted and independently reviewed and approved by the
dataset owner. The benchmark did not tune the prompt or model.

| Metric | Overall | Development | Held-out test |
|---|---:|---:|---:|
| Postings | 19 | 11 | 8 |
| Field exact accuracy | **163/228 — 71.5%** | 100/132 — 75.8% | 63/96 — 65.6% |
| Record exact match | 0/19 — 0% | 0/11 — 0% | 0/8 — 0% |

Additional measured results:

- Field-presence F1: **95.0%** overall (91.6% precision, 98.7% recall).
- Present-value accuracy: 103/154 (66.9%); genuinely missing-field accuracy:
  60/74 (81.1%).
- Mean successful-inference latency: **2.95 seconds**; median: 1.83 seconds.
- List-item precision, recall, and F1: 18.5%, 23.4%, and 20.7%.
- 30 URLs were attempted; 24 retrievals succeeded, 6 failed, and 5 successful
  snapshots were later excluded for quality, leaving the 19 evaluated postings.
- Gemini reported 71,915 total tokens across the 19 successful inferences.

The small held-out set scored 65.6% field accuracy; it was not used for prompt
optimization. The 71.5% overall accuracy and 95.0% presence F1 describe this
specific dataset and schema, not general job-posting extraction performance.

## Limitations and risks

- The benchmark is small and limited to one approved dataset; results may not
  generalize across countries, job boards, languages, or posting formats.
- Benefits extraction remains weak: combined list-item F1 was 20.7%. Location
  exact accuracy was 36.8%, and no record achieved a perfect match across all
  12 scored fields.
- Extraction quality depends on the text returned by the site. JavaScript-only
  pages and access-restricted pages are not supported; irrelevant page text can
  still affect results.
- Gemini inference transmits retrieved posting text to the configured Google
  model. Review the provider's data and billing terms before use; API quota or
  availability errors can interrupt a benchmark.
- The tracker append flow is a prototype and has no comprehensive audit log or
  rollback mechanism. It requires explicit confirmation, but users should test
  with a non-production sheet.
- Evaluation data, annotations, snapshots, generated predictions, and private
  experiment outputs are local-only and intentionally excluded from Git.

Run the automated tests (no live Gemini or Google Sheets calls):

```bash
python -m unittest discover -s tests -v
```

## Planned work

Potential future work includes safer updates to existing tracker rows,
structured audit logs, optional read-only Gmail triage, and improved handling
of irrelevant page content. These are not part of the verified benchmark.
