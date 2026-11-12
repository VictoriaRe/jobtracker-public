"""Reproducible, local-snapshot evaluation for job-posting extraction."""

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from src.extractor import extract_job_posting_with_metadata
from src.scraper import fetch_job_posting

ANNOTATION_FIELDS = (
    "company_name",
    "job_title",
    "location",
    "work_mode",
    "salary_amount",
    "salary_currency",
    "salary_period",
    "compensation_details",
    "other_benefits",
    "application_deadline",
    "contract_duration",
    "employment_type",
    "job_url",
)
LIST_FIELDS = frozenset(("compensation_details", "other_benefits"))
SCORED_FIELDS = tuple(field for field in ANNOTATION_FIELDS if field != "job_url")
ANNOTATION_VERSION = "2"
ANNOTATION_COLUMNS = (
    "annotation_version",
    "posting_id",
    "split",
    "source_url",
    *ANNOTATION_FIELDS,
    "annotation_mismatch_fields",
)
MISSING_VALUE = "__MISSING__"
SPLITS = ("development", "test", "pilot")
_UNKNOWN_FIELDS = {"work_mode", "contract_duration", "employment_type"}
_CURRENCY_SYMBOLS = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "C$": "CAD",
    "A$": "AUD",
}
_PERIOD_ALIASES = {
    "year": "annual",
    "yearly": "annual",
    "annually": "annual",
    "per year": "annual",
    "per annum": "annual",
    "annual": "annual",
    "month": "monthly",
    "per month": "monthly",
    "monthly": "monthly",
    "week": "weekly",
    "per week": "weekly",
    "weekly": "weekly",
    "hour": "hourly",
    "per hour": "hourly",
    "hourly": "hourly",
    "day": "daily",
    "per day": "daily",
    "daily": "daily",
}
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
)
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_UNANNOTATED = object()


def _normalise_text(value):
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = "".join(char if char.isalnum() else " " for char in text)
    return " ".join(text.split())


def _normalise_amount(value):
    text = str(value).strip().casefold()
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    matches = list(
        re.finditer(
            r"(?<![\w.])(\d+(?:\.\d+)?)(?:\s*(k|m|thousand|million))?",
            text,
        )
    )
    if not matches:
        return _normalise_text(text)

    units = {"k": Decimal(1000), "thousand": Decimal(1000),
             "m": Decimal(1000000), "million": Decimal(1000000)}
    last_unit = matches[-1].group(2)
    amounts = []
    for index, match in enumerate(matches):
        try:
            amount = Decimal(match.group(1))
        except InvalidOperation:
            return _normalise_text(text)
        unit = match.group(2)
        if not unit and index == len(matches) - 2 and last_unit:
            unit = last_unit
        if unit:
            amount *= units[unit]
        amounts.append(format(amount.normalize(), "f"))
    return "|".join(amounts)


def _normalise_currency(value):
    text = unicodedata.normalize("NFKC", str(value)).strip().upper()
    for symbol in sorted(_CURRENCY_SYMBOLS, key=len, reverse=True):
        if symbol in text:
            return _CURRENCY_SYMBOLS[symbol]
    letters = "".join(char for char in text if char.isalpha())
    return letters or _normalise_text(text)


def _normalise_period(value):
    text = " ".join(str(value).casefold().replace("_", " ").split())
    text = text.strip(" ./")
    return _PERIOD_ALIASES.get(text, _normalise_text(text))


def _normalise_date(value):
    text = str(value).strip()
    for date_format in _DATE_FORMATS:
        try:
            return datetime.strptime(text, date_format).date().isoformat()
        except ValueError:
            continue
    return _normalise_text(text)


def normalise_field(field, value):
    """Canonicalise only formatting variants; do not award semantic partial credit."""
    if value is None:
        return None
    if field in LIST_FIELDS:
        if not isinstance(value, (list, tuple, set)):
            raise ValueError(f"{field} must be a list of strings.")
        values = {_normalise_text(item) for item in value if _normalise_text(item)}
        return frozenset(values) if value else None
    text = str(value).strip()
    if not text or (field in _UNKNOWN_FIELDS and text.casefold() == "unknown"):
        return None
    if field == "salary_amount":
        return _normalise_amount(text)
    if field == "salary_currency":
        return _normalise_currency(text)
    if field == "salary_period":
        return _normalise_period(text)
    if field == "application_deadline":
        return _normalise_date(text)
    if field == "job_url":
        try:
            parsed = urlsplit(text)
            if parsed.scheme and parsed.netloc:
                return urlunsplit(
                    (parsed.scheme.casefold(), parsed.netloc.casefold(),
                     parsed.path.rstrip("/") or "/", parsed.query, "")
                )
        except ValueError:
            pass
    return _normalise_text(text)


def _read_annotations(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing_columns = set(ANNOTATION_COLUMNS) - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"Annotation CSV is missing columns: {', '.join(sorted(missing_columns))}"
            )
        rows = list(reader)

    seen_ids = set()
    for row in rows:
        if (row.get("annotation_version") or "").strip() != ANNOTATION_VERSION:
            raise ValueError(
                f"Annotation row must use version {ANNOTATION_VERSION}; "
                f"got {row.get('annotation_version')!r}."
            )
        posting_id = (row.get("posting_id") or "").strip()
        if not _ID_PATTERN.fullmatch(posting_id):
            raise ValueError(f"Invalid or empty posting_id in annotation CSV: {posting_id!r}")
        if posting_id in seen_ids:
            raise ValueError(f"Duplicate posting_id in annotation CSV: {posting_id}")
        seen_ids.add(posting_id)
        split = (row.get("split") or "").strip()
        if split not in SPLITS:
            raise ValueError(
                f"Unsupported split {split!r} for {posting_id}; expected {', '.join(SPLITS)}."
            )
        row["posting_id"] = posting_id
        row["split"] = split
        for field in ANNOTATION_FIELDS:
            value = (row.get(field) or "").strip()
            if not value:
                row[field] = _UNANNOTATED
            elif value == MISSING_VALUE:
                row[field] = None
            elif field in LIST_FIELDS:
                try:
                    items = json.loads(value)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{posting_id}.{field} must be a JSON string array."
                    ) from error
                if not isinstance(items, list) or not all(
                    isinstance(item, str) and item.strip() for item in items
                ):
                    raise ValueError(
                        f"{posting_id}.{field} must be a JSON array of non-empty strings."
                    )
                row[field] = items or None
            else:
                row[field] = value
        mismatch_value = (row.get("annotation_mismatch_fields") or "").strip()
        if mismatch_value:
            try:
                mismatches = json.loads(mismatch_value)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{posting_id}.annotation_mismatch_fields must be a JSON string array."
                ) from error
            if not isinstance(mismatches, list) or not all(
                field in SCORED_FIELDS for field in mismatches
            ):
                raise ValueError(
                    f"{posting_id}.annotation_mismatch_fields contains unknown fields."
                )
            row["annotation_mismatch_fields"] = sorted(set(mismatches))
        else:
            row["annotation_mismatch_fields"] = []
    return rows


def collect_snapshot(posting_id, split, url, snapshot_dir):
    if not _ID_PATTERN.fullmatch(posting_id):
        raise ValueError("posting_id may contain only letters, numbers, '.', '_' and '-'.")
    if split not in SPLITS:
        raise ValueError(f"split must be one of: {', '.join(SPLITS)}.")
    started = time.perf_counter()
    snapshot = {
        "snapshot_version": 1,
        "posting_id": posting_id,
        "split": split,
        "source_url": url,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        content = fetch_job_posting(url)
    except Exception as error:
        snapshot.update(
            retrieval_status="failed",
            retrieval_seconds=round(time.perf_counter() - started, 6),
            retrieval_error=f"{type(error).__name__}: {error}",
        )
        _write_snapshot(snapshot_dir, posting_id, snapshot)
        print(f"Website retrieval failed for {posting_id}: {snapshot['retrieval_error']}", file=sys.stderr)
        return False

    snapshot.update(
        retrieval_status="success",
        retrieval_seconds=round(time.perf_counter() - started, 6),
        content=content,
    )
    _write_snapshot(snapshot_dir, posting_id, snapshot)
    print(
        f"Saved {posting_id} snapshot "
        f"(website retrieval: {snapshot['retrieval_seconds']:.3f}s)."
    )
    return True


def _write_snapshot(snapshot_dir, posting_id, snapshot):
    target_dir = Path(snapshot_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{posting_id}.json"
    target.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _error_category(
    field, prediction, gold, prediction_present, gold_present
):
    prediction_value = prediction
    gold_value = gold
    if not gold_present and prediction_present:
        return "unsupported_prediction"
    if gold_present and not prediction_present:
        return "omission"
    if not prediction_present or not gold_present:
        return None
    if prediction_value == gold_value:
        return None
    if field in LIST_FIELDS:
        has_overlap = bool(prediction_value & gold_value)
    else:
        has_overlap = bool(
            set((prediction_value or "").split())
            & set((gold_value or "").split())
        )
    if has_overlap:
        return "partial_value_mismatch"
    return "incorrect_value"


def _score_predictions(records):
    field_counts = {
        field: Counter(
            correct=0,
            annotated=0,
            present_gold=0,
            present_correct=0,
            missing_gold=0,
            missing_correct=0,
            presence_tp=0,
            presence_fp=0,
            presence_fn=0,
            item_tp=0,
            item_fp=0,
            item_fn=0,
            formatting_equivalent=0,
        )
        for field in SCORED_FIELDS
    }
    record_counts = Counter(exact=0, scored=0)
    categories = Counter()
    unsupported_records = set()
    field_result_rows = []
    record_result_rows = []
    annotation_mismatch_rows = []
    annotation_mismatch_count = 0
    url_counts = Counter(scored=0, preserved=0)

    for record in records:
        for field in record.get("annotation_mismatch_fields", []):
            annotation_mismatch_count += 1
            annotation_mismatch_rows.append({
                "row_type": "annotation_mismatch",
                "posting_id": record["posting_id"],
                "field": field,
                "error_category": "annotation_mismatch",
                "count": 1,
            })
        if record["extraction_status"] != "success":
            continue
        scored_fields = []
        record_exact = True
        for field in SCORED_FIELDS:
            raw_gold = record["gold"].get(field, _UNANNOTATED)
            if raw_gold is _UNANNOTATED:
                continue
            counts = field_counts[field]
            counts["annotated"] += 1
            raw_prediction = record["prediction"].get(field)
            predicted = normalise_field(field, raw_prediction)
            gold = normalise_field(field, raw_gold)
            predicted_present = _value_is_present(field, raw_prediction)
            gold_present = _value_is_present(field, raw_gold)
            correct = predicted == gold
            counts["correct"] += int(correct)
            counts["present_gold"] += int(gold_present)
            counts["present_correct"] += int(correct and gold_present)
            counts["missing_gold"] += int(not gold_present)
            counts["missing_correct"] += int(not predicted_present and not gold_present)
            counts["presence_tp"] += int(predicted_present and gold_present)
            counts["presence_fp"] += int(predicted_present and not gold_present)
            counts["presence_fn"] += int(not predicted_present and gold_present)
            if correct and gold is not None and str(raw_prediction).strip() != str(raw_gold).strip():
                counts["formatting_equivalent"] += 1
            if field in LIST_FIELDS:
                predicted_items = predicted or frozenset()
                gold_items = gold or frozenset()
                counts["item_tp"] += len(predicted_items & gold_items)
                counts["item_fp"] += len(predicted_items - gold_items)
                counts["item_fn"] += len(gold_items - predicted_items)
            category = _error_category(
                field, predicted, gold, predicted_present, gold_present
            )
            if category:
                categories[category] += 1
                record_exact = False
                if category == "unsupported_prediction":
                    unsupported_records.add(record["posting_id"])
            scored_fields.append(correct)
            field_result_rows.append({
                "row_type": "field",
                "posting_id": record["posting_id"],
                "field": field,
                "annotated": 1,
                "correct": int(correct),
                "error_category": category or "",
                "annotation_mismatch": int(
                    field in record.get("annotation_mismatch_fields", [])
                ),
                "formatting_equivalent": int(
                    correct and gold is not None
                    and str(raw_prediction).strip() != str(raw_gold).strip()
                ),
            })

        source_url = normalise_field("job_url", record.get("source_url"))
        if source_url is not None:
            url_counts["scored"] += 1
            output_url = normalise_field(
                "job_url", (record["prediction"] or {}).get("job_url")
            )
            preserved = output_url == source_url
            url_counts["preserved"] += int(preserved)
            record["url_preserved"] = preserved
        else:
            record["url_preserved"] = None

        if scored_fields:
            record_counts["scored"] += 1
            record_counts["exact"] += int(record_exact)
            record_result_rows.append({
                "row_type": "record",
                "posting_id": record["posting_id"],
                "fields_scored": len(scored_fields),
                "exact_match": int(record_exact),
                "field_accuracy": round(sum(scored_fields) / len(scored_fields), 6),
                "url_preserved": record["url_preserved"],
            })

    field_summary = {}
    for field, counts in field_counts.items():
        presence_precision = _ratio(
            counts["presence_tp"], counts["presence_tp"] + counts["presence_fp"]
        )
        presence_recall = _ratio(
            counts["presence_tp"], counts["presence_tp"] + counts["presence_fn"]
        )
        presence_f1 = _ratio(
            2 * counts["presence_tp"],
            2 * counts["presence_tp"] + counts["presence_fp"] + counts["presence_fn"],
        )
        summary = {
            "annotated_fields": counts["annotated"],
            "exact_correct": counts["correct"],
            "exact_accuracy": _ratio(counts["correct"], counts["annotated"]),
            "present_fields": counts["present_gold"],
            "present_value_correct": counts["present_correct"],
            "present_value_accuracy": _ratio(
                counts["present_correct"], counts["present_gold"]
            ),
            "genuinely_missing_fields": counts["missing_gold"],
            "missing_fields_correct": counts["missing_correct"],
            "genuinely_missing_accuracy": _ratio(
                counts["missing_correct"], counts["missing_gold"]
            ),
            "presence_true_positive": counts["presence_tp"],
            "presence_false_positive": counts["presence_fp"],
            "presence_false_negative": counts["presence_fn"],
            "presence_precision": presence_precision,
            "presence_recall": presence_recall,
            "presence_f1": presence_f1,
            "formatting_equivalent": counts["formatting_equivalent"],
        }
        if field in LIST_FIELDS:
            item_precision = _ratio(
                counts["item_tp"], counts["item_tp"] + counts["item_fp"]
            )
            item_recall = _ratio(
                counts["item_tp"], counts["item_tp"] + counts["item_fn"]
            )
            summary["items"] = {
                "matched": counts["item_tp"],
                "predicted": counts["item_tp"] + counts["item_fp"],
                "gold": counts["item_tp"] + counts["item_fn"],
                "precision": item_precision,
                "recall": item_recall,
                "f1": _ratio(
                    2 * counts["item_tp"],
                    2 * counts["item_tp"] + counts["item_fp"] + counts["item_fn"],
                ),
            }
        field_summary[field] = summary

    annotated_total = sum(counts["annotated"] for counts in field_counts.values())
    correct_total = sum(counts["correct"] for counts in field_counts.values())
    present_total = sum(counts["present_gold"] for counts in field_counts.values())
    present_correct = sum(counts["present_correct"] for counts in field_counts.values())
    missing_total = sum(counts["missing_gold"] for counts in field_counts.values())
    missing_correct = sum(counts["missing_correct"] for counts in field_counts.values())
    presence_tp = sum(counts["presence_tp"] for counts in field_counts.values())
    presence_fp = sum(counts["presence_fp"] for counts in field_counts.values())
    presence_fn = sum(counts["presence_fn"] for counts in field_counts.values())
    list_item_counts = Counter()
    for field in LIST_FIELDS:
        list_item_counts.update({
            "tp": field_counts[field]["item_tp"],
            "fp": field_counts[field]["item_fp"],
            "fn": field_counts[field]["item_fn"],
        })
    summary = {
        "records_scored": record_counts["scored"],
        "record_exact_matches": record_counts["exact"],
        "record_exact_accuracy": _ratio(record_counts["exact"], record_counts["scored"]),
        "annotated_fields_scored": annotated_total,
        "field_exact_correct": correct_total,
        "field_exact_accuracy": _ratio(correct_total, annotated_total),
        "present_fields": present_total,
        "present_value_correct": present_correct,
        "present_value_accuracy": _ratio(present_correct, present_total),
        "genuinely_missing_fields": missing_total,
        "missing_fields_correct": missing_correct,
        "genuinely_missing_accuracy": _ratio(missing_correct, missing_total),
        "presence_precision": _ratio(presence_tp, presence_tp + presence_fp),
        "presence_recall": _ratio(presence_tp, presence_tp + presence_fn),
        "presence_f1": _ratio(
            2 * presence_tp, 2 * presence_tp + presence_fp + presence_fn
        ),
        "unsupported_predictions": categories["unsupported_prediction"],
        "records_with_unsupported_predictions": len(unsupported_records),
        "error_categories": dict(sorted(categories.items())),
        "annotation_mismatch_count": annotation_mismatch_count,
        "url_preservation": {
            "urls_scored": url_counts["scored"],
            "urls_preserved": url_counts["preserved"],
            "accuracy": _ratio(url_counts["preserved"], url_counts["scored"]),
        },
        "list_item_metrics": {
            "fields": sorted(LIST_FIELDS),
            "matched_items": list_item_counts["tp"],
            "predicted_items": list_item_counts["tp"] + list_item_counts["fp"],
            "gold_items": list_item_counts["tp"] + list_item_counts["fn"],
            "precision": _ratio(list_item_counts["tp"], list_item_counts["tp"] + list_item_counts["fp"]),
            "recall": _ratio(list_item_counts["tp"], list_item_counts["tp"] + list_item_counts["fn"]),
            "f1": _ratio(
                2 * list_item_counts["tp"],
                2 * list_item_counts["tp"] + list_item_counts["fp"] + list_item_counts["fn"],
            ),
        },
    }
    return {
        "summary": summary,
        "fields": field_summary,
        "records": record_result_rows,
        "error_categories": dict(sorted(categories.items())),
        "annotation_mismatches": annotation_mismatch_rows,
    }, field_result_rows + record_result_rows + annotation_mismatch_rows


def _ratio(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def _value_is_present(field, value):
    if value is None:
        return False
    if field in LIST_FIELDS:
        return isinstance(value, (list, tuple, set)) and any(
            isinstance(item, str) and item.strip() for item in value
        )
    text = str(value).strip()
    if field in _UNKNOWN_FIELDS and text.casefold() == "unknown":
        return False
    return bool(text)


def evaluate(annotations_path, snapshot_dir, output_dir, overwrite=False):
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_path}. "
            "Choose a new directory or pass --overwrite to replace its contents."
        )
    annotations = _read_annotations(annotations_path)
    records = []
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    for index, annotation in enumerate(annotations, start=1):
        posting_id = annotation["posting_id"]
        print(f"[{index}/{len(annotations)}] Processing {posting_id}...")
        snapshot_path = Path(snapshot_dir) / f"{posting_id}.json"
        gold = {
            field: annotation[field]
            for field in ANNOTATION_FIELDS
            if annotation[field] is not _UNANNOTATED
        }
        record = {
            "posting_id": posting_id,
            "split": annotation["split"],
            "source_url": annotation["source_url"],
            "retrieval_status": "missing_snapshot",
            "website_retrieval_seconds": None,
            "snapshot_load_seconds": None,
            "inference_seconds": None,
            "total_extraction_seconds": None,
            "model": model_name,
            "token_usage": None,
            "extraction_status": "not_run",
            "error_category": None,
            "error": None,
            "prediction": None,
            "url_preserved": None,
            "annotation_mismatch_fields": annotation["annotation_mismatch_fields"],
            "_gold_scoring": gold,
        }
        if not snapshot_path.exists():
            record.update(
                retrieval_status="missing_snapshot",
                extraction_status="not_run",
                error_category="missing_snapshot",
                error=f"Snapshot not found: {snapshot_path}",
            )
            print(f"  Snapshot missing; skipped.")
            records.append(record)
            continue

        snapshot_load_started = time.perf_counter()
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        record["snapshot_load_seconds"] = round(
            time.perf_counter() - snapshot_load_started, 6
        )
        if (
            snapshot.get("posting_id") != posting_id
            or snapshot.get("split") != annotation["split"]
        ):
            record.update(
                extraction_status="not_run",
                error_category="snapshot_metadata_mismatch",
                error=(
                    "Snapshot posting_id/split does not match the annotation row "
                    f"({posting_id}, {annotation['split']})."
                ),
            )
            print("  Snapshot metadata mismatch; skipped Gemini inference.", file=sys.stderr)
            records.append(record)
            continue
        record["retrieval_status"] = snapshot.get("retrieval_status", "failed")
        record["website_retrieval_seconds"] = snapshot.get("retrieval_seconds")
        record["source_url"] = snapshot.get("source_url") or record["source_url"]
        if snapshot.get("retrieval_status") != "success":
            record.update(
                extraction_status="not_run",
                error_category="website_retrieval_failure",
                error=snapshot.get("retrieval_error", "Website retrieval did not succeed."),
            )
            print(f"  Website retrieval failed; skipped Gemini inference.")
            records.append(record)
            continue

        print(
            "  Running Gemini extraction from saved snapshot "
            f"(website retrieval: {record['website_retrieval_seconds']}s; "
            f"snapshot load: {record['snapshot_load_seconds']:.6f}s)..."
        )
        started = time.perf_counter()
        try:
            posting, token_usage = extract_job_posting_with_metadata(
                snapshot["content"],
                job_url=snapshot.get("source_url"),
                model=model_name,
            )
        except Exception as error:
            record.update(
                inference_seconds=round(time.perf_counter() - started, 6),
                extraction_status="failed",
                error_category="llm_extraction_error",
                error=f"{type(error).__name__}: {error}",
                token_usage=getattr(error, "usage_metadata", None),
            )
            if record["website_retrieval_seconds"] is not None:
                record["total_extraction_seconds"] = round(
                    record["website_retrieval_seconds"] + record["inference_seconds"], 6
                )
            print(f"  Gemini extraction failed: {record['error']}", file=sys.stderr)
            records.append(record)
            continue

        inference_seconds = round(time.perf_counter() - started, 6)
        record.update(
            model=model_name,
            token_usage=token_usage,
            inference_seconds=inference_seconds,
            total_extraction_seconds=round(
                (record["website_retrieval_seconds"] or 0) + inference_seconds, 6
            ),
            extraction_status="success",
            prediction=posting.model_dump(),
        )
        print(
            f"  Done (Gemini inference: {inference_seconds:.3f}s; "
            f"total extraction: {record['total_extraction_seconds']:.3f}s; "
            f"token usage: {token_usage or 'unavailable'})."
        )
        records.append(record)

    score_records = []
    for record in records:
        record["gold"] = record["_gold_scoring"]
        score_records.append(record)
    results, result_rows = _score_predictions(score_records)
    split_results = {}
    for split in sorted({record["split"] for record in score_records}):
        split_score, _ = _score_predictions(
            [record for record in score_records if record["split"] == split]
        )
        split_results[split] = {
            "summary": split_score["summary"],
            "fields": split_score["fields"],
        }
    results["splits"] = split_results
    results["operational"] = {
        "postings": len(records),
        "website_retrieval_failures": sum(
            row["error_category"] == "website_retrieval_failure" for row in records
        ),
        "missing_snapshots": sum(row["error_category"] == "missing_snapshot" for row in records),
        "snapshot_metadata_mismatches": sum(
            row["error_category"] == "snapshot_metadata_mismatch" for row in records
        ),
        "llm_extraction_errors": sum(
            row["error_category"] == "llm_extraction_error" for row in records
        ),
        "successful_extractions": sum(row["extraction_status"] == "success" for row in records),
        "error_categories": {
            category: count
            for category, count in (
                ("website_retrieval_failure", sum(
                    row["error_category"] == "website_retrieval_failure" for row in records
                )),
                ("missing_snapshot", sum(
                    row["error_category"] == "missing_snapshot" for row in records
                )),
                ("snapshot_metadata_mismatch", sum(
                    row["error_category"] == "snapshot_metadata_mismatch" for row in records
                )),
                ("llm_extraction_error", sum(
                    row["error_category"] == "llm_extraction_error" for row in records
                )),
            )
            if count
        },
    }
    for record in records:
        record.pop("_gold_scoring", None)
        record.pop("gold", None)
        record.pop("annotation_mismatch_fields", None)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "predictions.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    prediction_columns = (
        "posting_id", "split", "source_url", "retrieval_status",
        "website_retrieval_seconds", "snapshot_load_seconds", "inference_seconds",
        "total_extraction_seconds", "extraction_status", "model",
        "url_preserved",
        "prompt_token_count", "candidates_token_count", "total_token_count",
        "thoughts_token_count", "cached_content_token_count",
        "error_category", "error",
    )
    with (output_path / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=[*prediction_columns, *(f"pred_{field}" for field in ANNOTATION_FIELDS)]
        )
        writer.writeheader()
        for record in records:
            usage = record["token_usage"] or {}
            row = {
                key: (
                    usage.get(key)
                    if key
                    in {
                        "prompt_token_count",
                        "candidates_token_count",
                        "total_token_count",
                        "thoughts_token_count",
                        "cached_content_token_count",
                    }
                    else record[key]
                )
                for key in prediction_columns
            }
            for field in ANNOTATION_FIELDS:
                value = (record["prediction"] or {}).get(field)
                row[f"pred_{field}"] = (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, list)
                    else value
                )
            writer.writerow(row)
    result_payload = {
        "evaluation_version": ANNOTATION_VERSION,
        **results,
        "result_rows": result_rows,
    }
    (output_path / "evaluation_results.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_path / "evaluation_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        columns = (
            "row_type", "posting_id", "field", "annotated", "correct",
            "error_category", "annotation_mismatch", "url_preserved", "count",
            "formatting_equivalent", "fields_scored",
            "exact_match", "field_accuracy", "present_value_accuracy",
            "genuinely_missing_accuracy", "presence_precision", "presence_recall",
            "presence_f1", "item_precision", "item_recall", "item_f1",
            "url_preservation_accuracy",
        )
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result_rows)
        for field, score in results["fields"].items():
            writer.writerow({
                "row_type": "field_summary",
                "field": field,
                "annotated": score["annotated_fields"],
                "correct": score["exact_correct"],
                "field_accuracy": score["exact_accuracy"],
                "present_value_accuracy": score["present_value_accuracy"],
                "genuinely_missing_accuracy": score["genuinely_missing_accuracy"],
                "presence_precision": score["presence_precision"],
                "presence_recall": score["presence_recall"],
                "presence_f1": score["presence_f1"],
                "item_precision": score.get("items", {}).get("precision"),
                "item_recall": score.get("items", {}).get("recall"),
                "item_f1": score.get("items", {}).get("f1"),
            })
        writer.writerow({
            "row_type": "overall_summary",
            "annotated": results["summary"]["annotated_fields_scored"],
            "correct": results["summary"]["field_exact_correct"],
            "field_accuracy": results["summary"]["field_exact_accuracy"],
            "exact_match": results["summary"]["record_exact_accuracy"],
            "present_value_accuracy": results["summary"]["present_value_accuracy"],
            "genuinely_missing_accuracy": results["summary"]["genuinely_missing_accuracy"],
            "presence_precision": results["summary"]["presence_precision"],
            "presence_recall": results["summary"]["presence_recall"],
            "presence_f1": results["summary"]["presence_f1"],
            "item_precision": results["summary"]["list_item_metrics"]["precision"],
            "item_recall": results["summary"]["list_item_metrics"]["recall"],
            "item_f1": results["summary"]["list_item_metrics"]["f1"],
            "url_preservation_accuracy": results["summary"]["url_preservation"]["accuracy"],
        })
        writer.writerow({
            "row_type": "url_preservation",
            "annotated": results["summary"]["url_preservation"]["urls_scored"],
            "correct": results["summary"]["url_preservation"]["urls_preserved"],
            "url_preservation_accuracy": results["summary"]["url_preservation"]["accuracy"],
        })
        for split, split_score in results["splits"].items():
            for field, score in split_score["fields"].items():
                writer.writerow({
                    "row_type": f"{split}_field_summary",
                    "field": field,
                    "annotated": score["annotated_fields"],
                    "correct": score["exact_correct"],
                    "field_accuracy": score["exact_accuracy"],
                })
            writer.writerow({
                "row_type": "split_summary",
                "field": split,
                "annotated": split_score["summary"]["annotated_fields_scored"],
                "correct": split_score["summary"]["field_exact_correct"],
                "field_accuracy": split_score["summary"]["field_exact_accuracy"],
                "exact_match": split_score["summary"]["record_exact_accuracy"],
                "present_value_accuracy": split_score["summary"]["present_value_accuracy"],
                "genuinely_missing_accuracy": split_score["summary"]["genuinely_missing_accuracy"],
                "presence_precision": split_score["summary"]["presence_precision"],
                "presence_recall": split_score["summary"]["presence_recall"],
                "presence_f1": split_score["summary"]["presence_f1"],
                "item_precision": split_score["summary"]["list_item_metrics"]["precision"],
                "item_recall": split_score["summary"]["list_item_metrics"]["recall"],
                "item_f1": split_score["summary"]["list_item_metrics"]["f1"],
            })
        for category, count in results["error_categories"].items():
            writer.writerow({
                "row_type": "field_error_category",
                "error_category": category,
                "count": count,
            })
        for category, count in results["operational"]["error_categories"].items():
            writer.writerow({
                "row_type": "operational_error_category",
                "error_category": category,
                "count": count,
            })
    print(
        f"Evaluation saved to {output_path} "
        f"(primary field exact accuracy: {results['summary']['field_exact_accuracy']}; "
        f"record exact accuracy: {results['summary']['record_exact_accuracy']})."
    )
    return results


def evaluation_cli(argv=None):
    parser = argparse.ArgumentParser(
        prog="python app.py eval",
        description="Collect local posting snapshots and evaluate Gemini extraction without Google Sheets.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="retrieve and save one public posting snapshot")
    collect.add_argument("--posting-id", required=True)
    collect.add_argument("--split", required=True, choices=SPLITS)
    collect.add_argument("--url", required=True)
    collect.add_argument("--snapshot-dir", default="data/snapshots")
    run = commands.add_parser("run", help="evaluate annotations against saved snapshots")
    run.add_argument("--annotations", required=True)
    run.add_argument("--snapshot-dir", default="data/snapshots")
    run.add_argument("--output-dir", default="data/evaluation")
    run.add_argument(
        "--overwrite",
        action="store_true",
        help="allow writing output files into a non-empty --output-dir",
    )
    args = parser.parse_args(argv)
    if args.command == "collect":
        success = collect_snapshot(args.posting_id, args.split, args.url, args.snapshot_dir)
        if not success:
            raise SystemExit(1)
        return
    evaluate(args.annotations, args.snapshot_dir, args.output_dir, overwrite=args.overwrite)
