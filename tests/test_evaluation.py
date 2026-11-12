import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel

from src.evaluation import (
    ANNOTATION_COLUMNS,
    ANNOTATION_FIELDS,
    LIST_FIELDS,
    _score_predictions,
    _read_annotations,
    evaluate,
    normalise_field,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_equivalent_salary_currency_period_and_date_formatting(self):
        self.assertEqual(
            normalise_field("salary_amount", "USD 60k–80k"),
            normalise_field("salary_amount", "60,000 to 80,000"),
        )
        self.assertEqual(normalise_field("salary_currency", "€"), "EUR")
        self.assertEqual(
            normalise_field("salary_period", "per annum"),
            normalise_field("salary_period", "Yearly"),
        )
        self.assertEqual(
            normalise_field("application_deadline", "November 12, 2026"),
            normalise_field("application_deadline", "2026-11-12"),
        )

    def test_unknown_categories_are_equivalent_to_confirmed_missing_values(self):
        self.assertIsNone(normalise_field("work_mode", "Unknown"))
        self.assertIsNone(normalise_field("employment_type", "unknown"))

    def test_presence_metrics_count_wrong_nonempty_values_as_present(self):
        prediction = {field: None for field in ANNOTATION_FIELDS}
        prediction.update(
            job_title="Data Scientist",
            salary_amount="100,000",
            work_mode="Unknown",
        )
        record = {
            "posting_id": "job-01",
            "extraction_status": "success",
            "prediction": prediction,
            "gold": {
                "job_title": "Senior Data Scientist",
                "salary_amount": None,
                "work_mode": None,
            },
        }

        results, rows = _score_predictions([record])

        self.assertEqual(results["summary"]["annotated_fields_scored"], 3)
        self.assertEqual(results["summary"]["field_exact_correct"], 1)
        self.assertEqual(results["summary"]["record_exact_matches"], 0)
        self.assertEqual(results["summary"]["unsupported_predictions"], 1)
        self.assertEqual(results["summary"]["presence_precision"], 0.5)
        self.assertEqual(results["summary"]["presence_recall"], 1.0)
        self.assertEqual(results["summary"]["presence_f1"], 0.666667)
        self.assertEqual(results["summary"]["present_value_accuracy"], 0.0)
        self.assertEqual(results["summary"]["genuinely_missing_accuracy"], 0.5)
        self.assertEqual(
            results["error_categories"],
            {"partial_value_mismatch": 1, "unsupported_prediction": 1},
        )
        self.assertEqual(
            {row["error_category"] for row in rows if row["row_type"] == "field"},
            {"", "unsupported_prediction", "partial_value_mismatch"},
        )

    def test_list_items_use_deterministic_exact_matching(self):
        prediction = {field: None for field in ANNOTATION_FIELDS}
        prediction.update(
            compensation_details=[
                "competitive salary",
                "yearly bonus",
                "gym membership",
            ],
            other_benefits=[
                "wellbeing initiatives",
                "training allowance",
            ],
            job_url="https://example.test/job",
        )
        gold = {
            "compensation_details": [
                "Competitive salary",
                "Yearly bonus",
                "premium pension",
            ],
            "other_benefits": [
                "wellbeing initiatives",
                "professional development",
            ],
            "job_url": "https://wrong.example/job",
        }
        record = {
            "posting_id": "job-list",
            "source_url": "https://example.test/job",
            "extraction_status": "success",
            "prediction": prediction,
            "gold": gold,
            "annotation_mismatch_fields": ["compensation_details"],
        }

        results, rows = _score_predictions([record])
        summary = results["summary"]

        self.assertEqual(summary["annotated_fields_scored"], 2)
        self.assertEqual(summary["record_exact_accuracy"], 0.0)
        self.assertEqual(summary["url_preservation"], {
            "urls_scored": 1,
            "urls_preserved": 1,
            "accuracy": 1.0,
        })
        self.assertEqual(summary["list_item_metrics"]["fields"], sorted(LIST_FIELDS))
        self.assertEqual(summary["list_item_metrics"]["matched_items"], 3)
        self.assertEqual(summary["list_item_metrics"]["predicted_items"], 5)
        self.assertEqual(summary["list_item_metrics"]["gold_items"], 5)
        self.assertEqual(summary["list_item_metrics"]["precision"], 0.6)
        self.assertEqual(summary["list_item_metrics"]["recall"], 0.6)
        self.assertEqual(summary["list_item_metrics"]["f1"], 0.6)
        self.assertEqual(summary["annotation_mismatch_count"], 1)
        self.assertEqual(
            [row["field"] for row in rows if row["row_type"] == "annotation_mismatch"],
            ["compensation_details"],
        )
        self.assertEqual(
            results["fields"]["compensation_details"]["presence_precision"], 1.0
        )
        self.assertEqual(
            normalise_field("other_benefits", [" Wellness initiative ", "WELLNESS INITIATIVE"]),
            frozenset({"wellness initiative"}),
        )
        self.assertEqual(
            normalise_field("compensation_details", ["pension", "bonus"]),
            normalise_field("compensation_details", ["bonus", "pension"]),
        )

    def test_error_categories_distinguish_unsupported_wrong_and_omitted_values(self):
        prediction = {field: None for field in ANNOTATION_FIELDS}
        prediction.update(
            company_name="Unrelated Co",
            job_title="Engineer",
            work_mode="Hybrid",
        )
        record = {
            "posting_id": "job-errors",
            "extraction_status": "success",
            "prediction": prediction,
            "gold": {
                "company_name": "Danone",
                "job_title": "Clinical Data Scientist",
                "location": "Utrecht",
                "work_mode": None,
            },
        }

        results, _ = _score_predictions([record])

        self.assertEqual(results["error_categories"], {
            "incorrect_value": 2,
            "omission": 1,
            "unsupported_prediction": 1,
        })

    def test_job_url_is_excluded_from_primary_record_accuracy(self):
        prediction = {field: None for field in ANNOTATION_FIELDS}
        prediction.update(
            job_title="Engineer",
            job_url="https://example.test/jobs/1",
        )
        results, _ = _score_predictions([{
            "posting_id": "job-url",
            "source_url": "https://example.test/jobs/1",
            "extraction_status": "success",
            "prediction": prediction,
            "gold": {
                "job_title": "Engineer",
                "job_url": "https://wrong.example/jobs/1",
            },
        }])

        self.assertEqual(results["summary"]["annotated_fields_scored"], 1)
        self.assertEqual(results["summary"]["record_exact_accuracy"], 1.0)
        self.assertEqual(results["summary"]["url_preservation"]["accuracy"], 1.0)


    def test_unannotated_fields_are_not_scored(self):
        results, rows = _score_predictions([{
            "posting_id": "job-02",
            "extraction_status": "success",
            "prediction": {"job_title": "Engineer"},
            "gold": {},
        }])

        self.assertEqual(results["summary"]["annotated_fields_scored"], 0)
        self.assertIsNone(results["summary"]["field_exact_accuracy"])
        self.assertEqual(rows, [])

    def test_v2_annotation_lists_and_manual_mismatch_flags_parse_as_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "annotations.csv"
            row = {
                "annotation_version": "2",
                "posting_id": "job-v2",
                "split": "pilot",
                "source_url": "https://example.test/job",
                "compensation_details": '["Competitive salary","Yearly bonus"]',
                "other_benefits": "__MISSING__",
                "annotation_mismatch_fields": '["compensation_details"]',
            }
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=ANNOTATION_COLUMNS)
                writer.writeheader()
                writer.writerow(row)

            parsed = _read_annotations(path)[0]

            self.assertEqual(
                parsed["compensation_details"],
                ["Competitive salary", "Yearly bonus"],
            )
            self.assertIsNone(parsed["other_benefits"])
            self.assertEqual(
                parsed["annotation_mismatch_fields"], ["compensation_details"]
            )

    def test_existing_results_are_not_overwritten_without_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "history"
            output.mkdir()
            preserved = output / "evaluation_results.json"
            preserved.write_text('{"historical": true}\n', encoding="utf-8")

            with self.assertRaises(FileExistsError):
                evaluate(root / "not-read.csv", root / "snapshots", output)

            self.assertEqual(
                preserved.read_text(encoding="utf-8"), '{"historical": true}\n'
            )

    @patch("src.evaluation.extract_job_posting_with_metadata")
    def test_local_snapshots_keep_retrieval_and_llm_errors_distinct(
        self, extract_job_posting_with_metadata
    ):
        class FakePosting(BaseModel):
            job_title: str | None = "Engineer"

        def fake_extract(content, **kwargs):
            if content == "LLM failure fixture":
                raise RuntimeError("model unavailable")
            return FakePosting(), {
                "prompt_token_count": 100,
                "candidates_token_count": 20,
                "total_token_count": 120,
                "thoughts_token_count": 8,
            }

        extract_job_posting_with_metadata.side_effect = fake_extract
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "annotations.csv"
            snapshots = root / "snapshots"
            snapshots.mkdir()
            output = root / "results"
            rows = [
                {"annotation_version": "2", "posting_id": "job-ok", "split": "development",
                 "source_url": "https://example.test/ok", "job_title": "Engineer"},
                {"annotation_version": "2", "posting_id": "job-retrieval", "split": "test",
                 "source_url": "https://example.test/denied", "job_title": "__MISSING__"},
                {"annotation_version": "2", "posting_id": "job-llm", "split": "development",
                 "source_url": "https://example.test/llm", "job_title": "Scientist"},
            ]
            with annotations.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=ANNOTATION_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            fixtures = {
                "job-ok": {
                    "posting_id": "job-ok", "split": "development",
                    "retrieval_status": "success", "retrieval_seconds": 0.25,
                    "source_url": "https://example.test/ok", "content": "Posting fixture",
                },
                "job-retrieval": {
                    "posting_id": "job-retrieval", "split": "test",
                    "retrieval_status": "failed", "retrieval_seconds": 0.1,
                    "retrieval_error": "HTTP 403",
                },
                "job-llm": {
                    "posting_id": "job-llm", "split": "development",
                    "retrieval_status": "success", "retrieval_seconds": 0.2,
                    "source_url": "https://example.test/llm", "content": "LLM failure fixture",
                },
            }
            for posting_id, fixture in fixtures.items():
                (snapshots / f"{posting_id}.json").write_text(
                    json.dumps(fixture), encoding="utf-8"
                )

            results = evaluate(annotations, snapshots, output)

            self.assertEqual(results["operational"]["website_retrieval_failures"], 1)
            self.assertEqual(results["operational"]["llm_extraction_errors"], 1)
            self.assertEqual(results["operational"]["successful_extractions"], 1)
            self.assertEqual(
                results["splits"]["development"]["summary"]["records_scored"], 1
            )
            predictions = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
            by_id = {record["posting_id"]: record for record in predictions}
            self.assertEqual(by_id["job-retrieval"]["error_category"], "website_retrieval_failure")
            self.assertEqual(by_id["job-llm"]["error_category"], "llm_extraction_error")
            self.assertEqual(by_id["job-ok"]["token_usage"]["thoughts_token_count"], 8)
            self.assertIsNotNone(by_id["job-ok"]["snapshot_load_seconds"])
            self.assertEqual(by_id["job-ok"]["website_retrieval_seconds"], 0.25)
            self.assertTrue((output / "predictions.csv").exists())
            self.assertTrue((output / "evaluation_results.csv").exists())
            self.assertTrue((output / "evaluation_results.json").exists())
            self.assertEqual(extract_job_posting_with_metadata.call_count, 2)


if __name__ == "__main__":
    unittest.main()
