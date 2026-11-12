import json
import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from src.extractor import (
    JobPostingExtractionError,
    extract_job_posting,
    extract_job_posting_with_metadata,
)
from src.schemas import JobPosting


class FakeClient:
    def __init__(self, response_text, usage_metadata=None):
        response = {"text": response_text}
        if usage_metadata is not None:
            response["usage_metadata"] = usage_metadata
        self.models = SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(**response)
        )


class ExtractorTests(unittest.TestCase):
    def test_schema_defaults_unknown_fields_to_null_or_unknown(self):
        posting = JobPosting(job_title="Engineer")

        self.assertIsNone(posting.company_name)
        self.assertIsNone(posting.salary_amount)
        self.assertIsNone(posting.compensation_details)
        self.assertIsNone(posting.other_benefits)
        self.assertEqual(posting.work_mode, "Unknown")
        self.assertEqual(posting.contract_duration, "unknown")
        self.assertEqual(posting.employment_type, "unknown")

    def test_extracts_validated_data_and_uses_supplied_url(self):
        client = FakeClient(json.dumps({
            "company_name": "Example Co",
            "job_title": "Engineer",
            "work_mode": "Remote",
        }))

        posting = extract_job_posting(
            "Example Co is hiring an Engineer, remote.",
            job_url="https://example.com/jobs/1",
            client=client,
            model="test-model",
        )

        self.assertEqual(posting.company_name, "Example Co")
        self.assertEqual(posting.work_mode, "Remote")
        self.assertEqual(posting.job_url, "https://example.com/jobs/1")

    def test_extracts_compensation_lists_and_gemini_usage_metadata(self):
        client = FakeClient(
            json.dumps({
                "compensation_details": ["Competitive salary", "Yearly bonus"],
                "other_benefits": ["Wellbeing initiatives", "Training"],
            }),
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=40,
                total_token_count=140,
                thoughts_token_count=12,
            ),
        )

        posting, usage = extract_job_posting_with_metadata(
            "Posting content", client=client, model="test-model"
        )

        self.assertEqual(posting.compensation_details, ["Competitive salary", "Yearly bonus"])
        self.assertEqual(posting.other_benefits, ["Wellbeing initiatives", "Training"])
        self.assertEqual(usage["prompt_token_count"], 100)
        self.assertEqual(usage["candidates_token_count"], 40)
        self.assertEqual(usage["total_token_count"], 140)
        self.assertEqual(usage["thoughts_token_count"], 12)

    def test_invalid_structured_output_retains_available_usage_metadata(self):
        client = FakeClient(
            json.dumps({"unexpected": "field"}),
            usage_metadata=SimpleNamespace(prompt_token_count=12, thoughts_token_count=3),
        )

        with self.assertRaises(JobPostingExtractionError) as raised:
            extract_job_posting_with_metadata(
                "Posting content", client=client, model="test-model"
            )

        self.assertEqual(
            raised.exception.usage_metadata,
            {"prompt_token_count": 12, "thoughts_token_count": 3},
        )

    def test_invalid_work_mode_is_reported_as_structured_output_error(self):
        client = FakeClient(json.dumps({"work_mode": "Mostly remote"}))

        with self.assertRaises(JobPostingExtractionError):
            extract_job_posting("Posting content", client=client)

    def test_rejects_empty_posting_input(self):
        with self.assertRaises(ValueError):
            extract_job_posting("  ", client=FakeClient("{}"))

    def test_schema_rejects_unexpected_fields(self):
        with self.assertRaises(ValidationError):
            JobPosting.model_validate({"unexpected": "value"})


if __name__ == "__main__":
    unittest.main()
