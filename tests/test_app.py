import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.schemas import JobPosting

from app import extract_job_posting_flow


class ExtractionFallbackTests(unittest.TestCase):
    @patch("app.extract_job_posting", return_value=JobPosting(job_title="Data Scientist"))
    @patch("builtins.input", return_value="Pasted job description")
    @patch("app.fetch_job_posting", side_effect=RuntimeError("unexpected parser failure"))
    def test_unexpected_scraper_failure_offers_paste_fallback(
        self, fetch, paste_input, extract
    ):
        output = io.StringIO()
        with redirect_stdout(output):
            extract_job_posting_flow("https://jobs.example/posting")

        fetch.assert_called_once()
        paste_input.assert_called_once()
        extract.assert_called_once_with(
            "Pasted job description", job_url="https://jobs.example/posting"
        )
        self.assertIn(
            "(RuntimeError): unexpected parser failure", output.getvalue()
        )


if __name__ == "__main__":
    unittest.main()
