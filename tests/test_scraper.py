import json
import unittest
from unittest.mock import patch

import requests

from src.scraper import (
    JobPostingRetrievalError,
    _validated_public_url,
    extract_page_content,
    fetch_job_posting,
)


class ScraperTests(unittest.TestCase):
    def test_extracts_json_ld_and_removes_navigation_and_cookie_content(self):
        html = """
        <html><body>
          <nav>Navigation</nav><div class="cookie-banner">Accept cookies</div>
          <script type="application/ld+json">
            {"@type":"JobPosting","title":"Engineer"}
          </script>
          <div class="job-header">Platform Engineer</div>
          <main><h1>Platform Engineer</h1><p>Build reliable systems.</p></main>
          <footer>Footer links</footer>
        </body></html>
        """

        text, jobs = extract_page_content(html)

        self.assertIn("Platform Engineer", text)
        self.assertNotIn("Navigation", text)
        self.assertNotIn("Accept cookies", text)
        self.assertNotIn("Footer links", text)
        self.assertEqual(jobs, [{"@type": "JobPosting", "title": "Engineer"}])

    def test_supports_json_ld_graph_and_skips_malformed_script(self):
        html = """
        <script type="application/ld+json">{bad json}</script>
        <script type="application/ld+json">
          {"@graph":[{"@type":["Thing","JobPosting"],"title":"Analyst"}]}
        </script>
        <p>Job description</p>
        """

        _, jobs = extract_page_content(html)

        self.assertEqual(jobs[0]["title"], "Analyst")

    def test_removes_nested_irrelevant_elements_without_visiting_decomposed_children(self):
        html = """
        <div class="cookie-consent">
          <section id="cookie-details">
            <span class="popup-content">Cookie preferences</span>
          </section>
        </div>
        <article><p>Clinical Data Scientist job description</p></article>
        """

        text, _ = extract_page_content(html)

        self.assertIn("Clinical Data Scientist", text)
        self.assertNotIn("Cookie preferences", text)

    def test_rejects_non_http_and_private_urls(self):
        for url in (
            "file:///etc/passwd",
            "http://127.0.0.1/job",
            "http://192.168.1.10/job",
            "http://localhost/job",
        ):
            with self.subTest(url=url), self.assertRaises(JobPostingRetrievalError):
                _validated_public_url(url)

    def test_preserves_job_header_and_accepts_public_ipv6_literal(self):
        text, _ = extract_page_content('<div class="job-header">Engineer</div>')

        self.assertIn("Engineer", text)
        self.assertEqual(
            _validated_public_url("https://[2606:4700:4700::1111]/jobs"),
            "https://[2606:4700:4700::1111]/jobs",
        )

    @patch("src.scraper.socket.getaddrinfo")
    def test_rejects_hostname_resolving_to_private_address(self, resolve):
        resolve.return_value = [(None, None, None, None, ("10.0.0.4", 0))]

        with self.assertRaises(JobPostingRetrievalError):
            _validated_public_url("https://jobs.example/jobs/1")

    @patch("src.scraper.socket.getaddrinfo")
    def test_fetch_uses_timeout_and_blocks_private_redirect(self, resolve):
        resolve.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
        session = requests.Session()
        response = requests.Response()
        response.status_code = 302
        response.headers["Location"] = "http://127.0.0.1/private"
        with patch.object(session, "get", return_value=response) as get:
            with self.assertRaises(JobPostingRetrievalError):
                fetch_job_posting("https://jobs.example/posting", session=session)

        self.assertEqual(get.call_args.kwargs["timeout"], (5, 15))
        self.assertFalse(get.call_args.kwargs["allow_redirects"])

    @patch("src.scraper.socket.getaddrinfo")
    def test_fetch_reports_access_denial_without_retrying(self, resolve):
        resolve.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
        session = requests.Session()
        response = requests.Response()
        response.status_code = 403
        with patch.object(session, "get", return_value=response) as get:
            with self.assertRaisesRegex(JobPostingRetrievalError, "access protections are not bypassed"):
                fetch_job_posting("https://jobs.example/posting", session=session)
        get.assert_called_once()

    def test_page_without_readable_content_fails(self):
        with self.assertRaises(JobPostingRetrievalError):
            extract_page_content("<script>not a posting</script>")


if __name__ == "__main__":
    unittest.main()
