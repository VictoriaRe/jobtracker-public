"""Public job-posting retrieval and HTML content extraction."""

import ipaddress
import json
import socket
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

REQUEST_TIMEOUT = (5, 15)
MAX_REDIRECTS = 5
USER_AGENT = "JobPilot job-posting reader (public pages only)"
_REMOVABLE_TAGS = (
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "form",
    "nav",
    "footer",
    "header",
    "aside",
)
_IRRELEVANT_HINTS = (
    "cookie",
    "navigation",
    "navbar",
    "footer",
    "menu",
    "breadcrumb",
    "popup",
    "modal",
    "consent",
)


class JobPostingRetrievalError(Exception):
    """Raised when a public posting cannot safely be retrieved."""


def _validated_public_url(url):
    if not isinstance(url, str):
        raise JobPostingRetrievalError("URL must be a string.")
    try:
        parsed = urlsplit(url.strip())
        port = parsed.port
    except ValueError as error:
        raise JobPostingRetrievalError("Invalid job posting URL.") from error

    if (
        parsed.scheme.lower() not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise JobPostingRetrievalError("Only public HTTP or HTTPS URLs are allowed.")

    hostname = parsed.hostname.rstrip(".")
    if hostname.lower() == "localhost" or hostname.lower().endswith(".localhost"):
        raise JobPostingRetrievalError("Localhost URLs are not allowed.")
    try:
        addresses = {ipaddress.ip_address(hostname)}
    except ValueError:
        try:
            addresses = {
                ipaddress.ip_address(result[4][0])
                for result in socket.getaddrinfo(hostname, port or (443 if parsed.scheme.lower() == "https" else 80))
            }
        except (OSError, ValueError) as error:
            raise JobPostingRetrievalError(
                f"Could not resolve the public posting host: {hostname}."
            ) from error
    if not addresses or any(not address.is_global for address in addresses):
        raise JobPostingRetrievalError("Private or non-public network addresses are not allowed.")

    netloc = hostname
    if ":" in hostname:
        netloc = f"[{hostname}]"
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit(
        (parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, "")
    )


def _is_irrelevant(element):
    hints = " ".join([element.get("id") or "", *(element.get("class") or [])]).casefold()
    return any(hint in hints for hint in _IRRELEVANT_HINTS)


def _json_ld_job_postings(soup):
    jobs = []

    def collect(value):
        if isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            item_type = value.get("@type", [])
            if isinstance(item_type, str):
                item_type = [item_type]
            if "JobPosting" in item_type:
                jobs.append(value)
            for key, child in value.items():
                if key in ("@graph", "mainEntity"):
                    collect(child)

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            collect(json.loads(script.string or script.get_text()))
        except (json.JSONDecodeError, TypeError):
            continue
    return jobs


def extract_page_content(html):
    """Return useful page text plus any valid JobPosting JSON-LD objects."""
    soup = BeautifulSoup(html, "html.parser")
    job_postings = _json_ld_job_postings(soup)

    elements_to_remove = [
        element
        for element in soup.find_all(True)
        if element.name in _REMOVABLE_TAGS or _is_irrelevant(element)
    ]
    removal_ids = {id(element) for element in elements_to_remove}
    top_level_removals = [
        element
        for element in elements_to_remove
        if not any(id(ancestor) in removal_ids for ancestor in element.parents)
    ]
    for element in top_level_removals:
        element.decompose()

    text = " ".join(soup.stripped_strings)
    if job_postings:
        return text, job_postings
    if not text:
        raise JobPostingRetrievalError("The page did not contain readable job-posting content.")
    return text, []


def fetch_job_posting(url, session=None):
    """Fetch and parse a public posting, validating every redirect destination."""
    current_url = _validated_public_url(url)
    http = session or requests.Session()
    http.trust_env = False

    for redirect_count in range(MAX_REDIRECTS + 1):
        try:
            response = http.get(
                current_url,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
                allow_redirects=False,
            )
        except requests.RequestException as error:
            raise JobPostingRetrievalError(f"Unable to retrieve job posting: {error}") from error

        if response.is_redirect or response.is_permanent_redirect:
            if redirect_count == MAX_REDIRECTS:
                raise JobPostingRetrievalError("The page redirected too many times.")
            location = response.headers.get("Location")
            if not location:
                raise JobPostingRetrievalError("The page returned a redirect without a destination.")
            current_url = _validated_public_url(urljoin(current_url, location))
            continue

        if response.status_code in (401, 403, 429):
            raise JobPostingRetrievalError(
                f"The website denied access (HTTP {response.status_code}); access protections are not bypassed."
            )
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            raise JobPostingRetrievalError(
                f"The website returned HTTP {response.status_code}."
            ) from error

        content, job_postings = extract_page_content(response.text)
        if job_postings:
            structured_data = json.dumps(job_postings, ensure_ascii=False)
            return f"JobPosting JSON-LD:\n{structured_data}\n\nPage text:\n{content}"
        return content

    raise JobPostingRetrievalError("The page could not be retrieved.")
