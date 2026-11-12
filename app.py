"""JobPilot MVP: interpret and propose job-application additions."""
import json
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build
from google import genai
from google.oauth2 import service_account

from src.extractor import extract_job_posting
from src.scraper import fetch_job_posting
from src.evaluation import evaluation_cli

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
FIELDS = ["Company Name", "Application Status", "Role", "Salary", "Date Submitted", "Link to Job Req", "Rejection Reason", "Notes"]


def sheets_service():
    credentials = service_account.Credentials.from_service_account_file(
        "service-account.json",
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return build("sheets", "v4", credentials=credentials)

def interpret(message, headers):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = f"""
    Extract one job application record from the user's text.

    Return ONLY a valid JSON object with these keys:
    company, role, status, date, url, notes.

    Use empty strings for missing information.
    Do not invent dates, links, or application statuses.
    Today's date is {date.today().isoformat()}.
    Interpret 'today' accordingly.
    Only set status to Applied if the user explicitly says they applied.

    User's message:
    {message}
    """

    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

    return json.loads(response.text)

def norm(s):
    return " ".join(str(s).casefold().split())


def extract_job_posting_flow(url):
    try:
        content = fetch_job_posting(url)
    except Exception as error:
        print(
            "Automatic extraction of page content failed "
            f"({type(error).__name__}): {error}"
        )
        content = input("Paste the job description (or press Enter to cancel): ").strip()
        if not content:
            print("No job description provided; nothing was extracted.")
            return

    job = extract_job_posting(content, job_url=url)
    print("\nExtracted job posting (review only; nothing was written to Google Sheets):")
    print(json.dumps(job.model_dump(), indent=2, ensure_ascii=False))


def main():
    load_dotenv()
    if len(sys.argv) >= 2 and sys.argv[1] == "eval":
        evaluation_cli(sys.argv[2:])
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "extract":
        if len(sys.argv) != 3:
            raise SystemExit("Usage: python app.py extract <public-job-posting-url>")
        extract_job_posting_flow(sys.argv[2])
        return

    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    if not sheet_id:
        raise ValueError("Set GOOGLE_SHEET_ID in .env")
    service = sheets_service()
    metadata = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
    title = metadata["sheets"][0]["properties"]["title"]
    quoted = "'" + title.replace("'", "''") + "'"
    values = service.spreadsheets().values().get(spreadsheetId=sheet_id, range=f"{quoted}!A1:AZ1000").execute().get("values", [])
    if not values:
        raise ValueError("First worksheet is empty; please add a header row first.")
    headers = values[0]
    print("Connected to:", title)
    print("Headers:", ", ".join(headers))
    message = input("Describe your job application: ").strip()
    if not message:
        return
    proposal = interpret(message, headers)
    print("\nParsed proposal:\n", json.dumps(proposal, indent=2, ensure_ascii=False))
    company_col = next((i for i, h in enumerate(headers) if norm(h) == norm("Company Name")), None)
    role_col = next((i for i, h in enumerate(headers) if norm(h) == norm("Role")), None)
    if company_col is None or role_col is None:
        raise ValueError("Expected 'Company Name' and 'Role' headers; adapt mappings before writing.")
    duplicates = [i + 2 for i, row in enumerate(values[1:]) if
                  company_col < len(row) and role_col < len(row) and
                  norm(row[company_col]) == norm(proposal.get("company", "")) and
                  norm(row[role_col]) == norm(proposal.get("role", ""))]
    if duplicates:
        print("Potential duplicate row(s):", duplicates)
        print("No changes made. Updating existing entries will be implemented next.")
        return
    mapping = {
        "Company Name": proposal.get("company", ""),
        "Application Status": proposal.get("status", ""),
        "Role": proposal.get("role", ""),
        "Date Submitted": proposal.get("date", ""),
        "Link to Job Req": proposal.get("url", ""),
        "Notes": proposal.get("notes", ""),
    }
    if not proposal.get("company") or not proposal.get("role"):
        print("Company or role missing. No changes made.")
        return
    row = [mapping.get(header, "") for header in headers]
    print("\nNew row preview:", row)
    if input("Add this row to Google Sheets? Type YES: ").strip() != "YES":
        print("Cancelled; no changes made.")
        return
    service.spreadsheets().values().append(
        spreadsheetId=sheet_id, range=f"{quoted}!A:AZ",
        valueInputOption="USER_ENTERED", insertDataOption="INSERT_ROWS",
        body={"values": [row]},
    ).execute()
    print("Added successfully.")


if __name__ == "__main__":
    main()
