"""Validated structured data for job posting extraction."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class JobPosting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: str | None = Field(
        default=None, description="Company name, or null if not stated."
    )
    job_title: str | None = Field(
        default=None, description="Job title, or null if not stated."
    )
    location: str | None = Field(
        default=None, description="Work location, or null if not stated."
    )
    work_mode: Literal["Remote", "Hybrid", "On-site", "Unknown"] = "Unknown"
    salary_amount: str | None = Field(
        default=None,
        description="Numerical salary amount or range only, preserving the stated numbers.",
    )
    salary_currency: str | None = Field(
        default=None, description="Salary currency code or symbol, or null if unstated."
    )
    salary_period: str | None = Field(
        default=None, description="Salary pay period, or null if unstated."
    )
    compensation_details: list[str] | None = Field(
        default=None,
        description="List of explicitly stated salary, bonus, pension, and leave details.",
    )
    other_benefits: list[str] | None = Field(
        default=None,
        description=(
            "List of explicitly stated wellbeing, professional development, "
            "and other non-financial perks."
        ),
    )
    application_deadline: str | None = Field(
        default=None, description="Application deadline as stated, or null."
    )
    contract_duration: Literal[
        "permanent", "temporary", "fixed-term", "unknown"
    ] = Field(
        default="unknown",
        description="Contract duration category; separate from employment hours.",
    )
    employment_type: Literal["full-time", "part-time", "unknown"] = Field(
        default="unknown",
        description="Employment hours category, separate from contract duration.",
    )
    job_url: str | None = Field(
        default=None, description="Public job posting URL, or null if unknown."
    )
