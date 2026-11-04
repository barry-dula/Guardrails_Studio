"""Standalone guardrail checks: deterministic (rules) vs model-based (LLM judge)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

DEFAULT_BANNED_KEYWORDS = ["hack", "exploit", "malware", "bomb", "jailbreak", "bypass"]


def keyword_patterns(keywords: list[str]) -> list[re.Pattern[str]]:
    """Match each keyword at a word start, so 'hack' also catches 'hacking'."""
    return [re.compile(rf"\b{re.escape(kw.strip())}", re.IGNORECASE) for kw in keywords if kw.strip()]


def first_match(text: str, patterns: list[re.Pattern[str]]) -> re.Match[str] | None:
    for pattern in patterns:
        if match := pattern.search(text):
            return match
    return None


@dataclass
class GuardResult:
    blocked: bool
    reason: str
    latency_ms: float


def deterministic_check(text: str, keywords: list[str]) -> GuardResult:
    start = time.perf_counter()
    match = first_match(text, keyword_patterns(keywords))
    latency = (time.perf_counter() - start) * 1000
    if match:
        return GuardResult(True, f"Matched banned keyword “{match.group(0)}”", latency)
    return GuardResult(False, "No banned keyword found", latency)


class SafetyVerdict(BaseModel):
    """Structured verdict returned by the LLM safety judge."""

    verdict: Literal["SAFE", "UNSAFE"] = Field(description="Overall safety verdict")
    reason: str = Field(description="One short sentence explaining the verdict")


INPUT_JUDGE_PROMPT = """You are a content-safety classifier for an AI assistant.
Decide whether the user input below is safe to process.
Mark it UNSAFE only if it seeks to cause real harm (e.g. attacking systems, weapons,
self-harm methods, illegal drug production). Educational questions are SAFE.

User input:
{text}"""

OUTPUT_JUDGE_PROMPT = """You are a safety reviewer. Evaluate whether the AI response below is
safe and appropriate to show to an end user. It is UNSAFE if it gives actionable
instructions for harm, leaks secrets or credentials, or contains hateful/abusive content.

AI response:
{text}"""


def judge(model: ChatOpenAI, prompt_template: str, text: str) -> SafetyVerdict:
    structured = model.with_structured_output(SafetyVerdict)
    return structured.invoke([{"role": "user", "content": prompt_template.format(text=text)}])


def model_based_check(model: ChatOpenAI, text: str) -> GuardResult:
    start = time.perf_counter()
    verdict = judge(model, INPUT_JUDGE_PROMPT, text)
    latency = (time.perf_counter() - start) * 1000
    return GuardResult(verdict.verdict == "UNSAFE", verdict.reason, latency)
