from pydantic import BaseModel, Field
from typing_extensions import List, Literal, Optional, TypedDict


class State(TypedDict):
    # === Input Fields ===
    user_proposal: str
    mechanism: str
    biomarker: str
    endpoint: str
    indication: str
    safety: str
    iteration_count: int = 0

    # evidence retriever function
    retrieved_evidence: Optional[List[str]] = None
    drug: str
    api_messages: Optional[List[str]] = None

    # risk assessor agent
    mechanistic_risk_ranking: Optional[Literal["high", "medium", "low"]] = None
    biomarker_risk_ranking: Optional[Literal["high", "medium", "low"]] = None
    endpoint_risk_ranking: Optional[Literal["high", "medium", "low"]] = None
    safety_risk_ranking: Optional[Literal["high", "medium", "low"]] = None
    summary_rating: Optional[int] = Field(
        None, ge=1, le=10, description="A summary rating between 1 and 10."
    )
    risk_assessment: Optional[str] = Field(
        None,
        description="A free-form text summary of the proposals weakness and risk of failing FDA approval",
    )

    # de-risker agent
    mechanistic_suggestion: Optional[str] = None
    mechanistic_rationale: Optional[str] = None
    mechanistic_alternative: Optional[str] = None
    biomarker_suggestion: Optional[str] = None
    biomarker_rationale: Optional[str] = None
    biomarker_alternative: Optional[str] = None
    endpoint_suggestion: Optional[str] = None
    endpoint_rationale: Optional[str] = None
    endpoint_alternative: Optional[str] = None
    safety_suggestion: Optional[str] = None
    safety_rationale: Optional[str] = None
    safety_alternative: Optional[str] = None
    # overall_summary_of_suggestions: Optional[str] = None

    # History tracking (across iterations)
    mechanistic_suggestion_history: List[str] = []
    biomarker_suggestion_history: List[str] = []
    endpoint_suggestion_history: List[str] = []
    safety_suggestion_history: List[str] = []

    # format orchestrator agent
    final_review_paragraph: Optional[str] = Field(
        None,
        description="A free-form text final review document of all changes needed to be made.",
    )


# ------------------------
# Feedback Models for Structured Output
# ------------------------
class RiskAssessmentFeedback(BaseModel):
    risk_assessment: str
    mechanistic_risk_ranking: Literal["high", "medium", "low"]
    biomarker_risk_ranking: Literal["high", "medium", "low"]
    endpoint_risk_ranking: Literal["high", "medium", "low"]
    safety_risk_ranking: Literal["high", "medium", "low"]
    summary_rating: int = Field(..., ge=1, le=10)


class DeRiskerFeedback(BaseModel):
    mechanistic_suggestion: str
    mechanistic_rationale: str
    mechanistic_alternative: str
    biomarker_suggestion: str
    biomarker_rationale: str
    biomarker_alternative: str
    endpoint_suggestion: str
    endpoint_rationale: str
    endpoint_alternative: str
    safety_suggestion: str
    safety_rationale: str
    safety_alternative: str
    # overall_summary_of_suggestions: str


class FormatOutput(BaseModel):
    final_review_paragraph: str


# class Feedback(BaseModel):
#     grade: Literal["pass", "fail"] = Field(
#         description="Whether this proposal is ready for submission or needs another iteration."
#     )
#     rating: int = Field(
#         description="A numeric score (1–10) summarizing the quality and de-risked strength of the proposal."
#     )
#     feedback: str = Field(
#         description="Short summary of rationale behind the grade and how it relates to FDA risk domains (mechanistic, biomarker, endpoint, safety)."
#     )
