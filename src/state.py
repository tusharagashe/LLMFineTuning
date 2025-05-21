from pydantic import BaseModel, Field
from typing_extensions import List, Literal, Optional, TypedDict


class State(TypedDict):
    # === Input Fields ===
    user_proposal: str

    retrieved_evidence: Optional[
        List[str]
    ]  # useful for future vector DB or NCT linking

    risk_assessment_and_rating: Optional[str]  # freeform LLM summary
    # risk_assessment_yaml: Optional[str]  # TODO:structured YAML with 4-domain severity

    proposal_feedback: Optional[str]  # bulleted critique / per-domain advice
    # risk_mitigation_yaml: Optional[str]  # TODO:optional if generated later

    improved_proposal: Optional[str]

    pass_or_fail: Optional[str]  # "Pass", "Fail", or a score bucket
    rating_score: Optional[int]  # 1–10 numeric summary
    iteration_count: Optional[int]  # for tracking reruns


class Feedback(BaseModel):
    grade: Literal["pass", "fail"] = Field(
        description="Whether this proposal is ready for submission or needs another iteration."
    )
    rating: int = Field(
        description="A numeric score (1–10) summarizing the quality and de-risked strength of the proposal."
    )
    feedback: str = Field(
        description="Short summary of rationale behind the grade and how it relates to FDA risk domains (mechanistic, biomarker, endpoint, safety)."
    )
