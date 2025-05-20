from pydantic import BaseModel, Field
from typing_extensions import List, Literal, Optional, TypedDict

# # Graph state
# class State(TypedDict):
#     retrieved_evidence: str
#     user_proposal: str
#     risk_assessment_and_rating: str
#     proposal_feedback: str
#     improved_proposal: str
#     pass_or_fail: str
#     # iteration_counts: int = 0


class State(TypedDict):
    # === Input Fields ===
    user_proposal: str

    # === Agent 1: Evidence Retriever ===
    retrieved_evidence: Optional[
        List[str]
    ]  # useful for future vector DB or NCT linking

    # === Agent 2: Risk Assessor ===
    risk_assessment_and_rating: Optional[str]  # freeform LLM summary
    risk_assessment_yaml: Optional[str]  # structured YAML with 4-domain severity

    # === Agent 3: Regulatory Critiquer ===
    proposal_feedback: Optional[str]  # bulleted critique / per-domain advice
    risk_mitigation_yaml: Optional[str]  # optional if generated later

    # === Agent 4: Proposal Writer ===
    improved_proposal: Optional[str]

    # === Evaluation Output ===
    pass_or_fail: Optional[str]  # "Pass", "Fail", or a score bucket
    rating_score: Optional[int]  # 1–10 numeric summary
    iteration_count: Optional[int]  # for tracking reruns


# # Schema for structured output to use in evaluation
# class Feedback(BaseModel):
#     grade: Literal["pass", "fail"] = Field(
#         description="Decide if the proposal passes and will be successful going through FDA review.",
#     )
#     feedback: str = Field(
#         description="If the proposal fails, provide the current rating and risk assessment to improve.",
#     )


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
