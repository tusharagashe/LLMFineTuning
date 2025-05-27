# Define reusable embedded prompt templates for each agent using LangChain PromptTemplate
from langchain.prompts import PromptTemplate

from ._constants import (
    SYSTEM_MESSAGES_COMBINED,
    SYSTEM_MESSAGES_DEFAULT,
    SYSTEM_MESSAGES_LANGFLOW,
    SYSTEM_MESSAGES_NEW_WF,
)

# ------------------------
# Agent 1: Evidence Retriever
# ------------------------
evidence_retriever_prompt = PromptTemplate.from_template(
    """
    You are an expert biomedical evidence retrieval agent.
    Your job is to retrieve past trial examples that relate to the following proposal:

    Mechanism: {mechanism}
    Biomarker: {biomarker}
    Endpoint: {endpoint}
    Indication: {indication}

    Retrieve both successful and failed precedent trials from public sources (FDA approvals, clinicaltrials.gov, etc.)
    that match one or more of these features. Your response should be a list of bullet points with citations when possible.
    """
)

# ------------------------
# Agent 2: Risk Assessor
# ------------------------
risk_assessor_prompt = PromptTemplate.from_template(
    """
    You are an FDA-style regulatory evaluator. Analyze the trial proposal below and assess risk in four key areas:

    Mechanism: {mechanism}
    Biomarker: {biomarker}
    Endpoint: {endpoint}
    Safety considerations: {safety}

    Provide the following structured outputs:
    - A risk level (High, Medium, Low) for each area
    - A brief rationale for each rating
    - A numeric score (1–10) for overall risk severity
    - A freeform paragraph summarizing the total risk profile

    Return your output in a structured dictionary format.
    """
)

# Conditional system messages for risk assessor
risk_assessor_initial_sysmsg = "You are evaluating a new clinical trial proposal. Provide a first-pass risk assessment across mechanism, biomarker, endpoint, and safety."
risk_assessor_revised_sysmsg = "This is a revised proposal with prior feedback. Update your risk evaluation, considering both original design and improvements."


def get_risk_assessor_sysmsg(iteration_count):
    return (
        risk_assessor_initial_sysmsg
        if iteration_count == 0
        else risk_assessor_revised_sysmsg
    )


# ------------------------
# Agent 3: De-risker
# ------------------------
derisker_prompt = PromptTemplate.from_template(
    """
    You are a regulatory strategist tasked with mitigating risks in a clinical trial.
    Use retrieved evidence of successful trials to provide your feedback.

    Original Proposal:
    {user_proposal}

    Risk Assessment Summary:
    {risk_assessment}
    
    Evidence Retrieved:
    {retrieved_evidence}

    For each of the following domains—Mechanism, Biomarker, Endpoint, Safety—
    suggest:
    - A mitigation strategy
    - A rationale (based on precedent or mechanistic logic)
    - An alternative pathway if mitigation is infeasible

    Output your response in a structured format, grouped by domain.
    """
)

# Conditional system messages for de-risker
base_derisker_sysmsg = (
    "You are providing expert feedback to reduce regulatory risk in the proposed trial."
)
revised_derisker_sysmsg = "You are reviewing an updated proposal and prior risk assessment. Suggest new strategies to reduce remaining risk."


def get_derisker_sysmsg(iteration_count):
    return base_derisker_sysmsg if iteration_count == 0 else revised_derisker_sysmsg


# ------------------------
# Agent 4: Format Orchestrator
# ------------------------
format_orchestrator_prompt = PromptTemplate.from_template(
    """
    Create a corporate-level professionally formatted regulatory review and mitigation
    strategy document under 1000 words, based on the following proposal and structured feedback.

    Proposal:
    {user_proposal}

    Final Risk Assessment:
    {risk_assessment}

    Domain-Specific Review:
    - Mechanism: {mechanistic_suggestion}
      Rationale: {mechanistic_rationale}
      Alternative: {mechanistic_alternative}
    - Biomarker: {biomarker_suggestion}
      Rationale: {biomarker_rationale}
      Alternative: {biomarker_alternative}
    - Endpoint: {endpoint_suggestion}
      Rationale: {endpoint_rationale}
      Alternative: {endpoint_alternative}
    - Safety: {safety_suggestion}
      Rationale: {safety_rationale}
      Alternative: {safety_alternative}

    Output a full regulatory review integrating the above points. Use professional, structured writing.
    """
)

# Format orchestrator system message (static)
format_orchestrator_sysmsg = "You are a professional medical reviewer writing for an FDA-style audience. Create a detailed, structured, 800–1000 word regulatory review of the proposal below, integrating the risk assessment and mitigation strategies. Your response should be returned as a single field: 'formatted_review'. Do not include section titles unless relevant. Use complete sentences and professional tone."


# ------------------------
# Global System Message Strategy Loader
# ------------------------


def get_sys_messages(strategy, iteration_count):
    return {
        "evidence_retriever": evidence_retriever_prompt.template,
        "risk_assessor": get_risk_assessor_sysmsg(iteration_count),
        "de_risker": get_derisker_sysmsg(iteration_count),
        "format_orchestrator": format_orchestrator_sysmsg,
    }
    # if strategy == "default"
    # else SYSTEM_MESSAGES_NEW_WF


# Example usage:
# formatted_prompt = risk_assessor_prompt.format(
#     mechanism="Anti-Aβ mAb",
#     biomarker="CSF total tau",
#     endpoint="MMSE over 24 weeks",
#     safety="ARIA risk based on similar mAbs"
# )


# from ._constants import (
#     SYSTEM_MESSAGES_COMBINED,
#     SYSTEM_MESSAGES_DEFAULT,
#     SYSTEM_MESSAGES_LANGFLOW,
#     SYSTEM_MESSAGES_NEW_WF,
# )


# def get_sys_messages(strategy):
#     if strategy == "default":
#         return SYSTEM_MESSAGES_DEFAULT
#     elif strategy == "langflow":
#         return SYSTEM_MESSAGES_LANGFLOW
#     elif strategy == "langflow_combined":
#         return SYSTEM_MESSAGES_COMBINED
#     else:
#         return SYSTEM_MESSAGES_NEW_WF
