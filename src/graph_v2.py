import os

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from ._constants import LLM_CONFIGS, MAX_ITERATIONS
from .apitools import (
    fetch_clinical_trials,
    fetch_fda_adverse_events,
    fetch_opentargets_associations,
    fetch_trials_in_literature,
)
from .prompts import (
    derisker_prompt,
    evidence_retriever_prompt,
    format_orchestrator_prompt,
    get_sys_messages,
    llm_api_tool_prompt,
    risk_assessor_prompt,
)
from .reranker_integration import RerankerIntegration
from .state import DeRiskerFeedback, FormatOutput, RiskAssessmentFeedback, State

# Load environment variables from .env file
load_dotenv()


class Workflow:
    def __init__(
        self,
        name: str = None,
        strategy: str = "default",
        model_name: str = "llama3.2",
        max_iter: int = MAX_ITERATIONS,
    ):
        model_config = LLM_CONFIGS[model_name]["model"]
        if model_name == "llama3.2":
            self.llm = ChatOllama(model=model_config)
        elif model_name == "gpt-4o":
            self.llm = ChatOpenAI(model=model_name, temperature=0)
        # self.llm = self.llm_base.with_structured_output(State)
        self.risk_assessor_llm = self.llm.with_structured_output(RiskAssessmentFeedback)
        self.de_risker_llm = self.llm.with_structured_output(DeRiskerFeedback)
        self.format_orchestrator_llm = self.llm.with_structured_output(FormatOutput)
        self.max_iterations = max_iter
        self.name = name
        self.system_messages = get_sys_messages(strategy, iteration_count=0)
        self.tools = [
            fetch_clinical_trials,
            fetch_fda_adverse_events,
            fetch_opentargets_associations,
            fetch_trials_in_literature,
        ]

    def evidence_retriever(self, state: State, config: RunnableConfig) -> dict:
        """
        Agent 1: Retrieve precedent evidence related to the proposal.

        This function simulates retrieval of past FDA outcomes or clinical trials that
        match the proposal's mechanism, indication, or endpoint design. In the future,
        this can be connected to a real vector database or tool API.

        Parameters
        ----------
        state : State
            The current workflow state, containing the user proposal and other metadata.
        config : RunnableConfig
            LangChain runtime configuration for the agent.

        Returns
        -------
        dict
            Dictionary containing a list of relevant evidence examples as `retrieved_evidence`.
        """
        # TODO: Vector DB call based on state.proposal
        # evidence = [
        #     "Lebrikizumab failed Phase 3 for asthma (NCT02918071) due to weak correlation of AER with symptom control.",
        #     "Dupilumab succeeded in similar eosinophilic population with endpoint of FEV1 + biomarker stratification (BLA761469).",
        # ]
        integration = RerankerIntegration(
            milvus_db="milvusdb/combined_fda_chunks_milvus.db",
            collection_name="fda_chunks",
        )
        evidence = integration.query_and_rerank(
            state["user_proposal"], top_k=10, top_n=5
        )

        llm_with_api_tools = self.llm.bind_tools(self.tools)

        input_prompt = llm_api_tool_prompt.format(
            user_proposal=state["user_proposal"], retrieved_evidence=evidence
        )

        response = llm_with_api_tools.invoke(
            [HumanMessage(content=input_prompt)], config
        )

        return {"retrieved_evidence": evidence, "api_messages": [response]}

    def should_call_tools(self, state: State):
        """Check if evidence_retriever wants to call external APIs"""

        if "api_messages" in state and state["api_messages"]:
            last_message = state["api_messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "api_tools"
        return "continue"

    def combine_evidence(self, state: State) -> dict:
        """Combine vdb retrieval evidence with API tool results"""

        all_evidence = list(state.get("retrieved_evidence", []))
        for msg in state.get("api_messages", []):
            if isinstance(msg, ToolMessage):
                all_evidence.append(f"External API Data: {msg.content}")

        return {"retrieved_evidence": all_evidence}

    def risk_assessor(self, state: State, config: RunnableConfig) -> dict:
        """
        Agent 2: Assess FDA-relevant risks and provides risk-ratingss of the proposal.

        Evaluates the proposal across four domains (Mechanistic, Biomarker, Endpoint, Safety)
        using structured outputs, a qualitative ranking (high, medium, low),
        and a numeric 1–10 score. Also provides a summary risk assessment in freeform text.

        Parameters
        ----------
        state : State
            The current workflow state, containing the user or revised proposal.
        config : RunnableConfig
            LangChain runtime configuration for the agent.

        Returns
        -------
        dict
            Dictionary containing:
            - `risk_assessment`: Text summary of the risk profile
            - `mechanistic_risk_ranking`: Risk Ranking of Mechanistic Domain
            - `biomarker_risk_ranking`: Risk Ranking of Biomarker Domain
            - `endpoint_risk_ranking`: Risk Ranking of Endpoint Domain
            - `safety_risk_ranking`: Risk Ranking of Safety Domain
            - `summary_rating`: Integer score (1–10) for overall summary ranking of proposal success
        """
        risk_assessor_system_message = self.system_messages["risk_assessor"]

        # if state.get("iteration_count", 0) > 0:
        #    input_content = f"""{state["user_proposal"]}
        #                    {state["proposal_feedback"]}"""
        # else:
        # input_content = f"""{state["user_proposal"]}"""
        print(state)
        input_prompt = risk_assessor_prompt.format(
            mechanism=state["mechanism"],
            biomarker=state["biomarker"],
            endpoint=state["endpoint"],
            safety=state["safety"],
        )
        response = self.risk_assessor_llm.invoke(
            [
                SystemMessage(content=risk_assessor_system_message),
                HumanMessage(content=input_prompt),
            ],
            config,
        )
        return response
        # return {
        #     "risk_assessment": response.risk_assessment,
        #     "mechanistic_risk_ranking": response.mechanistic,
        #     "biomarker_risk_ranking": response.biomarker,
        #     "endpoint_risk_ranking": response.endpoint,
        #     "safety_risk_ranking": response.safety,
        #     "summary_rating": response.summary_rating,
        # }

    def de_risker(self, state: State, config: RunnableConfig) -> dict:
        """
        Agent 3: Provide domain-specific critique and mitigation strategies.

        This agent addresses the weaknesses listed from the risk assessor agent,
        acknowledges and uses the evidence gathered from vector database
        and api web search tool and proposes solutions and strategies
        across the FDA-aligned categories of mechanism, biomarker,
        endpoint alignment, and safety. It also suggests alternative paths when
        appropriate.

        Parameters
        ----------
        state : State
            The current workflow state, containing the proposal and risk assessment.
        config : RunnableConfig
            LangChain runtime configuration for the agent.

        Returns
        -------
        dict
            Dictionary with:
            - `proposal_feedback`: Bullet-pointed feedback with mitigation suggestions
        """

        de_risker_system_message = self.system_messages["de_risker"]
        # proposal = (
        #     state["improved_proposal"]
        #     if state.get("iteration_count", 0) > 0
        #     else state["user_proposal"]
        # )
        input_prompt = derisker_prompt.format(
            user_proposal=state["user_proposal"],
            risk_assessment=state["risk_assessment"],
            retrieved_evidence=state["retrieved_evidence"],
        )
        response = self.de_risker_llm.invoke(
            [
                SystemMessage(content=de_risker_system_message),
                HumanMessage(content=input_prompt),
            ],
            config,
        )
        # return response
        return {  # TODO: EDIT THIS!!!!
            "mechanistic_suggestion": response.mechanistic_suggestion,
            "mechanistic_rationale": response.mechanistic_rationale,
            "mechanistic_alternative": response.mechanistic_alternative,
            "mechanistic_suggestion_history": state.get(
                "mechanistic_suggestion_history", []
            )
            + [response.mechanistic_suggestion],
            "biomarker_suggestion": response.biomarker_suggestion,
            "biomarker_rationale": response.biomarker_rationale,
            "biomarker_alternative": response.biomarker_alternative,
            "biomarker_suggestion_history": state.get(
                "biomarker_suggestion_history", []
            )
            + [response.biomarker_suggestion],
            "endpoint_suggestion": response.endpoint_suggestion,
            "endpoint_rationale": response.endpoint_rationale,
            "endpoint_alternative": response.endpoint_alternative,
            "endpoint_suggestion_history": state.get("endpoint_suggestion_history", [])
            + [response.endpoint_suggestion],
            "safety_suggestion": response.safety_suggestion,
            "safety_rationale": response.safety_rationale,
            "safety_alternative": response.safety_alternative,
            "safety_suggestion_history": state.get("safety_suggestion_history", [])
            + [response.safety_suggestion],
            # You can also track rationale/alternatives in the same way if needed
        }

        # return {
        #     "mechanistic_mitigation": response.mechanistic_suggestion,
        #     "mechanistic_rationale": response.mechanistic_rationale,
        #     "mechanistic_alternative": response.mechanistic_alternative,
        #     "biomarker_mitigation": response.biomarker_suggestion,
        #     "biomarker_rationale": response.biomarker_rationale,
        #     "biomarker_alternative": response.biomarker_alternative,
        #     "endpoint_mitigation": response.endpoint_suggestion,
        #     "endpoint_rationale": response.endpoint_rationale,
        #     "endpoint_alternative": response.endpoint_alternative,
        #     "safety_mitigation": response.safety_suggestion,
        #     "safety_rationale": response.safety_rationale,
        #     "safety_alternative": response.safety_alternative,
        #     "overall_summary_of_suggestions": response.overall_summary,
        # }

    def format_orchestrator(self, state: State, config: RunnableConfig) -> dict:
        """
        Agent 4: Format the output from the de-risker based on prior risk assessment and feedback.

        This agent generates a formatted review proposal based on the changes suggested by
        the de-risker and risk-assessor agents. The output is formatted as a concise,
        FDA-ready review under 1000 words.

        Parameters
        ----------
        state : State
            The current workflow state, containing previous versions of the proposal,
            the risk assessment, and regulatory feedback.
        config : RunnableConfig
            LangChain runtime configuration for the agent.

        Returns
        -------
        dict
            Dictionary containing:
            - `final_review_paragraph`: A refined review document
        """

        format_orchestrator_system_message = self.system_messages["format_orchestrator"]
        # proposal = (
        #     state["improved_proposal"]
        #     if state.get("iteration_count", 0) > 0
        #     else state["user_proposal"]
        # )
        # input_dict = {key: (value or "N/A") for key, value in state.items()}
        # input_prompt = format_orchestrator_prompt.format(**input_dict)

        input_prompt = format_orchestrator_prompt.format(
            user_proposal=state["user_proposal"],
            risk_assessment=state["risk_assessment"],
            # overall_summary=state["overall_summary_of_suggestions"],
            mechanistic_suggestion=state["mechanistic_suggestion"],
            mechanistic_rationale=state["mechanistic_rationale"],
            mechanistic_alternative=state["mechanistic_alternative"],
            biomarker_suggestion=state["biomarker_suggestion"],
            biomarker_rationale=state["biomarker_rationale"],
            biomarker_alternative=state["biomarker_alternative"],
            endpoint_suggestion=state["endpoint_suggestion"],
            endpoint_rationale=state["endpoint_rationale"],
            endpoint_alternative=state["endpoint_alternative"],
            safety_suggestion=state["safety_suggestion"],
            safety_rationale=state["safety_rationale"],
            safety_alternative=state["safety_alternative"],
        )
        response = self.format_orchestrator_llm.invoke(
            [
                SystemMessage(content=format_orchestrator_system_message),
                HumanMessage(content=input_prompt),
            ],
            config,
        )
        return response
        # return {"formatted_review": response.formatted_review}

    # f"""
    #     Provide a corporate-level professional formatted review and mitigation strategy
    #     for the user's original proposal.
    #     Original Proposal:\n{state["user_proposal"]}\n
    #     Final Risk Assessment:\n{state["risk_assessment"]}\n
    #     Summary of Suggestions: \n{state["overall_summary"]}\n
    #     Mechanistic Suggestions:\n{state["mechanistic_suggestion"]}\n
    #     Mechanistic Rationale:\n{state["mechanistic_rationale"]}\n
    #     Mechanistic Alternative:\n{state["mechanistic_alternative"]}\n
    #     Biomarker Suggestions:\n{state["biomarker_suggestion"]}\n
    #     Biomarker Rationale:\n{state["biomarker_rationale"]}\n
    #     Biomarker Alternative:\n{state["biomarker_alternative"]}\n
    #     Endpoint Suggestions:\n{state["endpoint_suggestion"]}\n
    #     Endpoint Rationale:\n{state["endpoint_rationale"]}\n
    #     Endpoint Alternative:\n{state["endpoint_alternative"]}\n
    #     Safety Suggestions:\n{state["safety_suggestion"]}\n
    #     Safety Rationale:\n{state["safety_rationale"]}\n
    #     Safety Alternative:\n{state["safety_alternative"]},
    #     """
    # )
    def route_proposal(self, state: State) -> str:
        """
        Decision function to determine the next step in the workflow.

        Based on the rating score and the number of iterations already run, this
        function decides whether to accept the improved proposal or return it for
        another round of critique and revision.

        Parameters
        ----------
        state : State
            The current workflow state, including score and iteration count.

        Returns
        -------
        str
            One of:
            - "pass": Proposal passes and workflow ends
            - "fail": Loop continues to another risk critique cycle
        """
        current_iter = state.get("iteration_count", 0)
        if current_iter >= self.max_iterations:
            return "pass"
        else:
            return "fail"

    def increment_iteration(self, state: State, config: RunnableConfig) -> dict:
        print("ITERATION:", state.get("iteration_count", 0))
        return {"iteration_count": state.get("iteration_count", 0) + 1}

    def human_feedback_collector(state: State, config: RunnableConfig) -> dict:
        print("Final Formatted Review:\n")
        print(state["formatted_review"])

        print("Enter your comment or decision:")
        feedback = input("Human says: ")

        return {"human_feedback": feedback}

    def build_graph(self, memory: MemorySaver) -> StateGraph:
        builder = StateGraph(State)

        builder.add_node("evidence_retriever", self.evidence_retriever)
        builder.add_node("api_tools", ToolNode(self.tools, messages_key="api_messages"))
        builder.add_node(
            "combine_evidence", self.combine_evidence
        )  # aggregates api responses with vector db response
        builder.add_node("risk_assessor", self.risk_assessor)
        builder.add_node("de_risker", self.de_risker)
        builder.add_node("format_orchestrator", self.format_orchestrator)
        builder.add_node("feedback_evaluator", self.increment_iteration)
        # builder.add_node("human_feedback", self.human_feedback_collector)

        builder.add_edge(START, "evidence_retriever")
        builder.add_conditional_edges(
            "evidence_retriever",
            self.should_call_tools,
            {"api_tools": "api_tools", "continue": "risk_assessor"},
        )
        builder.add_edge("api_tools", "combine_evidence")
        builder.add_edge("combine_evidence", "risk_assessor")
        builder.add_edge("risk_assessor", "de_risker")
        builder.add_edge("de_risker", "feedback_evaluator")

        # builder.add_edge("iteration_incrementer", "format_orchestrator")

        builder.add_conditional_edges(
            "feedback_evaluator",
            self.route_proposal,
            {
                "pass": "format_orchestrator",
                "fail": "risk_assessor",
            },
        )
        # builder.add_edge("format_orchestrator", "human_feedback")
        # builder.add_edge("human_feedback", END)

        graph = builder.compile(checkpointer=memory)
        return graph

    # def print_chat(self, state: State):
    #     print("ITERATION COUNT: ", self.iteration_count)
    #     print("Retrieve evidence:")
    #     print(state["retrieved_evidence"])
    #     print("Initial review:")
    #     print(state["risk_assessment_and_rating"])
    #     print("\n--- --- ---\n")
    #     if "proposal_feedback" in state:
    #         print("proposal_feedback:")
    #         print(state["proposal_feedback"])
    #         print("\n--- --- ---\n")

    #         print("improved_proposal:")
    #         print(state["improved_proposal"])
    #         print("grade:")
    #         print(state["pass_or_fail"])
    #     else:
    #         print("Proposal failed quality gate - lower than 5 threshold!")

    # def print_chat(self, state: State) -> None:
    #     print(f"ITERATION {state.get('iteration_count', 0)}")

    #     print("\n Original Proposal:")
    #     print(state["user_proposal"])

    #     if state.get("retrieved_evidence"):
    #         print("\n Retrieved Evidence:")
    #         for e in state["retrieved_evidence"]:
    #             print(f"- {e}")

    #     if state.get("risk_assessment_and_rating"):
    #         print("\n Risk Assessment Summary:")
    #         print(state["risk_assessment_and_rating"])

    #     if state.get("proposal_feedback"):
    #         print("\n Critique Feedback:")
    #         print(state["proposal_feedback"])

    #     if state.get("improved_proposal"):
    #         print("\n Rewritten Proposal:")
    #         print(state["improved_proposal"])

    #     print(
    #         f"\n Grade: {state.get('pass_or_fail')}  |  Score: {state.get('rating_score')}"
    #     )
