from IPython.display import Image, display
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ._constants import LLM_CONFIGS, MAX_ITERATIONS
from .prompts import get_sys_messages
from .state import Feedback, State


class Workflow:
    def __init__(self, name: str = None, strategy: str = "default"):
        # self.iteration_count = 0
        self.llm = ChatOllama(model=LLM_CONFIGS["llama3.2"]["model"])
        self.evaluator = self.llm.with_structured_output(Feedback)
        self.max_iterations = MAX_ITERATIONS
        self.name = name
        self.system_prompts = get_sys_messages(strategy)

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
        evidence = [
            "Lebrikizumab failed Phase 3 for asthma (NCT02918071) due to weak correlation of AER with symptom control.",
            "Dupilumab succeeded in similar eosinophilic population with endpoint of FEV1 + biomarker stratification (BLA761469).",
        ]
        return {"retrieved_evidence": evidence}

    def risk_assessment(self, state: State, config: RunnableConfig) -> dict:
        """
        Agent 2: Assess FDA-relevant risks and rate the proposal.

        Evaluates the proposal across four domains (Mechanistic, Biomarker, Endpoint, Safety)
        using structured outputs and a numeric 1–10 score. Also provides a summary risk
        assessment in freeform text.

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
            - `risk_assessment_and_rating`: Text summary of the risk profile
            - `pass_or_fail`: "pass" or "fail" decision
            - `rating_score`: Integer score (1–10) used for routing logic
        """
        risk_assessment_system_message = self.system_prompts["risk_assessment"]

        if state.get("iteration_count", 0) > 0:
            input_content = f"""Review this revised proposal: {state["improved_proposal"]}
                            Previous feedback: {state["proposal_feedback"]}
                            Return a 1–10 rating"""
        else:
            input_content = f"""Review this proposal: {state["user_proposal"]}
                            Return a 1–10 rating"""
        response = self.evaluator.invoke(
            [
                SystemMessage(content=risk_assessment_system_message),
                HumanMessage(content=input_content),
            ],
            config,
        )

        return {
            "risk_assessment_and_rating": response.feedback,
            "pass_or_fail": response.grade,
            "rating_score": response.rating,
        }

    def regulatory_risk_critquer(self, state: State, config: RunnableConfig) -> dict:
        """
        Agent 3: Provide domain-specific critique and mitigation strategies.

        This agent identifies weaknesses from the risk assessment and proposes
        improvements across FDA-aligned categories like mechanism, biomarker,
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

        risk_critique_system_message = self.system_prompts["risk_critiquer"]
        proposal = (
            state["improved_proposal"]
            if state.get("iteration_count", 0) > 0
            else state["user_proposal"]
        )

        response = self.llm.invoke(
            [
                SystemMessage(content=risk_critique_system_message),
                HumanMessage(
                    content=f"""Proposal:\n{proposal}\n\n
            Risk Assessment:\n{state["risk_assessment_and_rating"]}"""
                ),
            ],
            config,
        )

        return {"proposal_feedback": response.content}

    def proposal_writer(self, state: State, config: RunnableConfig) -> dict:
        """
        Agent 4: Rewrite the proposal based on prior risk assessment and feedback.

        This agent generates an improved proposal incorporating changes suggested by
        the risk critiquer and reviewer agents. The output is formatted as a concise,
        FDA-ready proposal under 250 words.

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
            - `improved_proposal`: A refined proposal ready for re-evaluation
        """

        proposal_writer_sys_message = self.system_prompts["proposal_writer"]
        proposal = (
            state["improved_proposal"]
            if state.get("iteration_count", 0) > 0
            else state["user_proposal"]
        )

        response = self.llm.invoke(
            [
                SystemMessage(content=proposal_writer_sys_message),
                HumanMessage(
                    content=f"""
                    Rewrite the following proposal using these insights:
                    Original Proposal:\n{proposal}
                    Risk Assessment:\n{state["risk_assessment_and_rating"]}
                    Critique:\n{state["proposal_feedback"]}
                    """
                ),
            ],
            config,
        )

        return {"improved_proposal": response.content}

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
            - "Accepted": Proposal passes and workflow ends
            - "Rejected + Feedback": Loop continues to another risk critique cycle
        """
        current_iter = state.get("iteration_count", 0)
        if current_iter >= self.max_iterations:
            return "Accepted"
        else:
            return "Rejected + Feedback"

    def increment_iteration(self, state: State, config: RunnableConfig) -> dict:
        print(state.get("iteration_count", 0))
        print(state.get("rating_score", -1))
        return {"iteration_count": state.get("iteration_count", 0) + 1}

    def build_graph(self, memory: MemorySaver) -> StateGraph:
        builder = StateGraph(State)

        builder.add_node("retrieve_evidence", self.evidence_retriever)
        builder.add_node("risk_assessment", self.risk_assessment)
        builder.add_node("regulatory_risk_critquer", self.regulatory_risk_critquer)
        builder.add_node("proposal_writer", self.proposal_writer)
        builder.add_node("iteration_incrementer", self.increment_iteration)

        builder.add_edge(START, "retrieve_evidence")
        builder.add_edge("retrieve_evidence", "risk_assessment")
        builder.add_edge("risk_assessment", "regulatory_risk_critquer")
        builder.add_edge("regulatory_risk_critquer", "proposal_writer")
        builder.add_edge("proposal_writer", "iteration_incrementer")

        builder.add_conditional_edges(
            "iteration_incrementer",
            self.route_proposal,
            {
                "Accepted": END,
                "Rejected + Feedback": "risk_assessment",
            },
        )
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

    def print_chat(self, state: State) -> None:
        print(f"ITERATION {state.get('iteration_count', 0)}")

        print("\n Original Proposal:")
        print(state["user_proposal"])

        if state.get("retrieved_evidence"):
            print("\n Retrieved Evidence:")
            for e in state["retrieved_evidence"]:
                print(f"- {e}")

        if state.get("risk_assessment_and_rating"):
            print("\n Risk Assessment Summary:")
            print(state["risk_assessment_and_rating"])

        if state.get("proposal_feedback"):
            print("\n Critique Feedback:")
            print(state["proposal_feedback"])

        if state.get("improved_proposal"):
            print("\n Rewritten Proposal:")
            print(state["improved_proposal"])

        print(
            f"\n Grade: {state.get('pass_or_fail')}  |  Score: {state.get('rating_score')}"
        )
