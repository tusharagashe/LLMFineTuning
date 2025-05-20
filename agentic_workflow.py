from _constants import LLM_CONFIGS, SYSTEM_MESSAGES
from IPython.display import Image, display
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict


# Graph state
class State(TypedDict):
    retrieved_evidence: str
    user_proposal: str
    risk_assessment_and_rating: str
    proposal_feedback: str
    improved_proposal: str
    pass_or_fail: str


# Schema for structured output to use in evaluation
class Feedback(BaseModel):
    grade: Literal["pass", "fail"] = Field(
        description="Decide if the proposal passes and will be successful going through FDA review. If the rating is higher than 9 out of 10, then it passes, else it fails.",
    )
    feedback: str = Field(
        description="If the proposal fails, provide the current rating and risk assessment to improve.",
    )


class Workflow:
    def __init__(self):
        self.llm = ChatOllama(model=LLM_CONFIGS["llama3.2"]["model"])
        self.evaluator = self.llm.with_structured_output(Feedback)

    # 3. Agent Functions
    @traceable(name="Evidence retreiver from database")
    def evidence_retriever(self, state: State, config: RunnableConfig):
        """Retrieve evidence from vector database of related successful and failed trials."""
        # TODO: Vector DB call based on state.proposal
        evidence = [
            "Lebrikizumab failed in Phase 3 due to lack of endpoint correlation"
        ]
        # state.retrieved_evidence = evidence
        # state.history.append("Retrieved evidence from vector DB")
        return {"retrieved_evidence": evidence}

    @traceable(name="FDA Risk Assessment")
    def risk_assessment(self, state: State, config: RunnableConfig):
        """First LLM call to provide a risk assessment of the user proposal"""
        risk_assessment_system_message = SYSTEM_MESSAGES["risk_assessment"]

        if state.get("improved_proposal"):
            grade = self.evaluator.invoke(
                [
                    SystemMessage(content=risk_assessment_system_message),
                    HumanMessage(
                        content=f"Review this proposal, provide a rating, and grade the proposal: {state['improved_proposal']} but take into account the feedback: {state['proposal_feedback']}"
                    ),
                ],
                config,
            )
        else:
            grade = self.evaluator.invoke(
                [
                    SystemMessage(content=risk_assessment_system_message),
                    HumanMessage(
                        content=f"Review this proposal, provide a rating, and grade the proposal: {state['user_proposal']}"
                    ),
                ],
                config,
            )
        return {
            "pass_or_fail": grade.grade,
            "risk_assessment_and_rating": grade.feedback,
        }

    # def check_rating(self, state: State):
    #     """Gate function to check if the risk rating is = -1 out of 10"""

    #     if "-1" in state["risk_assessment_and_rating"]:
    #         return "Fail"
    #     return "Fail"

    @traceable(name="FDA Risk Critquer")
    def regulatory_risk_critquer(self, state: State, config: RunnableConfig):
        """Second LLM call to provide specific feedback based on different categories of regulatory approval to improve the rating."""

        msg = self.llm.invoke(
            [
                SystemMessage(content=SYSTEM_MESSAGES["risk_critiquer"]),
                HumanMessage(
                    content=f"Here is the original proposal: {state['user_proposal']}. Here is the risk assessment \
                    and rating: {state['risk_assessment_and_rating']}"
                ),
            ],
            config,
        )
        return {"proposal_feedback": msg.content}

    @traceable(name="FDA Proposal Rewriter")
    def proposal_writer(self, state: State, config: RunnableConfig):
        """Third LLM call for writing a new proposal based on feedback."""

        msg = self.llm.invoke(
            [
                SystemMessage(content=SYSTEM_MESSAGES["proposal_writer"]),
                HumanMessage(
                    content=f"Here is the original proposal: {state['user_proposal']}. Here is the risk assessment \
                    and rating: {state['risk_assessment_and_rating']}. Here is the proposal feedback: {state['proposal_feedback']}"
                ),
            ],
            config,
        )
        return {"improved_proposal": msg.content}

    def route_proposal(self, state: State):
        """Route back to risk assessment agent or end based upon feedback from the evaluator"""

        if state["pass_or_fail"] == "pass":
            return "Accepted"
        elif state["pass_or_fail"] == "fail":
            return "Rejected + Feedback"

    def build_graph(self, memory: MemorySaver) -> StateGraph:
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("retrieve_evidence", self.evidence_retriever)
        workflow.add_node("risk_assessment", self.risk_assessment)
        workflow.add_node("regulatory_risk_critquer", self.regulatory_risk_critquer)
        workflow.add_node("proposal_writer", self.proposal_writer)

        # Add edges to connect nodes
        workflow.add_edge(START, "retrieve_evidence")
        workflow.add_edge("retrieve_evidence", "risk_assessment")
        # workflow.add_conditional_edges(
        #     "risk_assessment",
        #     self.check_rating,
        #     {"Fail": "regulatory_risk_critquer", "Pass": END},
        # )
        workflow.add_edge("risk_assessment", "regulatory_risk_critquer")
        workflow.add_edge("regulatory_risk_critquer", "proposal_writer")
        workflow.add_conditional_edges(
            "proposal_writer",
            self.route_proposal,
            {
                "Accepted": END,
                "Rejected + Feedback": "risk_assessment",
            },
        )
        graph = workflow.compile(checkpointer=memory)
        return graph

    def print_chat(self, state: State):
        print("Retrieve evidence:")
        print(state["retrieved_evidence"])
        print("Initial review:")
        print(state["risk_assessment_and_rating"])
        print("\n--- --- ---\n")
        if "proposal_feedback" in state:
            print("proposal_feedback:")
            print(state["proposal_feedback"])
            print("\n--- --- ---\n")

            print("improved_proposal:")
            print(state["improved_proposal"])
            print("grade:")
            print(state["pass_or_fail"])
        else:
            print("Proposal failed quality gate - lower than 5 threshold!")
