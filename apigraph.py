from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langsmith import traceable
import requests 

#need to update later to generally fit all api responses/integrate with larger workflow. 
# possibly have larger object that is passed through whole workflow 
class State(TypedDict):
    drug: str
    api_data: dict
    adverse_events: list

@traceable(name="FDA Adverse Events Endpoint")
def fetch_adverse_events(state: State) -> dict:
    drug = state["drug"]

    try:
        response = requests.get(
            "https://api.fda.gov/drug/event.json",
            params={"search": f'patient.drug.medicinalproduct:"{drug}"', "limit": 5}
        )
        response_json = response.json()
        events = []

        for result in response_json["results"]:
            reactions = result["patient"]["reaction"]
            for reaction in reactions:
                if "reactionmeddrapt" in reaction:
                    events.append(reaction["reactionmeddrapt"])

        events = list(set(events))
        
    except Exception as e:
        data = {"error": str(e)}
        events = []

    return {
        "drug": drug,
        "api_data": response_json,  
        "adverse_events": events  
    }


graph_builder = StateGraph(State)
graph_builder.add_node("fetch_api", fetch_adverse_events)
graph_builder.add_edge(START, "fetch_api")
graph_builder.add_edge("fetch_api", END)
graph = graph_builder.compile()


def get_raw_fda_data(drug: str) -> dict:
    state = {"drug": drug}
    result = graph.invoke(state)
    return result


if __name__ == "__main__":
    result = get_raw_fda_data("metformin")
    print(result["adverse_events"])