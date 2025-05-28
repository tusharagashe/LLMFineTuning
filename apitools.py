from langchain.tools import tool
import requests
from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
import operator

class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    drug: str
    trials: Optional[List[str]]
    adverse_events: Optional[List[str]]
    associations: Optional[List[str]]
    literature_failures: Optional[List[str]]


@tool
def fetch_clinical_trials(drug: str) -> dict:
    """Search ClinicalTrials.gov for trials involving a drug or condition with comprehensive status coverage (successes and failures)."""
    
    target_statuses = [
        "COMPLETED",
        "TERMINATED",     
        "WITHDRAWN",      
        "SUSPENDED",      
        "ACTIVE_NOT_RECRUITING"
    ]
    
    all_trials = []
    
    try:
        for status in target_statuses:
            params = {
                "query.term": drug,
                "filter.overallStatus": status,
                "fields": "NCTId,BriefTitle,BriefSummary,OverallStatus,WhyStopped,CompletionDate,StartDate",
                "pageSize": 3,
                "format": "json"
            }
            
            r = requests.get("https://clinicaltrials.gov/api/v2/studies", params=params)
            r.raise_for_status()
            
            data = r.json()
            studies = data.get("studies", [])
            
            for study in studies:
                protocol = study.get("protocolSection", {})
                identification = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                description_module = protocol.get("descriptionModule", {})
                
                
                title = identification.get("briefTitle", "")
                overall_status = status_module.get("overallStatus", "N/A")
                why_stopped = status_module.get("whyStopped", "")
                completion_date = status_module.get("primaryCompletionDate", {}).get("date", "")
                start_date = status_module.get("studyFirstSubmitDate", "")
                description = description_module.get("briefSummary", "")
                                                     
                trial_text = f"[{status}] {title} | Description: {description} (Status: {overall_status})"
                if completion_date:
                    trial_text += f" | Completed: {completion_date}"
                if why_stopped:
                    trial_text += f" | Reason: {why_stopped}"
                
                all_trials.append(trial_text)
            
    except Exception as e:
        all_trials = [f"Error fetching trials: {str(e)}"]
    
    return {"drug": drug, "trials": all_trials}

@tool
def fetch_fda_adverse_events(drug: str) -> State:
    """Get serious adverse events reported for a drug from OpenFDA."""
    try:
        params = {
            "search": f'patient.drug.medicinalproduct:"{drug}" AND serious:1',
            "limit": 5
        }
        r = requests.get("https://api.fda.gov/drug/event.json", params=params)
        r.raise_for_status()
        results = r.json()["results"]
        reactions = set()
        for result in results:
            for r in result["patient"]["reaction"]:
                reactions.add(r["reactionmeddrapt"])
        adverse_events = list(reactions)
    except Exception as e:
        adverse_events = [f"Error fetching adverse events: {str(e)}"]
    return {"drug": drug, "adverse_events": adverse_events}

@tool
def fetch_opentargets_associations(symbol: str):
    """Get top disease associations for a gene symbol from Open Targets."""
    
    search_query = {
        "query": """
        query search($queryString: String!) {
            search(queryString: $queryString, entityNames: ["target"]) {
                hits {
                    id
                    name
                    object {
                        ... on Target {
                            id
                            approvedSymbol
                        }
                    }
                }
            }
        }
        """,
        "variables": {"queryString": symbol}
    }
    
    try:
        resp = requests.post(
            "https://api.platform.opentargets.org/api/v4/graphql",
            json=search_query
        )
        resp.raise_for_status()
        
        hits = resp.json()["data"]["search"]["hits"]
        if not hits:
            return [f"Gene symbol '{symbol}' not found"]
        
        ensembl_id = hits[0]["id"]
        
        assoc_query = {
            "query": """
            query target($ensemblId: String!) {
                target(ensemblId: $ensemblId) {
                    associatedDiseases(page: {index: 0, size: 10}) {
                        rows {
                            disease {
                                id
                                name
                            }
                            score
                        }
                    }
                }
            }
            """,
            "variables": {"ensemblId": ensembl_id}
        }
        
        resp = requests.post(
            "https://api.platform.opentargets.org/api/v4/graphql",
            json=assoc_query
        )
        resp.raise_for_status()
        
        target_data = resp.json()["data"]["target"]
        if not target_data or not target_data["associatedDiseases"]["rows"]:
            return [f"No disease associations found for {symbol}"]
        
        associations = []
        for row in target_data["associatedDiseases"]["rows"]:
            disease_name = row["disease"]["name"]
            score = row["score"]
            associations.append(f"{disease_name} (score: {score:.2f})")
        
        return associations
        
    except requests.RequestException as e:
        return [f"API request failed: {str(e)}"]
    except KeyError as e:
        return [f"Unexpected response format: {str(e)}"]
    except Exception as e:
        return [f"Error: {str(e)}"]


#used europepmc because pubmed api is a bit more complex to decode and parse 
@tool
def fetch_trials_in_literature(query: str) -> State:
    """Search EuropePMC for clinical trial failures related to a query string."""
    try:
        params = {
            "query": f"{query} AND clinical trial AND failure",
            "format": "json",
            "pageSize": 5
        }
        r = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params=params)
        r.raise_for_status()
        articles = r.json()["resultList"]["result"]
        abstracts = [f"{a['title']} ({a.get('pubYear', '')}): {a.get('abstractText', 'No abstract')[:200]}..."
                     for a in articles]
    except Exception as e:
        abstracts = [f"Error fetching literature: {str(e)}"]
    return {"drug": query, "literature_failures": abstracts}
  