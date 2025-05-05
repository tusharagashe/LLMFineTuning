import requests as re
import json
import pandas as pd 
import io

API_server = "https://clinicaltrials.gov/api/v2"
def request_study(nct: str, format: 'json'):
    # Clinical Trial API V2 Parameters 
    url = f"{API_server}/studies/{nct}"
    params = {
        "format": format,
        "markupFormat": "markdown",
    }
    
    if format == 'json':
        
        headers = {
            "accept": "application/csv"
        }
        # Call
        response = re.get(url, headers = headers, params = params)
        # error 
        if response.status_code == 200:
            return pd.json_normalize(response.json())
    
    elif format == 'csv':
        headers = {
            "accept": "text/csv"
        }
        # Call
        response = re.get(url, headers = headers, params = params)
        # Error 
        if response.status_code == 200:
            return pd.read_csv(io.StringIO(response.text))
    
    else:
        print("Failed to retrieve data for {nct}: {response.status_code}")
        return None

if __name__ == '__main__':
    
    # Tanezumab
    terminated = 'NCT00863772'
    completed = 'NCT02709486'
    
    tanezumab_terminated = pd.DataFrame(request_study(terminated, 'json'))
    tanezumab_terminated.to_csv('gt-data/data/terminated/tanezumab_terminated.csv')
    
    tanezumab_completed = pd.DataFrame(request_study(completed, 'json'))
    tanezumab_completed.to_csv('gt-data/data/completed/tanezumab/tanezumab_completed.csv')
    
    
    # test = pd.DataFrame(request_study(completed), 
    #                     columns = 
    #                     ['NCT Number', 'Brief Summary', 'Conditions', 'Interventions', 'Primary Outcome Measures', 'Secondary Outcome Measures', 'Other Outcome Measures']
    #                     ).head()
    
    # print(test['Brief Summary'][0])