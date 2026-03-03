import json

import streamlit as st

# Load JSON
uploaded_file = st.file_uploader("Upload your regulatory JSON", type=["json"])
data = None
if uploaded_file:
    string_data = uploaded_file.read().decode("utf-8")
    data = json.loads(string_data)
else:
    st.warning("Please upload a JSON file to begin.")
    st.stop()

# Sidebar summary
st.sidebar.header("Trial Info")
st.sidebar.write("**Indication**:", data.get("indication"))
st.sidebar.write("**Mechanism**:", data.get("mechanism"))
st.sidebar.write("**Biomarker**:", data.get("biomarker"))
st.sidebar.write("**Endpoints**:", data.get("endpoint"))
st.sidebar.write("**Safety Notes**:", data.get("safety"))
st.sidebar.write("**Iteration Count**:", data.get("iteration_count"))
st.sidebar.write("**Summary Rating:**", data.get("summary_rating"))

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Overview",
        "Risk Ranking",
        "Suggestions",
        "Evidence",
        "Final Review",
    ]
)

# --- Overview Tab ---
with tab1:
    st.subheader("Proposal Summary")
    st.write(data.get("user_proposal"))

# --- Risk Ranking Tab ---
with tab2:
    st.subheader("Risk Rankings")
    st.write("**Mechanism Risk:**", data.get("mechanistic_risk_ranking"))
    st.write("**Biomarker Risk:**", data.get("biomarker_risk_ranking"))
    st.write("**Endpoint Risk:**", data.get("endpoint_risk_ranking"))
    st.write("**Safety Risk:**", data.get("safety_risk_ranking"))

# --- Suggestions Tab ---
with tab3:

    def show_section(title, suggestion, rationale, alternative, history):
        with st.expander(f"{title}"):
            st.write("**Suggestion**:", suggestion)
            st.write("**Rationale**:", rationale)
            st.write("**Alternative**:", alternative)
            st.write("**Iterative Feedback History**:")
            for item in history:
                st.markdown(f"- {item}")

    show_section(
        "Mechanism",
        data.get("mechanistic_suggestion"),
        data.get("mechanistic_rationale"),
        data.get("mechanistic_alternative"),
        data.get("mechanistic_suggestion_history"),
    )

    show_section(
        "Biomarker",
        data.get("biomarker_suggestion"),
        data.get("biomarker_rationale"),
        data.get("biomarker_alternative"),
        data.get("biomarker_suggestion_history"),
    )

    show_section(
        "Endpoint",
        data.get("endpoint_suggestion"),
        data.get("endpoint_rationale"),
        data.get("endpoint_alternative"),
        data.get("endpoint_suggestion_history"),
    )

    show_section(
        "Safety",
        data.get("safety_suggestion"),
        data.get("safety_rationale"),
        data.get("safety_alternative"),
        data.get("safety_suggestion_history"),
    )

# --- Retrieved Evidence Tab ---
with tab4:
    st.subheader("Retrieved Evidence")
    for i, entry in enumerate(data.get("retrieved_evidence", [])):
        with st.expander(f"Evidence {i + 1}"):
            st.text(entry[:5000])  # show up to 1000 characters, adjust as needed

# --- Final Review Tab ---
with tab5:
    st.subheader("Regulatory Review Summary")
    st.markdown(data.get("final_review_paragraph"))
