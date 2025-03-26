import pandas as pd
import openai
import json
import streamlit as st

# Load dataset
def load_data(file_path):
    """Load an Excel file into a Pandas DataFrame."""
    df = pd.read_excel(file_path, sheet_name=0)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df

# Function to interact with OpenAI LLM for rule suggestions
def query_llm(prompt, api_key):
    """Queries OpenAI GPT for compliance-related rule suggestions."""
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a compliance expert."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# Function to validate data based on profiling rules
def apply_profiling_rules(df, rules):
    """Applies profiling rules and flags invalid records."""
    flagged_records = {}
    
    for column, rule in rules.items():
        if column in df.columns:
            try:
                flagged = df[~df[column].apply(lambda x: eval(rule) if pd.notna(x) else True)]
                if not flagged.empty:
                    flagged_records[column] = flagged
            except Exception as e:
                flagged_records[column] = f"Error in rule '{rule}': {e}"

    return flagged_records

# Streamlit GUI
def compliance_ui():
    """Launches Streamlit-based Compliance Assistant UI."""
    st.title("🛡️ Interactive Compliance Assistant")

    # Upload dataset
    uploaded_file = st.file_uploader("📂 Upload an Excel file", type=["xlsx"])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        st.write("📊 **Dataset Preview:**", df.head())

        # Persistent rule storage
        if "profiling_rules" not in st.session_state:
            st.session_state.profiling_rules = {}

        # Rule Management
        st.subheader("📏 Define Profiling Rules")
        column = st.selectbox("Select Column", df.columns)
        rule = st.text_input("Enter Rule (e.g., 'x >= 0')")

        if st.button("➕ Add Rule"):
            st.session_state.profiling_rules[column] = rule
            st.success(f"✅ Rule added: `{column} -> {rule}`")

        # Display Current Rules
        st.subheader("📌 Current Profiling Rules")
        st.json(st.session_state.profiling_rules)

        # Modify or Delete Rules
        if st.session_state.profiling_rules:
            rule_to_modify = st.selectbox("🔄 Modify/Delete Rule", list(st.session_state.profiling_rules.keys()))
            new_rule = st.text_input(f"Enter New Rule for `{rule_to_modify}`")

            if st.button("🔄 Modify Rule"):
                st.session_state.profiling_rules[rule_to_modify] = new_rule
                st.success(f"✅ Rule modified: `{rule_to_modify} -> {new_rule}`")

            if st.button("❌ Delete Rule"):
                del st.session_state.profiling_rules[rule_to_modify]
                st.warning(f"⚠️ Rule deleted for `{rule_to_modify}`")

        # Validate Data
        if st.button("✅ Validate Data"):
            flagged = apply_profiling_rules(df, st.session_state.profiling_rules)
            if flagged:
                st.subheader("🚨 Flagged Records")
                for col, issues in flagged.items():
                    st.write(f"⚠️ **Column:** `{col}`")
                    st.dataframe(issues)
            else:
                st.success("✅ No issues detected!")

        # LLM Assistance
        st.subheader("🧠 Get AI-Powered Rule Suggestions")
        if st.button("🔍 Suggest Rules"):
            api_key = st.text_input("Enter OpenAI API Key (Hidden)", type="password")
            if api_key:
                prompt = f"Suggest data validation rules for the following columns: {', '.join(df.columns)}"
                llm_response = query_llm(prompt, api_key)
                st.write("💡 AI Suggestions:", llm_response)

# Run Streamlit app
if __name__ == "__main__":
    compliance_ui()
