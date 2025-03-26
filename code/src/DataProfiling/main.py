import openai
import pandas as pd
import json
import re
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from collections import defaultdict

# Load the CSV file
def load_csv(file_path):
    """Loads a CSV file into a Pandas DataFrame."""
    return pd.read_csv(file_path)

# Placeholder for API key
api_key = "<api_key_here>"

# Extract preview data
def extract_data(file_path):
    """Extracts preview data from CSV."""
    df = load_csv(file_path)
    sheet_previews = {"CSV Data": df.head()}
    data_frames = {"CSV Data": df}
    return sheet_previews, data_frames

# Function to query OpenAI LLM
def query_llm(prompt, api_key):
    """Queries OpenAI's LLM for regulatory interpretation."""
    response = openai.ChatCompletion.create(
        #model="gpt-4",
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an expert in regulatory compliance."},
                  {"role": "user", "content": prompt}],
        api_key=api_key
    )
    return response['choices'][0]['message']['content']


# Unsupervised Machine Learning for Data Consistency Validation
def detect_anomalies(df):
    """Uses Isolation Forest to detect anomalies in numerical data."""
    numeric_df = df.select_dtypes(include=['number']).dropna()
    if numeric_df.shape[1] == 0:
        return "No numerical data available for anomaly detection."
    
    model = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly'] = model.fit_predict(numeric_df)
    return df[df['Anomaly'] == -1]  # Return detected anomalies

def cluster_data(df, n_clusters=3):
    """Clusters numerical data using K-Means."""
    numeric_df = df.select_dtypes(include=['number']).dropna()
    if numeric_df.shape[1] == 0:
        return "No numerical data available for clustering."
    
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = model.fit_predict(numeric_df)
    return df

# Function to validate data against extracted rules
def validate_data(df, rules):
    """Validates the dataset against extracted rules."""
    validation_results = {}
    for column, rule in rules.items():
        if column in df.columns:
            try:
                validation_results[column] = df[column].apply(lambda x: eval(rule) if pd.notna(x) else False)
            except Exception as e:
                validation_results[column] = f"Validation error: {str(e)}"
    return validation_results

# Function to generate validation rules
def generate_validation_rules(df):
    """Creates validation rules based on dataset fields."""
    rules = []
    for i, column in enumerate(df.columns):
        rule = {
            "Field No": i + 1,
            "Field Name": column,
            "MDRM": "N/A",  # Placeholder, modify as needed
            "Description": f"Validation rule for {column}",
            "Allowable values": "x >= 0" if df[column].dtype in ['int64', 'float64'] else "String/Text"
        }
        rules.append(rule)
    return pd.DataFrame(rules)

def extract_profiling_rules(df):
    """Generate profiling rules using OpenAI LLM."""
    prompt = f"""
    Given the following dataset sample, extract profiling rules that define data quality, consistency, and formatting:
    {df.head().to_string()}
    
    Provide the rules in valid JSON format, such as:
    [
        {{"Field No":"<Field No>","Field Name": "<Field Name>","Validation Logic": "<Python Expression>","Description": "<Validation Logic>", "Allowable values":"<Allowable values>"}},
        ...
    ]    
    """
    
    response = openai.ChatCompletion.create(
        #model="gpt-4",
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in regulatory compliance and data profiling."},
            {"role": "user", "content": prompt}
        ],
        api_key=api_key
    )
    # Extract the content of the response
    response_content = response['choices'][0]['message']['content']
        
    start_index = response_content.find('json\n') + len('json\n') 
    end_index = response_content.find('\n```', start_index) 
    clean_response_content = response_content[start_index:end_index].strip()
            
    
    try:
        # Try to parse the content as JSON
        profiling_rules = json.loads(clean_response_content)
        
        # Check if profiling_rules is a valid list and contains entries
        if isinstance(profiling_rules, list) and profiling_rules:
            # Convert the JSON into a pandas DataFrame
            return pd.DataFrame(profiling_rules)
        else:
            print("No valid profiling rules found in the response.")
            return pd.DataFrame()  # Return an empty DataFrame if no valid rules
    
    except json.JSONDecodeError:
        # If the response isn't valid JSON, return an empty DataFrame or handle accordingly
        print("Error decoding JSON from response:", response_content)
        return pd.DataFrame()    
    #return response['choices'][0]['message']['content']


# Function to suggest remediation actions
def suggest_remediation(anomalies):
    """Provides remediation actions and explanations for flagged transactions."""
    remediation_actions = {}
    for sheet, df in anomalies.items():
        if isinstance(df, str):  # Skip sheets with no numerical data
            remediation_actions[sheet] = "No remediation needed."
            continue
        
        actions = []
        for index, row in df.iterrows():
            action = f"Row {index}: Investigate irregular value {row.to_dict()} and verify data accuracy. "
            action += "Potential remediation: Verify data entry, cross-check with source records, or escalate for review."
            actions.append(action)

        remediation_actions[sheet] = actions        
    return remediation_actions

def generate_validation_function(validation_logic):
    """Generate a Python function dynamically for validation."""
    def validate(value):
        try:
            #return eval(validation_logic, {"value": value, "re": re})
            return bool(re.match(validation_logic, str(value)))
        except Exception as e:
            return False  # Default to False if the validation logic fails
    return validate

def validate_data_logic(df, rules):
    """Validate the data using extracted profiling rules."""
    validation_results = []
    print(rules)
    for rule in rules:
        field_name = rule["Field Name"]
        validation_logic = rule["Validation Logic"]
        
        if field_name in df.columns:
            validate_func = generate_validation_function(validation_logic)
            
            invalid_rows = df[~df[field_name].astype(str).apply(validate_func)]
            
            for idx in invalid_rows.index:
                validation_results.append({
                    "Row": idx,
                    "Field": field_name,
                    "Issue": f"Value '{df.at[idx, field_name]}' does not satisfy rule: {validation_logic}"
                })
    
    return validation_results

# Adaptive Risk Scoring System
def calculate_risk_scores(df, past_violations):
    """Assigns adaptive risk scores based on anomalies, rule violations, and historical patterns."""
    risk_scores = defaultdict(int)
    
    # Identify anomalies using Isolation Forest
    numeric_df = df.select_dtypes(include=['number']).dropna()
    if not numeric_df.empty:
        model = IsolationForest(contamination=0.05, random_state=42)
        df['Anomaly'] = model.fit_predict(numeric_df)
        for index in df[df['Anomaly'] == -1].index:
            risk_scores[index] += 10  # High risk for anomalies
    
    # Check for past violations and adjust scores
    for index, row in df.iterrows():
        for col in df.columns:
            if (col, row[col]) in past_violations:
                risk_scores[index] += past_violations[(col, row[col])] * 5  # Increase score if past violations exist
    
    # Normalize risk scores (0-100 scale)
    if risk_scores:
        max_score = max(risk_scores.values())
        if max_score > 0:
            for key in risk_scores:
                risk_scores[key] = (risk_scores[key] / max_score) * 100
    
    df['Risk Score'] = df.index.map(lambda idx: risk_scores.get(idx, 0))
    return df

# Function to track past violations
def update_past_violations(df, past_violations):
    """Updates past violations with current high-risk transactions."""
    for index, row in df.iterrows():
        if row['Risk Score'] > 70:  # Consider >70 as a high-risk transaction
            for col in df.columns:
                past_violations[(col, row[col])] += 1
    return past_violations


# Function to save results to an Excel file
def save_results_to_excel(anomalies, clusters, validations, remediations, rules, risk_Score, output_path):
    """Saves the output results to an Excel file for easy extraction."""
    with pd.ExcelWriter(output_path) as writer:
        for sheet, df in anomalies.items():
            if isinstance(df, str):
                pd.DataFrame([df], columns=["Message"]).to_excel(writer, sheet_name=f"Anomalies_{sheet}", index=False)
            else:
                df.to_excel(writer, sheet_name=f"Anomalies_{sheet}")
        
        for sheet, df in clusters.items():
            if isinstance(df, str):
                pd.DataFrame([df], columns=["Message"]).to_excel(writer, sheet_name=f"Clusters_{sheet}", index=False)
            else:
                df.to_excel(writer, sheet_name=f"Clusters_{sheet}")
        
        for sheet, validation in validations.items():
            pd.DataFrame.from_dict(validation, orient='index').to_excel(writer, sheet_name=f"Validations_{sheet}")
        
        for sheet, remediation in remediations.items():
            pd.DataFrame(remediation, columns=["Remediation Actions"]).to_excel(writer, sheet_name=f"Remediations_{sheet}", index=False)
        
        rules.to_excel(writer, sheet_name="Validation Rules", index=False)
        risk_Score.to_excel(writer, sheet_name="Risk Score", index=False)

# Example usage
file_path = "Sample_H2_Data.csv"
output_path = "ValidationResults_H2.xlsx"
preview_data, data_frames = extract_data(file_path)

# Perform anomaly detection and clustering on each sheet
anomaly_results = {}
cluster_results = {}
validation_results = {}
validation_results_logic = {}
validation_final_results = {}
remediation_suggestions = {}
validation_rules_list = []

for sheet, df in data_frames.items():
    #anomaly_results[sheet] = detect_anomalies(df)
    #cluster_results[sheet] = cluster_data(df)  
    
    
    
    
    # Generate validation rules
    validation_rules_list.append(extract_profiling_rules(df))

    # Validate the dataset
    #validation_results_logic = validate_data_logic(df, validation_rules_list)
    if not validation_rules_list[-1].empty:
        validation_results_logic = validate_data_logic(df, validation_rules_list[-1].to_dict(orient="records"))
    else:
        validation_results_logic = []

    print(validation_results_logic)
    #validation_results[sheet] = {item['Row']: {'Field': item['Field'], 'Issue': item['Issue']} for item in validation_results_logic}#dict(validation_results_logic)
    for entry in validation_results_logic:
        row = entry["Row"]
        if row not in validation_final_results:
            validation_final_results[row] = []
        validation_final_results[row].append({"Field": entry["Field"], "Issue": entry["Issue"]})
    
    validation_results[sheet] = validation_final_results
    
    # Combine validation rules into a single DataFrame
    validation_rules_df = pd.concat(validation_rules_list, ignore_index=True)
    print(validation_rules_df)

    # Load past violations (simulate historical tracking)
    past_violations = defaultdict(int)
    df = calculate_risk_scores(df, past_violations)
    past_violations = update_past_violations(df, past_violations)

    anomaly_results[sheet] = detect_anomalies(df)
    cluster_results[sheet] = cluster_data(df)
    # Generate remediation suggestions
    remediation_suggestions[sheet] = suggest_remediation(anomaly_results)[sheet]

# Save results to an Excel file
save_results_to_excel(anomaly_results, cluster_results, validation_results, remediation_suggestions, validation_rules_df, df, output_path)

# Construct a prompt using the preview data (modify as needed)
prompt = f"""
Analyze the following dataset preview and provide an interpretation of regulatory reporting requirements:
{preview_data}
"""

# Query the LLM (ensure to replace api_key with actual value)
response = query_llm(prompt, api_key)
# Load the file
df = load_csv("Sample_H2_Data.csv")

