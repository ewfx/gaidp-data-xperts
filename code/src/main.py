import openai
import pandas as pd
import json
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# Load the CSV file
def load_csv(file_path):
    """Loads a CSV file into a Pandas DataFrame."""
    return pd.read_csv(file_path)

# Placeholder for API key
api_key = "api-key-here"

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

def extract_profiling_rules(df):
    """Generate profiling rules using OpenAI LLM."""
    prompt = f"""
    Given the following dataset sample, extract profiling rules that define data quality, consistency, and formatting:
    {df.head().to_string()}
    
    Provide the rules in valid JSON format, such as:
    [
        {{"Field No":"<Field No>","Field Name": "<Field Name>","MDRM":"<MDRM>","Description": "<Validation Logic>", "Allowable values":"<Allowable values>"}},
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
    
    # Clean up the response (remove any preface text)
    clean_response_content = response_content.strip()
    clean_response_content = clean_response_content.split("json\n")[1].split("\n")[0]
            
    
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

# Function to save results to an Excel file
def save_results_to_excel(anomalies, clusters, validations, remediations, rules, output_path):
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

# Example usage
file_path = "C:\\Users\\YourUser\\Downloads\\Sample_Data.csv"
output_path = "C:\\Users\\YourUser\\Downloads\\ValidationResults.xlsx"
preview_data, data_frames = extract_data(file_path)

# Perform anomaly detection and clustering on each sheet
anomaly_results = {}
cluster_results = {}
validation_results = {}
remediation_suggestions = {}
validation_rules_list = []

for sheet, df in data_frames.items():
    anomaly_results[sheet] = detect_anomalies(df)
    cluster_results[sheet] = cluster_data(df)
    
    # Extract rules dynamically based on numeric columns
    extracted_rules = {col: "x >= 0" for col in df.select_dtypes(include=['number']).columns}  
    validation_results[sheet] = validate_data(df, extracted_rules)
    
    # Generate remediation suggestions
    remediation_suggestions[sheet] = suggest_remediation(anomaly_results)
    
    # Generate validation rules
    validation_rules_list.append(extract_profiling_rules(df))

# Combine validation rules into a single DataFrame
validation_rules_df = pd.concat(validation_rules_list, ignore_index=True)
print(validation_rules_df)

# Save results to an Excel file
save_results_to_excel(anomaly_results, cluster_results, validation_results, remediation_suggestions, validation_rules_df, output_path)

# Construct a prompt using the preview data (modify as needed)
prompt = f"""
Analyze the following dataset preview and provide an interpretation of regulatory reporting requirements:
{preview_data}
"""

# Load the file
df = load_csv("C:\\Users\\mahen\\Downloads\\Sample_H2_Data.csv")

# Extract profiling rules
response = extract_profiling_rules(df)
print(response)
