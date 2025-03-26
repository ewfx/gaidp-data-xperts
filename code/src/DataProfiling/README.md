# Regulatory Compliance Data Profiling & Validation

## Overview
This project is designed to automate regulatory compliance data profiling, anomaly detection, validation, and remediation using Python. It leverages machine learning techniques such as Isolation Forest and K-Means clustering for data consistency validation and OpenAI's GPT-based LLM for extracting profiling rules.

## Features
- **CSV Data Processing**: Loads CSV files and extracts data previews.
- **Machine Learning Models**:
  - Isolation Forest for anomaly detection.
  - K-Means clustering for data segmentation.
- **Validation Rule Extraction**: Utilizes OpenAI's LLM to generate profiling rules dynamically.
- **Data Validation**: Applies generated rules to validate dataset fields.
- **Remediation Suggestions**: Provides recommendations for flagged anomalies.
- **Excel Report Generation**: Saves the results, including anomalies, clusters, validation results, and profiling rules, into an Excel file.

## Prerequisites
### Dependencies
Ensure the following Python libraries are installed:
```bash
pip install pandas scikit-learn openai
```
### API Key
- You need an OpenAI API key to use LLM-based profiling.
- Replace `api_key` in the script with your actual API key.

## Usage
### 1. Load Data
Specify the file path of your CSV data:
```python
file_path = "C:\\Users\\YourUser\\Downloads\\Sample_H2_Data.csv"
output_path = "C:\\Users\\YourUser\\Downloads\\ValidationResults_H2.xlsx"
```
### 2. Run the Script
Execute the script to process the data:
```bash
python compliance_script.py
```
### 3. Output
The script generates:
- **Anomalies Report**: Detected outliers in numerical data.
- **Clustered Data**: Segmented numerical data based on K-Means.
- **Validation Results**: Checks data against extracted profiling rules.
- **Remediation Suggestions**: Actionable insights for flagged anomalies.
- **Validation Rules**: Auto-generated rules extracted from the dataset.

## Functions
### Data Extraction
- `load_csv(file_path)`: Reads the CSV file.
- `extract_data(file_path)`: Extracts preview data from CSV.

### LLM Integration
- `query_llm(prompt, api_key)`: Queries OpenAI's GPT model for regulatory interpretation.
- `extract_profiling_rules(df)`: Uses LLM to extract profiling rules.

### Machine Learning Models
- `detect_anomalies(df)`: Uses Isolation Forest to detect anomalies.
- `cluster_data(df, n_clusters)`: Applies K-Means clustering.

### Validation & Remediation
- `validate_data(df, rules)`: Validates data against extracted rules.
- `suggest_remediation(anomalies)`: Suggests actions for flagged records.

### Output & Reporting
- `save_results_to_excel(anomalies, clusters, validations, remediations, rules, output_path)`: Saves results into an Excel file.

## Example Output
```
Validation Rules:
---------------------------------
| Field No | Field Name | Allowable Values |
|---------|-----------|----------------|
| 1       | Amount    | x >= 0        |
| 2       | Date      | String/Text   |

Anomalies Detected:
---------------------------------
| Index | Amount | Date  | Anomaly |
|-------|--------|-------|---------|
| 10    | -500   | 2023-01-02 | -1 |
```

## Notes
- Ensure the dataset has valid numerical values to perform anomaly detection and clustering.
- The OpenAI API may have rate limits; handle requests accordingly.


