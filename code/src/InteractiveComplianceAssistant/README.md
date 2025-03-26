# Compliance Assistant - Interactive Data Validation Tool

## Overview
This project is an **Interactive Compliance Assistant** built using **Streamlit**, **OpenAI's GPT-4o**, and **Pandas**. It allows users to upload an Excel dataset, define profiling rules for data validation, apply them to detect anomalies, and receive AI-powered rule suggestions.

## Features
✅ Upload and preview Excel datasets  
✅ Define, modify, and delete profiling rules  
✅ Validate data based on user-defined rules  
✅ Highlight flagged records that fail validation  
✅ Generate AI-powered rule suggestions using OpenAI LLM  
✅ User-friendly **Streamlit** interface  

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- pip (Python package manager)

### Install Dependencies
```sh
pip install pandas openai streamlit
```

## Usage
### 1️⃣ Run the Application
Execute the following command:
```sh
streamlit run interactiveCompliance.py
```

### 2️⃣ Upload a Dataset
- Click **"Upload an Excel file"** and select a `.xlsx` file.
- The dataset preview will be displayed.

### 3️⃣ Define Profiling Rules
- Select a column from the dropdown.
- Enter a validation rule (e.g., `x >= 0`).
- Click **"Add Rule"** to save the rule.
- View current rules in JSON format.

### 4️⃣ Modify or Delete Rules
- Choose an existing rule from the dropdown.
- Modify or delete it using the respective buttons.

### 5️⃣ Validate Data
- Click **"Validate Data"** to check records against rules.
- If any data violates a rule, it will be flagged.
- A table of flagged records is displayed.

### 6️⃣ AI-Powered Rule Suggestions
- Click **"Suggest Rules"** to get AI-generated validation rules.
- Enter your **OpenAI API Key** (secure input field).
- The assistant suggests validation rules for your dataset.


## Future Enhancements
🔹 Add support for **multiple sheets** in Excel files  
🔹 Implement **automatic rule learning** based on past validations  
🔹 Integrate **more ML models** for anomaly detection  

