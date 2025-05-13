# Automated HTS Code Classifiation and Duty Rate Calculation

![Cover Image](codes_and_duties.jpg)

## Overview

**Automated Codes and Duties** is a Python-based system for automated classification of products according to the Harmonized Tariff Schedule (HTS) and calculation of applicable duties. It leverages LLMs, external APIs, and workflow automation to streamline customs and compliance processes.

## Features

- **Automated HTS Classification:** Uses LLM-powered agents to analyze product descriptions and select the most likely HTS codes.
- **Duty Rate Calculation:** Integrates with APIs (Tariffy, SimplyDuty) to fetch up-to-date duty rates for classified products.
- **Email Automation:** Automatically sends classification results and duty calculations to users via email.
- **Modular Agent Architecture:** Includes agents for chapter selection, code extraction, and multi-level code refinement.
- **API Service:** Exposes a FastAPI endpoint for programmatic classification and duty calculation.

## Project Structure

- `classification_and_duties_deploy.py`: Main API and workflow logic.
- `agents/`: Modular agent classes for each step of the classification process.
- `files/`: Data files for HTS codes and chapter descriptions.
- `requirements.txt`: Python dependencies.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set environment variables:**  
   Configure API keys for Google GenAI, Tariffy, SimplyDuty, Composio and Logfire in a `.env` file.

3. **Run the API server:**
   ```bash
   python classification_and_duties_deploy.py
   ```

## Usage

- **API Endpoint:**  
  `POST /classify`  
  To receive HTS classification and duty rates, submit product data with the following required fields:

  ```json
  "data": {
    "caller": {
      "email": "your email here",
    },
    "value": {
      "General Information": {
        "Invoice Number": "your unique id number here"},
        },
    "Items": [
        {
          "Description": "product description here",
          "Country of Origin": "country of origin here"
        },
    ]
    }
    ```
  
  You can have multiple items in the list.
  Results emailed to the caller email.