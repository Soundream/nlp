# News NLP Processing Guide

## 1. Project Overview
This project aims to perform NLP analysis on news texts, focusing on the market impact assessment of companies mentioned in the news.

## 2. Data Input
- Input Format: JSON file
- Each news item contains the following fields:
  - date: News publication date
  - source: News source
  - title: News title
  - content: News content
  - tickers: List of related company tickers (e.g., ["STNG US", "GOGL NO"])
  - company_names: List of company names corresponding to tickers (e.g., ["Star Bulk", "Golden Ocean"])

### 2.1 Input File Format Example
```json
[
  {
    "date": "2024-01-01",
    "source": "Sina Finance",
    "title": "Company Launches New Product",
    "content": "Detailed news content...",
    "tickers": ["STNG US", "GOGL NO"],
    "company_names": ["Star Bulk", "Golden Ocean"]
  },
  {
    "date": "2024-01-02",
    "source": "East Money",
    "title": "Industry Analysis Report",
    "content": "Detailed news content...",
    "tickers": ["MS US", "NTRS US"],
    "company_names": ["Morgan Stanley", "Northern Trust"]
  }
]
```

## 3. Configuration Information

### 3.1 Environment Requirements
- Python Version: 3.10
- Required Packages:
  - pandas
  - numpy
  - transformers
  - torch
  - FinGPT

### 3.2 FinGPT Configuration
- Model: Use local FinGPT model
- Batch Size: Recommended 8
- Max Sequence Length: 512
- Device: GPU recommended

## 4. Processing Flow

### 4.1 Data Extraction and Company Filtering
1. Load company_info.csv to get all valid tickers
2. For each news item:
   - Extract tickers and company_names
   - Split tickers to get base ticker (e.g., "STNG" from "STNG US")
   - Compare with company_info.csv tickers
   - Keep only companies where base ticker exists in company_info.csv
   - If no matching companies found, skip this news item
   - Create filtered lists of tickers and company_names for matching companies

### 4.2 Company Market Impact Analysis
Using FinGPT to analyze news content, output market impact level (-2 to 2):

1. Market Impact Level (-2 to 2):
   - 2: Strongly Positive
     * Clear positive financial indicators (e.g., revenue growth >20%)
     * Major business expansion or strategic success
     * Significant competitive advantage gained
   - 1: Positive
     * Moderate positive developments
     * Minor business improvements
     * Favorable market conditions
   - 0: Neutral (use sparingly)
     * Only use when news is purely factual without clear sentiment
     * Company announcements without clear impact
     * General industry updates without specific implications
   - -1: Negative
     * Moderate negative developments
     * Minor business setbacks
     * Unfavorable market conditions
   - -2: Strongly Negative
     * Clear negative financial indicators (e.g., revenue decline >20%)
     * Major business setbacks or strategic failures
     * Significant competitive disadvantage

### 4.3 Output Files

#### 4.3.1 Market Impact File (market_impact.csv)
- Format: CSV
- Column Structure:
  - Date (date)
  - Source (source)
  - Ticker1 (e.g., STNG)
  - Ticker2 (e.g., GOGL)
  - ... (dynamically add new ticker columns)
- Content: Fill in market impact level (-2 to 2) for each company
- Note: Column names use tickers, not company names

#### 4.3.2 Company Mentions File (company_mentions.csv)
- Format: CSV
- Column Structure:
  - Date (date)
  - Source (source)
  - Ticker1 (e.g., STNG)
  - Ticker2 (e.g., GOGL)
  - ... (dynamically add new ticker columns)
- Content: 
  - 1: If company ticker is present in the filtered tickers list
  - 0: If company ticker is not present
- Note: Column names use tickers, not company names

## 5. Processing Rules

### 5.1 Market Impact Level Rules
- If news targets specific company, assign value only to that company
- If news targets entire industry, assign same value to all related companies
- If news contains both specific company and industry analysis, assign values separately
- Avoid using neutral (0) unless absolutely necessary
- Focus on concrete financial and business impacts rather than general sentiment
- When analyzing with FinGPT, use company_names for context but output using tickers

### 5.2 Company Ticker Processing
1. Ticker Validation:
   - Load all valid tickers from company_info.csv
   - For each news item's tickers:
     * Split to get base ticker (e.g., "STNG" from "STNG US")
     * Check if base ticker exists in company_info.csv
     * Keep only matching companies
   - Skip news items with no matching companies

2. Company Name and Ticker Mapping:
   - Maintain index correspondence between company_names and tickers
   - Use company_names for FinGPT analysis
   - Use tickers for output file column names

## 6. Technical Implementation Points
1. Use pandas for CSV file processing
2. Implement dynamic column expansion mechanism
3. Ensure consistent date format
4. Optimize batch processing with FinGPT
5. Add error handling and logging
6. Implement efficient ticker validation against company_info.csv
7. Maintain index correspondence between company_names and tickers

## 7. Output File Examples

### market_impact.csv
```
date,source,STNG,GOGL,MS,NTRS
2024-01-01,source1,2,0,-1
2024-01-02,source2,1,1,1
```

### company_mentions.csv
```
date,source,STNG,GOGL,MS,NTRS
2024-01-01,source1,1,0,1
2024-01-02,source2,1,1,1
```

## 8. Important Notes
1. Ensure FinGPT model stability
2. Pay attention to performance optimization when processing large data
3. Regularly backup processing results
4. Maintain processing logs
5. Implement exception handling mechanism
6. Strictly validate tickers against company_info.csv
7. Minimize use of neutral (0) impact level
8. Maintain proper mapping between company_names and tickers
9. Skip news items with no valid companies 