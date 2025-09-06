# RTGS-Style AI Analyst for Telangana Open Data

Branch: `agentic-ai`

## 📌 Overview
This repository implements a **CLI-first agentic workflow** that ingests public datasets from the [Telangana Open Data Portal](https://data.telangana.gov.in/), cleans and standardizes them into **analysis-ready form**, and generates **clear, evidence-backed insights** for policymakers.  

The goal is to prototype a **Real-Time Governance System (RTGS)** that can take raw government data and turn it into reliable evidence for decision-making.

---

## 🚀 Current Progress (as of 12 PM, Day 1)
- ✅ Repository initialized with branch `agentic-ai`
- ✅ Project structure defined (`src/`, `data/`, `outputs/`, `logs/`, `traces/`)
- ✅ **Ingestion script (`ingest.py`)** built to load and merge multiple monthly tanker datasets
- ✅ **Cleaning script (`clean.py`)** implemented for standardization and delivery gap calculation
- ✅ Initial dataset scope: **Hyderabad Water Tanker Deliveries (2024, 6 months)**
- ✅ Generated:
  - `tankers_raw_merged.csv` (merged dataset, 1078 rows)
  - `tankers_cleaned.csv` (cleaned dataset, consistent schema)
  - Profile + Data Quality reports in `outputs/`

---

## 📊 Dataset Manifest
- **Sector:** Urban / Water and Sanitation  
- **Dataset:** *Hyderabad Metropolitan Water Supply & Sewerage Board (HMWSSB) — Water Tanker Deliveries*  
- **Scope:** 6 months from 2024 — March, April, May, July, August, December  
- **Rows merged:** 1,078  
- **Key fields:** `year`, `month`, `division`, `section`, `noofbookings`, `delivered`
