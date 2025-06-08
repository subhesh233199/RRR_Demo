import os
import re
import json
import runpy
import base64
import sqlite3
import hashlib
import time
from typing import List, Dict, Tuple, Any, Union
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from crewai import Agent, Task, Crew, Process, LLM
from langchain_openai import AzureChatOpenAI
import ssl
import warnings
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tenacity import retry, stop_after_attempt, wait_fixed
from copy import deepcopy
import pdfplumber
from crewai import Agent, Task, Crew, Process  # Ensure imports in function scope for clarity
from shared_state import shared_state
from app_logging import logger
import base64
from io import BytesIO

# Now use as usual:
logger.info("Starting Task Management crew setup.")
logger.error("Something failed in parsing.")

def get_base64_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error reading image {image_path}: {str(e)}")
        return ""

def convert_windows_path(path: str) -> str:
    path = path.replace('\\', '/')
    path = path.replace('//', '/')
    return path

def get_pdf_files_from_folder(folder_path: str) -> List[str]:
    """
    Retrieves all PDF files from specified folder.
    
    Args:
        folder_path (str): Path to folder containing PDFs
        
    Returns:
        List[str]: List of full paths to PDF files
        
    Raises:
        FileNotFoundError: If folder doesn't exist or no PDFs found
    """
    pdf_files = []
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")
   
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.pdf'):
            full_path = os.path.join(folder_path, file_name)
            pdf_files.append(full_path)
   
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in the folder {folder_path}.")
   
    return pdf_files

def extract_hyperlinks_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    """
    Extracts hyperlinks and their context from PDF.
    
    Args:
        pdf_path (str): Path to PDF file
        
    Returns:
        List[Dict[str, str]]: List of hyperlink information
    """
    hyperlinks = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page_num, page in enumerate(reader.pages, start=1):
                if '/Annots' in page:
                    for annot in page['/Annots']:
                        annot_obj = annot.get_object()
                        if annot_obj['/Subtype'] == '/Link' and '/A' in annot_obj:
                            uri = annot_obj['/A']['/URI']
                            text = page.extract_text() or ""
                            context_start = max(0, text.find(uri) - 50)
                            context_end = min(len(text), text.find(uri) + len(uri) + 50)
                            context = text[context_start:context_end].strip()
                            hyperlinks.append({
                                "url": uri,
                                "context": context,
                                "page": page_num,
                                "source_file": os.path.basename(pdf_path)
                            })
    except Exception as e:
        logger.error(f"Error extracting hyperlinks from {pdf_path}: {str(e)}")
    return hyperlinks


# def enhance_report_markdown(md_text):
#     # Remove markdown code fences
#     cleaned = re.sub(r'^```markdown\n|\n```$', '', md_text, flags=re.MULTILINE)
#     cleaned = re.sub(r'(\|.+\|)\n\s*(\|-+\|)', r'\1\n\2', cleaned)
#     cleaned = re.sub(r'\b[4t/]\b', '→', cleaned)
#     cleaned = re.sub(r'\s*\|\s*', ' | ', cleaned)
#     cleaned = re.sub(r'[ ]{2,}', ' ', cleaned)

#     status_map = {
#         "MEDIUM RISK": "**MEDIUM RISK**",
#         "HIGH RISK": "**HIGH RISK**",
#         "LOW RISK": "**LOW RISK**",
#         "ON TRACK": "**ON TRACK**"
#     }
#     for k, v in status_map.items():
#         cleaned = cleaned.replace(k, v)

#     cleaned = re.sub(r'^#\s+(.+)$', r'# \1\n', cleaned, flags=re.MULTILINE)
#     cleaned = re.sub(r'^##\s+(.+)$', r'## \1\n', cleaned, flags=re.MULTILINE)
#     cleaned = re.sub(r'^\s*-\s+(.+)', r'- \1', cleaned, flags=re.MULTILINE)
#     cleaned = re.sub(r'\s*(### )', r'\n\n\1', cleaned)
#     cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
#     cleaned = cleaned.strip()

#     # --- THIS PATCH WILL RESTRUCTURE ALL TABLE BLOBS AFTER EACH "### ..." HEADER ---
#     def fix_all_tables(md):
#         result = []
#         lines = md.splitlines()
#         i = 0
#         while i < len(lines):
#             line = lines[i]
#             if line.startswith('### '):
#                 # Add heading
#                 result.append(line)
#                 i += 1
#                 # Collect table blob (might be all one line)
#                 table_blob = []
#                 # Skip blank lines after heading
#                 while i < len(lines) and lines[i].strip() == '':
#                     result.append('')
#                     i += 1
#                 # If table blob is glued in one line, or spans multiple lines
#                 while i < len(lines) and lines[i].lstrip().startswith('|'):
#                     table_blob.append(lines[i].strip())
#                     i += 1
#                 # If table blob is just one long line, split into rows
#                 if len(table_blob) == 1:
#                     row_cells = [p.strip() for p in table_blob[0].split('|') if p.strip()]
#                     # Detect columns from header row (usually the first row after heading)
#                     # Find possible column counts (usually 4 or 5)
#                     # Try 4 and 5, pick the one that matches best with markdown table format
#                     possible_cols = []
#                     for col_count in range(3, 8):
#                         if len(row_cells) % col_count == 0:
#                             possible_cols.append(col_count)
#                     if possible_cols:
#                         ncol = possible_cols[-1]
#                     else:
#                         ncol = 4  # fallback
#                     # Now split into lines
#                     for j in range(0, len(row_cells), ncol):
#                         result.append('| ' + ' | '.join(row_cells[j:j+ncol]) + ' |')
#                 else:
#                     # Already line-split table
#                     result.extend(table_blob)
#             else:
#                 result.append(line)
#                 i += 1
#         return '\n'.join(result)

#     cleaned = fix_all_tables(cleaned)
#     cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
#     cleaned = cleaned.strip()
#     # DEBUG: Uncomment to print what gets sent to the frontend
#     print("\n----MARKDOWN DEBUG OUTPUT----\n")
#     print(cleaned[1350:3390])
#     print("\n----------------------------\n")
#     return cleaned.encode('utf-8').decode('utf-8')
def enhance_report_markdown(md_text):
    return md_text
def evaluate_with_llm_judge(source_text: str, generated_report: str) -> dict:
    judge_llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_deployment=os.getenv("DEPLOYMENT_NAME"),
        temperature=0,
        max_tokens=512,
        timeout=None,
    )
   
    prompt = f"""Act as an impartial judge evaluating report quality. You will be given:
1. ORIGINAL SOURCE TEXT (extracted from PDF)
2. GENERATED REPORT (created by AI)

Evaluate based on:
- Data accuracy (50% weight): Does the report correctly reflect the source data?
- Analysis depth (30% weight): Does it provide meaningful insights?
- Clarity (20% weight): Is the presentation clear and professional?

ORIGINAL SOURCE:
{source_text}

GENERATED REPORT:
{generated_report}

INSTRUCTIONS:
1. For each category, give a score (integer) out of its maximum:
    - Data accuracy: [0-50]
    - Analysis depth: [0-30]
    - Clarity: [0-20]
2. Add up to a TOTAL out of 100.
3. Give a brief 2-3 sentence evaluation.
4. Use EXACTLY this format:
Data accuracy: [0-50]
Analysis depth: [0-30]
Clarity: [0-20]
TOTAL: [0-100]
Evaluation: [your evaluation]

Your evaluation:"""

    try:
        response = judge_llm.invoke(prompt)
        response_text = response.content

        # Robust extraction: matches label anywhere on line, any case, extra spaces, "35/50" or "35"
        def extract_score(label, default=0):
            regex = re.compile(rf"{label}\s*:\s*(\d+)", re.IGNORECASE)
            for line in response_text.splitlines():
                match = regex.search(line)
                if match:
                    return int(match.group(1))
            return default

        data_accuracy = extract_score("Data accuracy", 0)
        analysis_depth = extract_score("Analysis depth", 0)
        clarity = extract_score("Clarity", 0)
        total = extract_score("TOTAL", data_accuracy + analysis_depth + clarity)

        # Extract evaluation: combine lines after "Evaluation:" or the last non-score line
        evaluation = ""
        eval_regex = re.compile(r"evaluation\s*:\s*(.*)", re.IGNORECASE)
        found_eval = False
        for line in response_text.splitlines():
            match = eval_regex.match(line)
            if match:
                evaluation = match.group(1).strip()
                found_eval = True
                break
        # If not found, fallback: concatenate all lines not containing a score label
        if not found_eval:
            non_score_lines = [
                l for l in response_text.splitlines()
                if not any(lbl in l.lower() for lbl in ["data accuracy", "analysis depth", "clarity", "total"])
            ]
            evaluation = " ".join(non_score_lines).strip()

        return {
            "data_accuracy": data_accuracy,
            "analysis_depth": analysis_depth,
            "clarity": clarity,
            "total": total,
            "text": evaluation
        }
    except Exception as e:
        logger.error(f"Error parsing judge response: {e}\nResponse was:\n{locals().get('response_text', '')}")
        return {
            "data_accuracy": 0,
            "analysis_depth": 0,
            "clarity": 0,
            "total": 0,
            "text": "Could not parse evaluation"
        }