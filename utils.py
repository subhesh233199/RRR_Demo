"""
This module provides utility functions used throughout the application.
It includes functions for:
- File path handling
- PDF processing
- Image conversion
- Report enhancement
- LLM-based evaluation

These utilities support the main analysis functionality by providing
common operations needed across different parts of the application.
"""

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
    """
    Converts an image file to base64 string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded string of the image, or empty string if conversion fails
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")
        return ""

def convert_windows_path(path: str) -> str:
    """
    Converts Windows path format to standard format.
    
    Args:
        path (str): Windows-style path
        
    Returns:
        str: Standardized path format
    """
    path = path.replace('\\', '/')
    path = path.replace('//', '/')
    return path

def get_pdf_files_from_folder(folder_path: str) -> List[str]:
    """
    Gets list of PDF files from a folder.
    
    Args:
        folder_path (str): Path to the folder
        
    Returns:
        List[str]: List of PDF file paths
        
    Raises:
        ValueError: If folder doesn't exist or contains no PDFs
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")
        
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise ValueError(f"No PDF files found in folder: {folder_path}")
        
    return [os.path.join(folder_path, f) for f in pdf_files]

def extract_hyperlinks_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    """
    Extracts hyperlinks from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        List[Dict[str, str]]: List of dictionaries containing link information
            Each dict has keys:
            - 'text': Link text
            - 'url': Link URL
            - 'page': Page number
    """
    hyperlinks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                for link in page.hyperlinks:
                    hyperlinks.append({
                        'text': link.get('text', ''),
                        'url': link.get('url', ''),
                        'page': page_num
                    })
    except Exception as e:
        logging.error(f"Error extracting hyperlinks from {pdf_path}: {str(e)}")
    return hyperlinks

def enhance_report_markdown(md_text: str) -> str:
    """
    Enhances markdown report with better formatting.
    
    Args:
        md_text (str): Original markdown text
        
    Returns:
        str: Enhanced markdown text
    """
    # Add any markdown enhancements here
    return md_text

def evaluate_with_llm_judge(source_text: str, generated_report: str) -> dict:
    """
    Evaluates the quality of the generated report using LLM.
    
    This function uses an LLM to evaluate the generated report against
    the source text, checking for accuracy, completeness, and quality.
    
    Args:
        source_text (str): Original text from PDFs
        generated_report (str): Generated analysis report
        
    Returns:
        dict: Evaluation results containing:
            - accuracy_score: Score for factual accuracy
            - completeness_score: Score for coverage of information
            - quality_score: Score for overall report quality
            - feedback: Detailed feedback on the report
    """
    try:
        llm = AzureChatOpenAI(
            deployment_name=os.getenv('DEPLOYMENT_NAME'),
            openai_api_version=os.getenv('AZURE_API_VERSION'),
            openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            temperature=0.1
        )
        
        prompt = f"""
        Evaluate the following generated report against the source text.
        Provide scores (0-100) for:
        1. Accuracy: How well does it reflect the source data?
        2. Completeness: How much of the important information is covered?
        3. Quality: How well is it written and structured?
        
        Also provide specific feedback on strengths and areas for improvement.
        
        Source Text:
        {source_text[:1000]}...
        
        Generated Report:
        {generated_report}
        
        Format your response as JSON with these fields:
        {{
            "accuracy_score": <score>,
            "completeness_score": <score>,
            "quality_score": <score>,
            "feedback": "<detailed feedback>"
        }}
        """
        
        response = llm.invoke(prompt)
        
        def extract_score(label: str, default: int = 0) -> int:
            """
            Extracts a score from the LLM response.
            
            Args:
                label (str): Score label to extract
                default (int): Default value if extraction fails
                
            Returns:
                int: Extracted score or default value
            """
            try:
                match = re.search(f'"{label}":\s*(\d+)', response)
                return int(match.group(1)) if match else default
            except:
                return default
        
        return {
            "accuracy_score": extract_score("accuracy_score"),
            "completeness_score": extract_score("completeness_score"),
            "quality_score": extract_score("quality_score"),
            "feedback": response
        }
    except Exception as e:
        logging.error(f"Error in LLM evaluation: {str(e)}")
        return {
            "accuracy_score": 0,
            "completeness_score": 0,
            "quality_score": 0,
            "feedback": f"Evaluation failed: {str(e)}"
        }