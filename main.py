"""
This module implements the main FastAPI application for the RRR Release Analysis Tool.
It provides endpoints for analyzing PDF reports and managing the analysis process.

The application supports:
- PDF analysis with caching
- Multiple product pipelines (TM and WST)
- Visualization generation
- Report generation
- Health checks

The module uses FastAPI for the web framework and implements proper error handling,
logging, and thread safety throughout the analysis process.
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
from crewai import Agent, Task, Crew, Process
from shared_state import shared_state
from app_logging import logger
from tm_product_config import (process_task_output
                               ,validate_metrics
                               ,clean_json_output
                               ,setup_crew_tm
                               ,run_fallback_visualization
                               ,extract_section_from_pdf
                               ,validate_report
                               ,run_full_analysis_tm)
from wst_product_config import (
    extract_tables_text_for_version,
    table_indices,
    setup_crew_wst,
    run_full_analysis_wst,
)
from models import (FolderPathRequest
                    ,AnalysisResponse 
                    , MetricItem)
from utils import (convert_windows_path
                   ,get_base64_image
                   ,get_pdf_files_from_folder
                   ,extract_hyperlinks_from_pdf
                   ,enhance_report_markdown
                   ,evaluate_with_llm_judge)
from cache_utils import (init_cache_db
                         ,hash_string
                         ,hash_pdf_contents
                         ,get_cached_report
                         ,store_cached_report
                         ,cleanup_old_cache)

# Configure logging
logger.info("Starting Task Management crew setup.")
logger.error("Something failed in parsing.")

# Disable SSL verification for development
ssl._create_default_https_context = ssl._create_unverified_context

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RRR Release Analysis Tool",
    description="API for analyzing release readiness reports"
)

os.makedirs("visualizations", exist_ok=True)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize cache database
init_cache_db()

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_pdfs(request: FolderPathRequest):
    """
    Analyzes PDF reports in the specified folder.
    
    This endpoint processes PDF files in the given folder path, extracts metrics,
    generates visualizations, and produces a comprehensive analysis report.
    
    The analysis process includes:
    1. Cache check/cleanup
    2. PDF file processing
    3. Metrics extraction
    4. Visualization generation
    5. Report generation
    6. Result caching
    
    Args:
        request (FolderPathRequest): Request containing folder path and options
        
    Returns:
        AnalysisResponse: Analysis results including metrics, visualizations,
            and report
            
    Raises:
        HTTPException: If folder path is invalid or processing fails
    """
    try:
        if request.clear_cache:
            cleanup_old_cache()

        folder_path = convert_windows_path(request.folder_path)
        folder_path = os.path.normpath(folder_path)
        folder_path_hash = hash_string(folder_path)
        pdf_files = get_pdf_files_from_folder(folder_path)
        pdfs_hash = hash_pdf_contents(pdf_files)
        logger.info(f"Computed hashes - folder_path_hash: {folder_path_hash}, pdfs_hash: {pdfs_hash}")

        if not request.clear_cache:
            cached_response = get_cached_report(folder_path_hash, pdfs_hash)
            if cached_response:
                logger.info(f"Cache hit for folder_path_hash: {folder_path_hash}")
                return cached_response

        logger.info(f"Cache miss for folder_path_hash: {folder_path_hash} or cache clear requested, running full analysis")

        # Product routing logic
        product = getattr(request, "product", "TM").upper()  # Default to TM if missing

        if product == "TM":
            logger.info("Routing to TM pipeline")
            response = await run_full_analysis_tm(request)
        elif product == "WST":
            logger.info("Routing to WST pipeline")
            response = await run_full_analysis_wst(request)
        else:
            logger.error(f"Unsupported product: {product}")
            raise HTTPException(status_code=400, detail=f"Unsupported product: {product}")

        store_cached_report(folder_path_hash, pdfs_hash, response)
        return response

    except Exception as e:
        logger.error(f"Error in /analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        plt.close('all')  # Clean up matplotlib resources

# Mount static files directory for visualizations
app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Status indicating the service is healthy
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)