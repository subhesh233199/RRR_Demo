"""
This module defines Pydantic models for request/response validation and data structures.
These models ensure type safety and data validation throughout the application.

The models include:
- FolderPathRequest: For validating folder path requests
- AnalysisResponse: For structuring analysis results
- MetricItem: For representing individual metric data points
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

class FolderPathRequest(BaseModel):
    """
    Pydantic model for validating folder path requests.
    
    This model ensures that incoming requests contain valid folder paths
    and optional parameters for cache control and product selection.
    
    Attributes:
        folder_path (str): Path to the folder containing PDF reports.
            Must be a non-empty string.
            
        clear_cache (bool): Whether to clear the cache before analysis.
            Defaults to False.
            
        product (str): Which product pipeline to run.
            Can be either "TM" or "WST". Defaults to "TM".
    
    Validators:
        validate_folder_path: Ensures folder path is not empty
    """
    folder_path: str
    clear_cache: bool = False
    product: str = "TM"  # Default to TM product pipeline

    @validator('folder_path')
    def validate_folder_path(cls, v):
        """
        Validates that the folder path is not empty.
        
        Args:
            v (str): The folder path to validate
            
        Returns:
            str: The validated folder path
            
        Raises:
            ValueError: If the folder path is empty
        """
        if not v:
            raise ValueError('Folder path cannot be empty')
        return v

class AnalysisResponse(BaseModel):
    """
    Pydantic model for structuring analysis responses.
    
    This model defines the structure of the response returned after
    analyzing PDF reports. It includes metrics, visualizations, report
    content, evaluation results, hyperlinks, and a brief summary.
    
    Attributes:
        metrics (Dict): Processed metrics data extracted from PDFs.
            Contains all analyzed metrics with their values and trends.
            
        visualizations (List[str]): Base64 encoded visualization images.
            Each string represents a chart or graph generated from the metrics.
            
        report (str): Generated markdown report.
            Contains the full analysis report in markdown format.
            
        evaluation (Dict): Quality evaluation of the analysis.
            Contains scores and feedback on the analysis quality.
            
        hyperlinks (List[Dict]): Extracted hyperlinks from PDFs.
            Each dictionary contains link information from the source PDFs.
            
        brief_summary (str): Brief summary of the analysis.
            A concise overview of the key findings.
    """
    metrics: Dict
    visualizations: List[str]
    report: str
    evaluation: Dict
    hyperlinks: List[Dict]
    brief_summary: str

class MetricItem(BaseModel):
    """
    Pydantic model for representing individual metric data points.
    
    This model structures the data for individual metrics, including
    version information, values, status, and trends.
    
    Attributes:
        version (str): The version number this metric applies to.
            Format: "X.Y" (e.g., "25.1")
            
        value (Union[float, str]): The metric value.
            Can be a number or string depending on the metric type.
            
        status (str): The status of this metric.
            One of: "ON TRACK", "MEDIUM RISK", "RISK", "NEEDS REVIEW"
            
        trend (Union[str, None]): The trend indicator.
            Format: "↑ (X%)", "↓ (X%)", or "→"
            Defaults to None if not calculated.
    """
    version: str
    value: Union[float, str]
    status: str
    trend: Union[str, None] = None



