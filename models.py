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

# Pydantic models
class FolderPathRequest(BaseModel):
    """
    Pydantic model for folder path validation.

    Attributes:
        folder_path (str): Path to the folder containing PDF reports
        clear_cache (bool): Whether to clear the cache before analysis (default: False)
        product (str): Which product pipeline to run ("TM" or "WST"), default "TM"
    Validators:
        validate_folder_path: Ensures folder path is not empty
    """
    folder_path: str
    clear_cache: bool = False
    product: str = "TM"  # <--- Add this line

    @validator('folder_path')
    def validate_folder_path(cls, v):
        if not v:
            raise ValueError('Folder path cannot be empty')
        return v
class AnalysisResponse(BaseModel):
    """
    Pydantic model for analysis response.
    
    Attributes:
        metrics (Dict): Processed metrics data
        visualizations (List[str]): Base64 encoded visualization images
        report (str): Generated markdown report
        evaluation (Dict): Quality evaluation of the analysis
        hyperlinks (List[Dict]): Extracted hyperlinks from PDFs
        brief_summary: Brief summary of the analysis
    """
    metrics: Dict
    visualizations: List[str]
    report: str
    evaluation: Dict
    hyperlinks: List[Dict]
    brief_summary: str

class MetricItem(BaseModel):
    version: str
    value: Union[float, str]
    status: str
    trend: Union[str, None] = None# (Add other models as needed)



