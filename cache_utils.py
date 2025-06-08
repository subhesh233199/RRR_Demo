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
from models import AnalysisResponse
from shared_state import shared_state
from app_logging import logger

CACHE_TTL_SECONDS = 3 * 24 * 60 * 60  # 3 days



# SQLite database setup
def init_cache_db():
    """
    Initializes SQLite database for caching analysis results.
    
    Creates:
    - report_cache table with columns:
        - folder_path_hash (TEXT)
        - pdfs_hash (TEXT)
        - report_json (TEXT)
        - created_at (INTEGER)
    """
    conn = sqlite3.connect('cache.db')
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS report_cache (
            folder_path_hash TEXT PRIMARY KEY,
            pdfs_hash TEXT NOT NULL,
            report_json TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def hash_string(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def hash_pdf_contents(pdf_files: List[str]) -> str:
    hasher = hashlib.md5()
    for pdf_path in sorted(pdf_files):
        try:
            with open(pdf_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        except Exception as e:
            logger.error(f"Error hashing PDF {pdf_path}: {str(e)}")
            raise
    return hasher.hexdigest()

def get_cached_report(folder_path_hash: str, pdfs_hash: str) -> Union[AnalysisResponse, None]:
    """
    Retrieves cached analysis results if available and not expired.
    
    Args:
        folder_path_hash (str): Hash of folder path
        pdfs_hash (str): Hash of PDF contents
        
    Returns:
        Union[AnalysisResponse, None]: Cached results or None
    """
    try:
        conn = sqlite3.connect('cache.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT report_json, created_at
            FROM report_cache
            WHERE folder_path_hash = ? AND pdfs_hash = ?
        ''', (folder_path_hash, pdfs_hash))
        result = cursor.fetchone()
        conn.close()

        if result:
            report_json, created_at = result
            current_time = int(time.time())
            if current_time - created_at < CACHE_TTL_SECONDS:
                report_dict = json.loads(report_json)
                return AnalysisResponse(**report_dict)
            else:
                with shared_state.lock:
                    conn = sqlite3.connect('cache.db')
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM report_cache WHERE folder_path_hash = ?', (folder_path_hash,))
                    conn.commit()
                    conn.close()
        return None
    except Exception as e:
        logger.error(f"Error retrieving cached report: {str(e)}")
        return None
    
def store_cached_report(folder_path_hash: str, pdfs_hash: str, response: AnalysisResponse):
    """
    Stores analysis results in cache.
    
    Args:
        folder_path_hash (str): Hash of folder path
        pdfs_hash (str): Hash of PDF contents
        response (AnalysisResponse): Analysis results to cache
    """
    try:
        report_json = json.dumps(response.dict())
        current_time = int(time.time())
        with shared_state.lock:
            conn = sqlite3.connect('cache.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO report_cache (folder_path_hash, pdfs_hash, report_json, created_at)
                VALUES (?, ?, ?, ?)
            ''', (folder_path_hash, pdfs_hash, report_json, current_time))
            conn.commit()
            conn.close()
        logger.info(f"Cached report for folder_path_hash: {folder_path_hash}")
    except Exception as e:
        logger.error(f"Error storing cached report: {str(e)}")

def cleanup_old_cache():
    try:
        current_time = int(time.time())
        with shared_state.lock:
            conn = sqlite3.connect('cache.db')
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM report_cache
                WHERE created_at < ?
            ''', (current_time - CACHE_TTL_SECONDS,))
            deleted_rows = cursor.rowcount
            conn.commit()
            conn.close()
        logger.info(f"Cleaned up old cache entries, deleted {deleted_rows} rows")
    except Exception as e:
        logger.error(f"Error cleaning up old cache entries: {str(e)}")