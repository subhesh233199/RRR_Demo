# RRR Release Analysis Tool

## Overview

The **RRR Release Analysis Tool** is a full-stack application for automated analysis of software release readiness reports (PDFs). It leverages AI (CrewAI, OpenAI/Azure LLMs), FastAPI, and a modern web frontend to extract, analyze, visualize, and summarize key metrics from release documents for TM and WST products.

The tool provides:
- Automated extraction and structuring of metrics from PDF reports
- AI-generated executive and brief summaries
- Visualizations of trends and key metrics
- Caching for fast repeated analysis
- A user-friendly web interface for uploading folders, viewing results, and downloading reports

---

## Features

- **AI-Powered Analysis**: Uses LLMs to extract, clean, and analyze metrics from unstructured PDF tables.
- **Multi-Product Support**: Handles both TM and WST product report formats.
- **Caching**: Avoids redundant computation by caching results based on file content.
- **Visualizations**: Generates trend charts and grouped bar charts for all key metrics.
- **Quality Evaluation**: Uses an LLM judge to score the quality of the generated report.
- **Web Frontend**: Modern, responsive UI for easy interaction and report download.

---

## Project Structure

```
.
├── app_logging.py           # Centralized logging configuration
├── cache_utils.py           # SQLite-based caching logic for analysis results
├── main.py                  # FastAPI backend, API endpoints, and app entry point
├── models.py                # Pydantic models for request/response validation
├── shared_state.py          # Thread-safe shared state for cross-component data
├── wst_product_config.py    # WST product-specific extraction, analysis, and crew setup
├── tm_product_config.py     # TM product-specific extraction, analysis, and crew setup
├── utils.py                 # Utility functions (file handling, PDF, markdown, etc.)
├── frontend.html            # Web UI (Bootstrap, JS, Markdown rendering)
└── requirements.txt         # (Recommended: add this file for dependencies)
```

---

## Requirements

- **Python 3.12.4** (recommended for full compatibility)
- See `requirements.txt` for Python package dependencies

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Install Python 3.12.4

Ensure you are using Python 3.12.4. You can check your version with:

```bash
python --version
```

If you need to install Python 3.12.4, download it from [python.org](https://www.python.org/downloads/release/python-3124/).

### 3. Install Python Dependencies

Create a `requirements.txt` with the following (if not present):

```txt
fastapi
uvicorn
pydantic
python-dotenv
PyPDF2
pdfplumber
matplotlib
numpy
tenacity
crewAI
langchain-openai
```

Then install:

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file in the root directory with your Azure OpenAI credentials:

```
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_API_VERSION=2023-05-15
DEPLOYMENT_NAME=your-deployment
```

### 5. Run the Backend

```bash
python main.py
```

The FastAPI server will start at `http://127.0.0.1:8080`.

### 6. Launch the Frontend

Open `frontend.html` in your web browser.  
**Note:** The frontend expects the backend to be running on `localhost:8080`.

---

## Usage

1. **Select Product**: Choose TM or WST.
2. **Enter Folder Path**: Provide the path to a folder containing your PDF reports.
3. **(Optional) Clear Cache**: Check to force re-analysis.
4. **Analyze**: Click "Analyze" to start processing.
5. **View Results**: See executive summary, brief summary, visualizations, and evaluation.
6. **Edit/Download**: Edit summaries or download as HTML/PDF.

---

## File Descriptions

- **main.py**: FastAPI app, `/analyze` endpoint, product routing, and cache integration.
- **models.py**: Pydantic models for request validation and response structure.
- **cache_utils.py**: Handles SQLite caching, hash generation, and cache cleanup.
- **utils.py**: File/path utilities, PDF file listing, image encoding, markdown enhancement, and LLM-based evaluation.
- **shared_state.py**: Thread-safe global state for metrics, report parts, and visualization status.
- **tm_product_config.py**: TM-specific extraction, AI crew setup, fallback visualization, and analysis logic.
- **st_product_config.py**: WST-specific extraction, AI crew setup, fallback visualization, and analysis logic.
- **app_logging.py**: Centralized logging configuration for the whole app.
- **frontend.html**: Bootstrap-based web UI, with JS for API calls, markdown rendering, and downloads.

---

## Caching

- Results are cached based on a hash of the folder path and PDF file contents.
- Cache is stored in `cache.db` (SQLite) and expires after 3 days.
- Use the "Clear Cache" option to force re-analysis.

---

## Extending

- To add new product types, create a new `*_product_config.py` and update the routing logic in `main.py`.
- To add new metrics or change report structure, update the relevant product config and models.

---

## Troubleshooting

- **No PDFs found**: Ensure the folder path is correct and contains `.pdf` files.
- **API errors**: Check the backend logs for details.
- **LLM errors**: Ensure your Azure OpenAI credentials are correct and you have access to the required deployment.

---

## License

MIT License (or specify your license here)

---

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [LangChain](https://github.com/hwchase17/langchain)
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- [Bootstrap](https://getbootstrap.com/)

---

**For questions or contributions, please open an issue or pull request!** 