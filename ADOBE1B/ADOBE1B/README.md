# Adobe1B - PDF Document Processing Application

This application processes PDF documents using sentence transformers and TF-IDF to extract relevant sections based on a given persona and job description.

## Features

- Extracts and ranks PDF pages based on semantic similarity to query embeddings
- Uses sentence transformers for semantic understanding
- Applies TF-IDF scoring for keyword relevance
- Supports containerized deployment for offline execution
- Pre-caches machine learning models during build time

## Files

- `main_code.py` - Main application script
- `cache_models.py` - Script to pre-download and cache ML models
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container build instructions
- `.dockerignore` - Files to exclude from Docker context

## Core Methodology

- **LLM Embeddings**:  
  Uses [`all-MiniLM-L6-v2`](https://www.sbert.net/docs/pretrained_models.html) (~80MB) from SentenceTransformers to generate dense semantic vectors of both the task (persona + job) and the PDF page contents. This allows us to measure how contextually relevant each page is, even if it uses different wording.

- **TF-IDF Hybrid Scoring**:  
  Along with semantic similarity, we compute a classical term frequency-inverse document frequency score to emphasize important keywords. Final scores combine both metrics for better relevance.

- **Diverse Query Prompts**:  
  Instead of using just one static query, we generate 3–5 variant prompts using templates derived from the persona and job. This improves robustness by allowing the model to match text against different expressions of intent.

- **Smart Page Filtering**:  
  Before scoring, we filter out empty, blank, or nearly-empty pages using token length and whitespace density checks, ensuring only meaningful content is processed.

- **Heading Heuristics**:  
  We extract likely section titles using rules such as: all-uppercase text, bold large font (if available), or title-case short lines at the top of the page. This helps identify section boundaries without relying on layout data.

- **Clean Refined Text**:  
  Final extracted content is cleaned to remove non-informative characters (like bullet symbols •, page numbers, headers/footers) using regex and structure-aware parsing. This produces more natural output for downstream use.


## Building the Docker Image (power shell)

```power shell
docker build -t pdf-processor .
```

## Running the Container

### Using Docker Run (power shell)

```power shell
# Mount input and output directories
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none pdf-processor
```

### Using Custom Input/Output Paths

```bash
docker run -v /path/to/input:/custom/input -v /path/to/output:/custom/output pdf-processor --input-dir /custom/input --output-dir /custom/output
```

## Input Requirements

The input directory must contain:
- `input.json` - Metadata file with document list, persona, and job description
- PDF files referenced in the metadata

### Example input.json structure:

```json
{
    "documents": [
        {
            "filename": "document1.pdf",
            "title": "Document 1"
        }
    ],
    "persona": {
        "role": "HR professional"
    },
    "job_to_be_done": {
        "task": "Create and manage fillable forms for onboarding and compliance."
    }
}
```

## Output

The application generates an `output.json` file in the output directory containing:
- Extracted sections ranked by relevance
- Subsection analysis with cleaned text
- Processing metadata

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main_code.py --input-dir /path/to/input --output-dir /path/to/output
```

## Model Caching

The application uses the `all-MiniLM-L6-v2` sentence transformer model, which is automatically downloaded and cached during the Docker build process for offline execution.

## Testing with Sample Data

You can test the application using the sample data in the `Test/` directory:

```bash
python main_code.py --input-dir ./Test/input --output-dir ./Test/output
```
