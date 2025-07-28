# SECTION 1: Setup and Imports
import os
import sys
import json
import re
import fitz  # PyMuPDF
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse

# SECTION 2: Utility Functions
def clean_text(text):
    text = re.sub(r'[•●\u2022]+', '', text)
    text = re.sub(r'\s{2,}', ' ', text.replace('\n', ' ')).strip()
    return text

def extract_heading_candidates(page_text):
    lines = page_text.split("\n")
    candidates = []
    
    # Strategy 1: Look for traditional heading patterns
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if not line:
            continue
            
        # Remove common PDF artifacts
        clean_line = re.sub(r'^\d+\s*', '', line)  # Remove leading numbers
        clean_line = re.sub(r'\s*\d+\s*$', '', clean_line)  # Remove trailing page numbers
        clean_line = clean_line.strip()
        
        if (len(clean_line) > 3 and len(clean_line) < 100 and 
            len(clean_line.split()) <= 12 and
            (clean_line.isupper() or clean_line.istitle() or 
             any(word[0].isupper() for word in clean_line.split() if word))):
            candidates.append(clean_line)
    
    # Strategy 2: Look for lines with specific formatting patterns
    for line in lines[:15]:
        line = line.strip()
        if not line:
            continue
            
        # Look for bold-like patterns or section indicators
        if (re.match(r'^[A-Z][a-zA-Z\s:.-]+$', line) and 
            len(line) > 5 and len(line) < 80 and
            len(line.split()) <= 10):
            candidates.append(line)
    
    # Strategy 3: Extract meaningful phrases from the beginning
    if not candidates:
        first_meaningful_lines = []
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 10 and len(line) < 150:
                # Clean up and extract first meaningful sentence
                clean_line = re.sub(r'^[^\w]*', '', line)
                clean_line = re.sub(r'[^\w\s:.-].*$', '', clean_line)
                if clean_line and len(clean_line.split()) >= 2:
                    first_meaningful_lines.append(clean_line[:80])
        
        if first_meaningful_lines:
            candidates.extend(first_meaningful_lines[:2])
    
    # Strategy 4: Fallback to document keywords
    if not candidates:
        # Look for document-specific keywords that might indicate content type
        doc_keywords = re.findall(r'\b(?:Learn|Guide|Tips|How to|Tutorial|Instructions|Overview|Introduction|Chapter|Section|Part)\s+[A-Za-z\s-]+\b', page_text[:500])
        if doc_keywords:
            candidates.append(doc_keywords[0][:60])
    
    # Final fallback with more context
    if not candidates:
        words = page_text.split()[:15]  # First 15 words
        if len(words) >= 3:
            candidates.append(' '.join(words))
        else:
            candidates.append("Document Content")
    
    # Return the best candidate, prioritizing shorter, cleaner titles
    if candidates:
        # Sort by length and cleanliness, prefer shorter titles
        candidates = sorted(set(candidates), key=lambda x: (len(x), x.lower()))
        return [candidates[0]]
    
    return ["Document Section"]

def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if len(text.strip()) < 50:  # Skip blank pages
            continue
        pages.append((i+1, text))
    doc.close()
    return pages

def parse_arguments():
    """Parse command line arguments for input and output directories."""
    parser = argparse.ArgumentParser(description='Process PDF documents for persona-based content extraction')
    
    # Default to current directory structure for local development, Docker will override these
    default_input = './input' if os.path.exists('./input') else '/app/input'
    default_output = './output' if os.path.exists('./output') else '/app/output'
    
    parser.add_argument('--input-dir', '-i', default=default_input, 
                       help='Input directory containing input.json and PDF files (default: ./input or /app/input)')
    parser.add_argument('--output-dir', '-o', default=default_output, 
                       help='Output directory for results (default: ./output or /app/output)')
    return parser.parse_args()

def main():
    """Main function to orchestrate the PDF processing pipeline."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        input_dir = args.input_dir
        output_dir = args.output_dir

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # SECTION 3: Load Metadata
        metadata_path = os.path.join(input_dir, "input.json")
        if not os.path.exists(metadata_path):
            print(f"Error: metadata file not found at {metadata_path}")
            sys.exit(1)

        with open(metadata_path) as f:
            metadata = json.load(f)

        input_docs = [doc["filename"] for doc in metadata["documents"]]
        persona = metadata["persona"] if isinstance(metadata["persona"], str) else metadata["persona"]["role"]
        job = metadata["job_to_be_done"] if isinstance(metadata["job_to_be_done"], str) else metadata["job_to_be_done"]["task"]

        # SECTION 4: Dynamic Query Builder
        base_prompt = f"You are a {persona} and your goal is to {job}."
        queries = [
            base_prompt,
            f"Which sections in the documents help a {persona} to {job}?",
            f"What content should be used to {job}?",
            f"Identify key suggestions, locations, or tips for: {job}",
            f"Find relevant sections to achieve the goal: {job}"
        ]

        # SECTION 5: Load Sentence Embedding Model
        print("Load sentence embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')  # ~80MB
        query_embeddings = model.encode(queries)

        # SECTION 6: Process PDFs and Score Pages
        print("Processing PDF documents...")
        ranked_pages = []
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf.fit(queries)

        for doc_name in input_docs:
            pdf_path = os.path.join(input_dir, doc_name)
            if not os.path.exists(pdf_path):
                print(f"Warning: PDF file not found: {pdf_path}")
                continue
            
            print(f"Processing: {doc_name}")
            pages = extract_pages(pdf_path)
            page_texts = [clean_text(text) for (_, text) in pages]
            
            if not page_texts:  # Skip if no valid pages found
                continue
                
            embeddings = model.encode(page_texts)
            tfidf_scores = tfidf.transform(page_texts).mean(axis=1).A1

            for (page_num, text), embedding, tfidf_score in zip(pages, embeddings, tfidf_scores):
                semantic_score = max([util.cos_sim(qe, embedding).item() for qe in query_embeddings])
                final_score = 0.7 * semantic_score + 0.3 * tfidf_score

                ranked_pages.append({
                    "document": doc_name,
                    "page_number": page_num,
                    "text": text,
                    "score": final_score
                })

        if not ranked_pages:
            print("Error: No valid pages found in any PDF documents")
            sys.exit(1)

        # SECTION 7: Sort Pages and Enforce Document Diversity
        ranked_pages = sorted(ranked_pages, key=lambda x: -x["score"])
        seen_docs = set()
        final_ranked = []

        for page in ranked_pages:
            if page["document"] not in seen_docs:
                seen_docs.add(page["document"])
                final_ranked.append(page)
            if len(final_ranked) == 5:
                break

        # SECTION 8: Build Output JSON
        output = {
            "metadata": {
                "input_documents": input_docs,
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }

        for rank, page in enumerate(final_ranked, 1):
            heading = extract_heading_candidates(page["text"])[0]
            output["extracted_sections"].append({
                "document": page["document"],
                "section_title": heading,
                "importance_rank": rank,
                "page_number": page["page_number"]
            })
            output["subsection_analysis"].append({
                "document": page["document"],
                "refined_text": clean_text(page["text"]),
                "page_number": page["page_number"]
            })

        # SECTION 9: Save Output
        output_path = os.path.join(output_dir, "output.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        print(f"Final Output saved to {output_path}")
        print("Process completed successfull!")

    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
