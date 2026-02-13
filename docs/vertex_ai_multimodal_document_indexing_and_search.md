# Vertex AI Multimodal Document Indexing & Search

This guide outlines the architecture and implementation for building an advanced Multimodal Document Search Agent using Google Cloud Vertex AI Search and the Agent Development Kit (ADK). By combining AI-driven metadata extraction with Vertex AI's native layout parsing and image annotation, we can perform highly accurate, grounded searches against complex documents like engineering reports, safety guidelines, and schematics.

## 1. Architecture Overview

The pipeline consists of three main stages:
1. **Pre-processing and Metadata Extraction:** Automatically extracting rich metadata from raw PDFs using Gemini models.
2. **Ingestion and Indexing:** Uploading documents to Google Cloud Storage (GCS) and indexing them in a Vertex AI Search Data Store with advanced chunking and image analysis enabled.
3. **Agent Integration & Grounding:** Querying the index via an ADK Search Agent and processing the retrieved grounding metadata to provide accurate citations.

---

## 2. Pre-processing & Metadata Extraction

Raw documents often lack structured data. We use a dedicated script (e.g., `metadata_extraction_from_pdf.py`) to process each PDF through a Gemini model (e.g., `gemini-2.5-flash`) to infer attributes such as:

*   **Document Type** & **Category**
*   **Topic Tags** & **Entities Mentioned**
*   **Two-line Summary**
*   **Technical Complexity** (High/Medium/Low)

### Generating the JSONL file
Vertex AI Search requires metadata to be provided in a specific JSONL format alongside the `gcsSource` URIs. The script generates `metadata.jsonl` where each line follows this schema:

```json
{
  "id": "md5-hash-of-file",
  "structData": {
    "title": "Document Title",
    "category": "Document Category",
    "ai_inferred_attributes": { ... }
  },
  "content": {
    "mimeType": "application/pdf",
    "uri": "gs://<your-bucket-name>/path/to/doc.pdf"
  }
}
```

**To generate metadata for a dataset:**
```bash
./venv/bin/python src/document_preprocessing/metadata_extraction_from_pdf.py data/documents/source/<dataset_name> \
    --gcs-base-uri gs://<bucket_name>/data/documents/source/<dataset_name> \
    --category "Safety Reports and Guidelines" \
    --infer-ai-attributes \
    --project <project_id> \
    --location <location>
```

---

## 3. Data Ingestion & Index Creation

### 3.1 Uploading to GCS
Before creating the search index, both the PDF documents and the generated `metadata.jsonl` file must be uploaded to a GCS bucket. 

```bash
# Example upload command
gsutil -m cp -r data/documents/source/<your-dataset-name>/* gs://<your-bucket-name>/data/documents/source/<your-dataset-name>/
```

### 3.2 Creating the Advanced Data Store
To leverage multimodal capabilities, we configure the Vertex AI Data Store with specific layout parsing and chunking configurations via the REST API or Terraform. 

Key configurations:
*   `layoutBasedChunkingConfig`: Ensures chunks respect the visual layout of the PDF (e.g., keeping tables or sections together).
*   `includeAncestorHeadings`: Injects the document's header hierarchy into the chunk context.
*   `enable_image_annotation: True`: *Crucial* for multimodal search. When enabled, Vertex AI uses a multimodal Large Language Model (LLM) to analyze embedded images (such as JPEGs, PNGs, diagrams, and charts) during the document ingestion phase. It automatically generates text descriptions for these visuals, making their content fully searchable and accessible for Retrieval-Augmented Generation (RAG).
    *   *Note: Enabling this feature can increase document ingestion time and may incur additional costs, as it actively calls an LLM to process and describe every significant image found in the document.*

**To create the Data Store and import documents:**

*Using placeholders:*
```bash
./venv/bin/python src/document_preprocessing/create_advanced_datastore.py \
    --project-id <project_id> \
    --location <location> \
    --collection "default_collection" \
    --data-store-id "<data_store_id>" \
    --display-name "<display_name>" \
    --gcs-uri "gs://<bucket_name>/data/documents/source/<dataset_name>/metadata.jsonl"
```

*Example with anonymized values:*
```bash
./venv/bin/python src/document_preprocessing/create_advanced_datastore.py \
    --project-id my-gcp-project-123 \
    --location global \
    --collection default_collection \
    --data-store-id "my-safety-reports-v1" \
    --display-name "My Safety Reports" \
    --gcs-uri "gs://my-gcs-bucket-1/data/documents/source/safety_reports_1/metadata.jsonl"
```
*(Optional: Add `--skip-create` if the Data Store already exists and you only want to import)*

---

## 4. Search Agent Integration

The application utilizes the Google ADK to facilitate orchestrated agentic interactions with the search index.

### 4.1 VertexAiSearchTool
The core search functionality is handled by injecting the `VertexAiSearchTool` into the agent's toolset.
```python
doc_search_tool = VertexAiSearchTool(
    project="<your-gcp-project-id>",
    location="global",
    data_store_id="<your-data-store-id>"
)
```

### 4.2 Handling Grounding and Citations
A major requirement for enterprise document search is traceability. When the agent uses the search tool, the LLM response contains `grounding_metadata`.

We use the ADK's `after_model_callback` pattern to intercept the response, extract the chunk URIs, and format them into a user-friendly citation map:

1. **Extraction:** Intercept `llm_response.grounding_metadata.grounding_chunks`.
2. **State Management:** Map raw GCS URIs to anonymized or simplified citation keys (e.g., `uri_1`, `uri_2`) and store them in the `callback_context.state`.
3. **Sequential Pipeline:** Pass the response and the state to a secondary `citation_agent` (via `SequentialAgent`) that acts exclusively to reformat the text, replacing inline LLM citations with the mapped `<#href: uri_X>` tags.

This ensures the user sees an authoritative, clean response while the underlying system tracks exact origin URIs.
