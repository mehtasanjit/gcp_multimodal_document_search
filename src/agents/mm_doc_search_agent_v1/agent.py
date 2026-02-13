import os
import re

from google.cloud import aiplatform
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools import VertexAiSearchTool

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

PROJECT_ID = ""
LOCATION = ""
# Define the Vertex AI Search Data Store ID
# Format: projects/{project}/locations/{location}/collections/{collection}/dataStores/{dataStore}
DATA_STORE_ID = ""

aiplatform.init(
    project=PROJECT_ID,
    location=LOCATION
)

# Initialize the Vertex AI Search Tool
# This tool uses the Advanced Data Store we created with image annotation capabilities
vertex_ai_search = VertexAiSearchTool(
    data_store_id=DATA_STORE_ID
)

def process_doc_search_agent_response_for_grounding_metadata(
    callback_context: CallbackContext, llm_response: LlmResponse
):
    """Callback to print grounding metadata chunks from the response."""
    if llm_response.grounding_metadata and llm_response.grounding_metadata.grounding_chunks:
        
        print("\n[Grounding Metadata] Original Chunks:")
        for i, chunk in enumerate(llm_response.grounding_metadata.grounding_chunks):
            if chunk.retrieved_context and chunk.retrieved_context.uri:
                print(f"  Chunk {i+1} Original URI: {chunk.retrieved_context.uri}")

        # Retrieve or initialize the URI map from session state
        # state is accessed via callback_context.state, which behaves like a dict
        uri_map = callback_context.state.get("grounding_metadata_uri_map", {})
        
        # We need to know the next index for new URIs
        # We can't just use len(uri_map) if we might have gaps or want strict ordering, 
        # but for this simple masking, len+1 is fine as we append.
        
        for chunk in llm_response.grounding_metadata.grounding_chunks:
            if chunk.retrieved_context and chunk.retrieved_context.uri:
                original_uri = chunk.retrieved_context.uri
                
                # Check if this URI is already mapped
                found_key = None
                for key, value in uri_map.items():
                    if value == original_uri:
                        found_key = key
                        break
                
                if not found_key:
                    next_id = len(uri_map) + 1
                    found_key = f"uri_{next_id}"
                    uri_map[found_key] = original_uri
                
                # Mask the URI in the response object
                chunk.retrieved_context.uri = found_key

        # Save the updated map back to session state
        callback_context.state["grounding_metadata_uri_map"] = uri_map

        print(llm_response.grounding_metadata.grounding_chunks)

def process_doc_search_output_formatting_response_to_replace_uri(
    callback_context: CallbackContext, llm_response: LlmResponse
):
    # Retrieve the URI map from session state
    uri_map = callback_context.state.get("grounding_metadata_uri_map", {})
    if not uri_map:
        return

    # Access content parts
    if not llm_response.content or not llm_response.content.parts:
        return

    # Replace [[uri_X]] with [actual_uri]
    def replace_uri(match):
        uri_key = match.group(1)
        actual_uri = uri_map.get(uri_key, uri_key)
        return f"[{actual_uri}]"

    # Iterate over parts and replace text in text parts
    for part in llm_response.content.parts:
        if part.text:
            new_text = re.sub(r"\[\[(uri_\d+)\]\]", replace_uri, part.text)
            part.text = new_text


# Define the Agent System Instruction
INSTRUCTION = """
  
  **Role:**
    * You are an advanced multimodal document search agent.
    * Your primary source of information is provided by the `vertex_ai_search_tool` tool which is a document search tool.

  **When a user asks a question:**
    * **Analyze** the user's query to identify key technical terms and concepts.
    * **Search** the documentation using the `vertex_ai_search` tool. Use specific keywords.
    * **Synthesize** a clear, concise, and accurate answer based *only* on the search results.
    * **Cite** the document names if available in the search results.
    * If the search results are insufficient, state that you cannot find the information in the data store docs. DO NOT hallucinate information.

  **Key capabilities of your search tool:**
    * It searches a collection of PDF manuals and technical guides.
    * It has advanced indexing for image content (tables, diagrams), so you can ask about visual details using descriptive text.

  **Tone:** Professional, precise, and helpful.

"""

MM_DOC_SEARCH_AGENT_OUTPUT_FORMATTING_INSTRUCTION = """

  **Role:**
    * You are a helpful assistant that formats the output of a search agent.
    * Your goal is to ensure that every claim in the text is cited with the correct URI from the provided grounding chunks.

  **Input:**
    * **Agent Response:** The original text generated by the search agent.
    * **Grounding Chunks:** A list of chunks used to ground the response. Each chunk has a content snippet and a URI (e.g., uri_1, uri_2).

  **Instructions:**
    *   Rewrite the **Agent Response** to include inline citations.
    *   The citation format must be **double square brackets** containing the URI, e.g., `[[uri_1]]`.
    *   Place the citation immediately after the sentence or clause it supports.
    *   Do not change the meaning or facts of the original text.
    *   Ensure the text flows naturally.
    *   If multiple chunks support a statement, include all of them, e.g., `[[uri_1]] [[uri_2]]`.
    *   Only use URIs provided in the Grounding Chunks.

"""

# Define the Agent
mm_doc_search_agent = Agent(
    name="mm_doc_search_agent",
    model="gemini-2.5-flash",
    description="This agent searches MM documents data store to answer user queries using Vertex AI Search.",
    instruction=INSTRUCTION,
    tools=[vertex_ai_search],
    after_model_callback=process_doc_search_agent_response_for_grounding_metadata
)

mm_doc_search_agent_output_formatting_agent = Agent(
    name="mm_doc_search_agent_output_formatting_agent",
    model="gemini-2.5-flash",
    description="This agent formats the output of a search agent.",
    instruction=MM_DOC_SEARCH_AGENT_OUTPUT_FORMATTING_INSTRUCTION,
    after_model_callback=process_doc_search_output_formatting_response_to_replace_uri
)

root_agent = SequentialAgent(
    name="mm_doc_search_agent_with_output_formatting",
    description="This agent searches MM documents data store to answer user queries using Vertex AI Search.",
    sub_agents=[
        mm_doc_search_agent, 
        mm_doc_search_agent_output_formatting_agent
    ]
)

