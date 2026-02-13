import json
import time
import requests
import google.auth
from google.auth.transport.requests import Request

import argparse

def get_authenticated_session(project_id):
    credentials, _ = google.auth.default()
    credentials.refresh(Request())
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
        "X-Goog-User-Project": project_id
    })
    return session

def create_data_store(session, project_id, location, collection, data_store_id, display_name):
    parent = f"projects/{project_id}/locations/{location}/collections/{collection}"
    url = f"https://discoveryengine.googleapis.com/v1beta/{parent}/dataStores?dataStoreId={data_store_id}"
    
    payload = {
        "displayName": display_name,
        "industryVertical": "GENERIC",
        "solutionTypes": ["SOLUTION_TYPE_SEARCH"],
        "contentConfig": "CONTENT_REQUIRED",
        "documentProcessingConfig": {
            "chunkingConfig": {
                "layoutBasedChunkingConfig": {
                    "chunkSize": 500,
                    "includeAncestorHeadings": True
                }
            },
            "defaultParsingConfig": {
                "layoutParsingConfig": {
                    "enableImageAnnotation": True 
                }
            }
        }
    }
    
    print(f"Creating Data Store via REST API: {data_store_id}...")
    response = session.post(url, json=payload)
    
    if response.status_code == 200:
        print("Operation started successfully.")
        return response.json()
    elif response.status_code == 409:
        print(f"Data Store {data_store_id} already exists. Skipping creation.")
        return True
    else:
        print(f"Error creating Data Store: {response.status_code}")
        print(response.text)
        return None

def import_documents(session, project_id, location, collection, data_store_id, gcs_uri):
    parent = f"projects/{project_id}/locations/{location}/collections/{collection}/dataStores/{data_store_id}/branches/default_branch"
    url = f"https://discoveryengine.googleapis.com/v1beta/{parent}/documents:import"
    
    payload = {
        "gcsSource": {
            "inputUris": [gcs_uri],
            "dataSchema": "document"
        },
        "reconciliationMode": "INCREMENTAL"
    }
    
    print(f"Importing documents from {gcs_uri}...")
    response = session.post(url, json=payload)
    
    if response.status_code == 200:
        print("Import Operation started.")
        print(json.dumps(response.json(), indent=2))
        return response.json()
    else:
        print(f"Error importing documents: {response.status_code}")
        print(response.text)
        return None

def main():
    parser = argparse.ArgumentParser(description="Create Advanced Data Store and Import Documents")
    parser.add_argument("--project-id", required=True, help="Google Cloud Project ID")
    parser.add_argument("--location", default="global", help="Vertex AI Search Location")
    parser.add_argument("--collection", default="default_collection", help="Collection Name")
    parser.add_argument("--data-store-id", required=True, help="Data Store ID")
    parser.add_argument("--display-name", required=True, help="Data Store Display Name")
    parser.add_argument("--gcs-uri", required=True, help="GCS URI for metadata.jsonl")
    parser.add_argument("--skip-create", action="store_true", help="Skip data store creation, only import")
    
    args = parser.parse_args()
    
    session = get_authenticated_session(args.project_id)
    
    if not args.skip_create:
        create_res = create_data_store(session, args.project_id, args.location, args.collection, args.data_store_id, args.display_name)
        if create_res:
            print("Waiting 10 seconds for Data Store creation to propagate...")
            time.sleep(10)
            import_documents(session, args.project_id, args.location, args.collection, args.data_store_id, args.gcs_uri)
    else:
        import_documents(session, args.project_id, args.location, args.collection, args.data_store_id, args.gcs_uri)

if __name__ == "__main__":
    main()

