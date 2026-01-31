import sys
import os
import json

# Add project root to python path
sys.path.append(os.getcwd())

try:
    from backend.main import app
    
    # Generate OpenAPI schema
    openapi_schema = app.openapi()
    
    # Downgrade to OpenAPI 3.0.2 for better tool compatibility
    openapi_schema["openapi"] = "3.0.2"
    
    # Save to file
    with open("endee_api_collection.json", "w") as f:
        json.dump(openapi_schema, f, indent=2)
        
    print("Successfully exported OpenAPI schema to endee_api_collection.json")
except Exception as e:
    print(f"Error exporting OpenAPI schema: {str(e)}")
