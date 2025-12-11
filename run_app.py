#!/usr/bin/env python3
import os
import sys

def setup_environment():
    """Setup environment variables and check requirements"""
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    
    # Check if vectordb exists
    if not os.path.exists("vectordb_openai"):
        print("❌ Vector database not found!")
        print("Please run the RAG-Using_FAISS.ipynb notebook first to create the vector database.")
        return False
    
    print("✅ Environment setup complete!")
    return True

if __name__ == "__main__":
    if setup_environment():
        print("🚀 Starting Flask RAG Application...")
        print("📱 Open http://localhost:5000 in your browser")
        
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        sys.exit(1)
