<<<<<<< HEAD
# Strands AWS RAG Assistant

A conversational AI assistant powered by Strands Agents for AWS course materials with RAG (Retrieval-Augmented Generation) capabilities.

## Features

- 🤖 **Conversational AI** with memory across sessions
- 📚 **RAG Integration** with AWS course PDFs using FAISS vector database
- 📄 **PDF Upload** - Expand knowledge base with your own documents
- 📷 **Image Analysis** - Upload architecture diagrams for AI analysis using GPT-4 Vision
- 🔄 **Real-time Streaming** - Word-by-word response generation
- 📊 **Langfuse Tracking** - Complete observability and analytics
- 🎨 **Strands Theme** - Professional green/black UI design
- 🔍 **Smart Search** - Typo-tolerant AWS service recognition
- 📖 **Source Attribution** - Shows which PDFs information came from

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- Langfuse account (optional, for tracking)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd Strands
```

2. **Create virtual environment**
```bash
python -m venv flask_env
source flask_env/bin/activate  # On Windows: flask_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements_flask.txt
```

4. **Set up environment variables**
Update the API keys in `app.py`:
```python
# OpenAI API Key
client_args={"api_key": "your-openai-api-key"}

# Langfuse Keys (optional)
os.environ["LANGFUSE_PUBLIC_KEY"] = "your-public-key"
os.environ["LANGFUSE_SECRET_KEY"] = "your-secret-key"
```

5. **Run the application**
```bash
python app.py
```

6. **Open your browser**
Navigate to `http://localhost:5000`

## Usage

### Chat Interface
- Ask questions about AWS services and concepts
- Get contextual responses with conversation memory
- Responses include source attribution from course materials

### PDF Upload
- Click "📄 Upload PDF" in the header
- Upload AWS-related PDFs to expand the knowledge base
- Duplicate detection prevents re-uploading same files

### Image Analysis
- Click the camera button (📷) next to the send button
- Upload architecture diagrams for AI analysis
- Get explanations of AWS services and design patterns

## Project Structure

```
Strands/
├── app.py                    # Main Flask application
├── templates/
│   └── index.html           # Frontend chat interface
├── requirements_flask.txt   # Python dependencies
├── run_app.py              # Launch script
├── uploads/                # Temporary file storage
├── vectordb_openai/        # FAISS vector database
└── README.md              # This file
```

## Key Components

### Backend (`app.py`)
- **Flask API** with streaming endpoints
- **Strands Agent** integration with tools
- **FAISS vector search** for RAG functionality
- **OpenAI GPT-4 Vision** for image analysis
- **Session management** for conversation memory
- **Langfuse integration** for observability

### Frontend (`templates/index.html`)
- **Modern chat interface** with Strands branding
- **Real-time streaming** display
- **File upload** capabilities (PDF + images)
- **Responsive design** for mobile/desktop

### Tools
- `aws_course_search` - Searches vector database for relevant content
- `analyze_architecture_diagram` - Analyzes uploaded architecture images

## Configuration

### API Keys
Update these in `app.py`:
- OpenAI API key for GPT-4 and embeddings
- Langfuse keys for tracking (optional)

### Model Settings
```python
model = OpenAIModel(
    model_id="gpt-4o",
    params={
        "max_tokens": 1000,
        "temperature": 0.7
    }
)
```

## Features in Detail

### RAG (Retrieval-Augmented Generation)
- Uses FAISS for fast similarity search
- Embeds documents using OpenAI text-embedding-3-large
- Provides source attribution with PDF names
- Handles duplicate document detection

### Conversation Memory
- Maintains context across messages in a session
- Stores last 10 messages for context
- Session-based isolation between users

### Image Analysis
- GPT-4 Vision integration for architecture diagrams
- Automatic analysis of AWS services and patterns
- Explains design decisions and best practices

### Streaming Responses
- Word-by-word response generation
- Real-time display without loading screens
- Clean formatting without markdown delimiters

## Deployment

### Local Development
```bash
python app.py
```

### Production Considerations
- Set up proper environment variables
- Use production WSGI server (gunicorn, uwsgi)
- Configure reverse proxy (nginx)
- Set up SSL certificates
- Use persistent storage for vector database

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Your License Here]

## Support

For issues and questions, please open a GitHub issue or contact the development team.
=======
# AWS Course RAG Agent (Strands + OpenAI)

A Strands-Agents project that provides a **retrieval-augmented generation (RAG)** assistant over AWS course material (AI Practitioner & Cloud Practitioner). It leverages OpenAI embeddings, FAISS as a vector store, Langfuse for observability, and RAGAS for evaluation.

---

## 🚀 Features

- **RAG over AWS Course PDFs**: Ask questions about the AI Practitioner and Cloud Practitioner course material.  
- **OpenAI Embeddings**: Use `text-embedding-3-large` (or your preferred embedding model).  
- **FAISS Vector Store**: Store chunk embeddings for efficient similarity search.  
- **Current Time Tool**: A built-in Strands tool to fetch the current timestamp.  


---

## 🔭 Future Enhancements & Planning

- **Observability with Langfuse**  
  - Instrument the Strands agent to send traces to **Langfuse**, capturing inputs, LLM calls, tool executions, latencies, and costs.  
  - Use Langfuse tracing to understand how the agent reasons (which chunks were retrieved, how the LLM responded, etc.).  
  - Monitor performance metrics such as token usage, error rates, and model latency over time to optimize the agent. 
  - Optionally leverage OpenTelemetry integration for more detailed telemetry and to aggregate traces from other parts of your system. 

- **Evaluation with RAGAS**  
  - Integrate **RAGAS** (Retrieval-Augmented Generation Assessment System) to evaluate your RAG pipeline automatically. 
  - Track core metrics like **faithfulness**, **answer relevancy**, **context precision**, and **context recall** to diagnose where the system can improve.  
  - Set up periodic evaluation (e.g., nightly or on every major update) to assess how changes to embeddings, chunking, or the LLM affect quality.  
  - Build a dashboard (or use existing reporting) to visualize RAGAS scores over time and make data-driven decisions on improving retrieval / generation.


>>>>>>> 0530009ce4dce42a8b8e93d30f1b4cea0f461b43
