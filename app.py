from flask import Flask, render_template, request, jsonify, Response, stream_template
import os
import json
import time
import hashlib
from werkzeug.utils import secure_filename
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.tools import tool
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Langfuse for tracking
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store uploaded file hashes to prevent duplicates
uploaded_files = set()

# Initialize OpenAI model
model = OpenAIModel(
    client_args={"api_key": os.getenv("OPENAI_API_KEY")},
    model_id="gpt-4o",
    params={"max_tokens": 1000, "temperature": 0.7}
)

# Initialize embeddings
embedding_model = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-large"
)
PERSIST_DIR = "vectordb_openai/"

def get_file_hash(file_path):
    """Generate hash for file content to detect duplicates"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def process_pdf(file_path):
    """Process PDF and add to vector database"""
    try:
        # Load and split PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(docs)
        
        # Load existing vectorstore or create new one
        try:
            vectorstore = FAISS.load_local(PERSIST_DIR, embedding_model, allow_dangerous_deserialization=True)
            # Add new documents
            vectorstore.add_documents(splits)
        except:
            # Create new vectorstore if doesn't exist
            vectorstore = FAISS.from_documents(splits, embedding_model)
        
        # Save updated vectorstore
        vectorstore.save_local(PERSIST_DIR)
        return len(splits)
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

@tool
def aws_course_search(query: str, k: int = 5) -> str:
    """Search AWS course content for relevant information."""
    try:
        vs = FAISS.load_local(PERSIST_DIR, embedding_model, allow_dangerous_deserialization=True)
        docs = vs.similarity_search(query, k=k)
        
        if not docs:
            return "No relevant information found."
        
        # Get content and sources
        content_parts = []
        sources = set()
        
        for doc in docs:
            content_parts.append(doc.page_content)
            source = doc.metadata.get("source", "")
            if source:
                # Extract just the filename from full path
                source_name = os.path.basename(source)
                sources.add(source_name)
        
        # Combine content and sources
        combined_content = "\n\n".join(content_parts)
        if sources:
            source_list = ", ".join(sorted(sources))
            combined_content += f"\n\nSources: {source_list}"
        
        return combined_content
    except Exception as e:
        return f"Error searching: {str(e)}"

@tool
def analyze_architecture_diagram(image_description: str) -> str:
    """Analyze architecture diagrams and explain AWS services and design patterns."""
    return f"Architecture Analysis: {image_description}"

# Initialize agent
agent = Agent(
    name="AWS Course Assistant",
    model=model,
    tools=[aws_course_search, analyze_architecture_diagram],
    system_prompt="You are an AWS course assistant. Use the aws_course_search tool to find relevant information. For architecture diagrams, use analyze_architecture_diagram tool to explain services and design patterns. Always provide clear, direct answers without mentioning tool usage. If users make typos or misspellings in AWS service names (like 'bedrok' for 'bedrock', 'lamda' for 'lambda'), understand their intent and respond about the correct service they meant.",
    record_direct_tool_call=True,
    trace_attributes={
        "session.id": str(uuid.uuid4()),
        "user.id": "web-user",
        "langfuse.tags": ["Web-RAG-App"]
    }
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        session_id = request.form.get('session_id', 'default')
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            return jsonify({'error': 'Only image files are allowed'}), 400
        
        # Initialize session history if not exists
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        # Add image upload to conversation history
        conversation_history[session_id].append(f"User: [Uploaded architecture diagram: {file.filename}]")
        
        # Save image temporarily for analysis
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        # Use OpenAI Vision to analyze the image
        import base64
        with open(temp_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Analyze with GPT-4 Vision
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this AWS architecture diagram. Identify the services shown and explain why this architecture might be chosen. Focus on AWS services, data flow, and architectural patterns. Use simple bullet points without markdown formatting."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        analysis = response.choices[0].message.content
        
        # Clean markdown from analysis
        clean_analysis = analysis.replace('**', '').replace('###', '').replace('##', '').replace('#', '')
        clean_analysis = clean_analysis.replace('- ', '• ').replace('* ', '• ')
        
        # Add analysis to conversation history
        conversation_history[session_id].append(f"Assistant: {clean_analysis}")
        
        # Clean up temp file
        os.remove(temp_path)
        
        return jsonify({
            'analysis': clean_analysis,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Check if file already processed
        file_hash = get_file_hash(file_path)
        if file_hash in uploaded_files:
            os.remove(file_path)  # Remove duplicate
            return jsonify({'message': 'This PDF is already available in the database'})
        
        # Process PDF
        chunks_added = process_pdf(file_path)
        uploaded_files.add(file_hash)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify({
            'message': f'PDF processed successfully! Added {chunks_added} chunks to the database.',
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Store conversation history per session
conversation_history = {}

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '')
        session_id = request.json.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Initialize session history if not exists
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        # Add user message to history
        conversation_history[session_id].append(f"User: {user_message}")
        
        # Build context from recent conversation (last 10 messages)
        recent_history = conversation_history[session_id][-10:]
        context = "\n".join(recent_history)
        
        # Create message with context
        contextual_message = f"Previous conversation:\n{context}\n\nPlease respond to the latest user message considering the conversation context."
        
        def generate():
            try:
                # Get response with context
                response = agent(contextual_message)
                response_text = str(response).strip()
                
                # Add AI response to history
                conversation_history[session_id].append(f"Assistant: {response_text}")
                
                # Clean up response - remove tool indicators AND markdown formatting
                lines = response_text.split('\n')
                clean_lines = []
                
                for line in lines:
                    line = line.strip()
                    if (line.startswith('Tool #') or 
                        line.startswith('Agent:') or 
                        line.startswith('Chunk ') or
                        not line):
                        continue
                    
                    # Remove markdown formatting
                    line = line.replace('**', '').replace('###', '').replace('##', '').replace('#', '')
                    line = line.replace('- ', '• ').replace('* ', '• ')
                    clean_lines.append(line)
                
                clean_response = '\n'.join(clean_lines).strip()
                
                # Stream word by word
                words = clean_response.split(' ')
                
                for word in words:
                    yield f"data: {json.dumps({'chunk': word + ' '})}\n\n"
                    time.sleep(0.05)
                
                yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
