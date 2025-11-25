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


