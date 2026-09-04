# Campus Copilot RAG 🎓⚡

A production-grade, stateful college intelligence assistant powered by LangGraph, FAISS vector search, and Groq inference. The system decouples offline document ingestion from real-time inference and features an **Agent Inspector** to observe active graph branches, entity extraction, and reasoning paths in real time.

---

## 📸 Interface Preview

<div align="center">
<img width="1903" height="789" alt="Screenshot 2026-09-04 131340" src="https://github.com/user-attachments/assets/39320172-1160-459c-be63-187ac38e05b7" />


*Figure: Streamlit interface featuring the live LangGraph Agent Inspector, branch routing, and contextual retrieval reasoning.*

</div>

---

## 📊 Pipeline Comparison Matrix

| Feature | Standard Toy RAG | Campus Copilot Architecture |
| :--- | :--- | :--- |
| **Ingestion Pipeline** | Re-computed at every app launch | Decoupled pre-indexing via `ingestion.py` |
| **State Inspection** | Black-box query execution | Real-time graph state visualizer (`Agent Inspector`) |
| **Branch Routing** | Single-chain hardcoded prompt | Intent-based dynamic routing (e.g., `FEE`, `ACADEMICS`) |
| **Vector Store** | Ephemeral memory array | Persistent serialized disk index (`/vectorstores`) |
| **Inference Latency** | High-latency remote endpoints | High-throughput hardware acceleration via Groq |
| **Diagnostics** | Unhandled crashes on bad models | Built-in pre-flight validation via `verify_groq.py` |

---

## 🏛️ System Architecture Flowchart

```mermaid
flowchart TD
    A["📄 Document Corpus<br>(PDFs / Regulations in data/)"] --> B["⚙️ ingestion.py<br>• Document Extraction<br>• Recursive Chunking<br>• Dense Vector Embeddings"]
    B --> C["💾 /vectorstores Disk Index<br>(Persistent Offline Store)"]
    
    subgraph Inference ["Inference Pipeline"]
        D["👤 Student Query"] --> E["🖥️ app_ui.py<br>(Streamlit Frontend)"]
        E --> F["🧠 LangGraph Router<br>• Intent Classification (e.g., FEE)<br>• Entity Extraction (Programme)"]
        F --> G["🔍 FAISS Retriever<br>(Top-K Keyword & Dense Search)"]
        G --> H["⚡ Groq Inference Engine<br>(Chain-of-Thought Synthesis)"]
        H --> I["💬 Grounded Response + Live State Visualizer"]
    end
    
    C --> G
