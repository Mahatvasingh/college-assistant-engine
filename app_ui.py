import uuid
import streamlit as st
from langchain_core.messages import HumanMessage

# Import the compiled graph directly from your script (assuming your file is named rag_app.py)
from rag_app import app

# ------------------------------------------------------------
# 1. UI CONFIGURATION & CUSTOM STYLING
# ------------------------------------------------------------
st.set_page_config(
    page_title="CampusAI | Smart Student Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark glassmorphism-inspired interface styles
st.markdown("""
<style>
    .reportview-container {
        background: #0f172a;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 4px;
        word-break: break-word;
    }
    .badge-academic { color: #4ade80; font-weight: 600; }
    .badge-fee { color: #facc15; font-weight: 600; }
    .badge-general { color: #60a5fa; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 2. SESSION STATE & MEMORY INITIALIZATION
# ------------------------------------------------------------
# Persistent session ID to bind with LangGraph's MemorySaver checkpointer
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# UI Chat transcript storage
if "chat_transcript" not in st.session_state:
    st.session_state.chat_transcript = [
        {
            "role": "assistant",
            "content": "👋 Hi there! I'm your universal campus advisor. Ask me anything about academic regulations, exam dates, fee structures, refunds, or general policies."
        }
    ]

# Telemetry metadata from RouteDecision
if "agent_telemetry" not in st.session_state:
    st.session_state.agent_telemetry = {
        "category": "idle",
        "search_query": "Waiting for input...",
        "programme": "None detected yet"
    }

# ------------------------------------------------------------
# 3. SIDEBAR: REAL-TIME GRAPH INSPECTOR
# ------------------------------------------------------------
with st.sidebar:
    st.title("🎓 Agent Inspector")
    st.caption("Live LangGraph State Visualizer")
    st.divider()

    cat = st.session_state.agent_telemetry["category"]
    badge_markup = {
        "academic": '<span class="badge-academic">● ACADEMIC</span>',
        "fee": '<span class="badge-fee">● FEE</span>',
        "general": '<span class="badge-general">● GENERAL</span>',
        "idle": '<span style="color: #64748b;">○ IDLE</span>'
    }.get(cat, f"<span>{cat.upper()}</span>")

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Active Graph Branch</div>
        <div class="metric-value">{badge_markup}</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Detected Student Programme</div>
        <div class="metric-value">📚 {st.session_state.agent_telemetry["programme"]}</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">FAISS Search Keyword Query</div>
        <div class="metric-value" style="font-family: monospace; font-size: 0.85rem;">
            {st.session_state.agent_telemetry["search_query"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Reset Chat & Memory", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_transcript = [
            {
                "role": "assistant",
                "content": "Conversation and memory cleared. What would you like to know?"
            }
        ]
        st.session_state.agent_telemetry = {
            "category": "idle",
            "search_query": "Waiting for input...",
            "programme": "None detected yet"
        }
        st.rerun()

# ------------------------------------------------------------
# 4. CHAT INTERFACE DISPLAY
# ------------------------------------------------------------
st.title("Campus Academic & Fee Assistant")
st.caption("Powered by LangGraph, Dual FAISS Retrievers, and Qwen / Groq")

# Render previous turns
for message in st.session_state.chat_transcript:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------------------------------------------------
# 5. EXECUTION & STATE SYNC
# ------------------------------------------------------------
if user_prompt := st.chat_input("Ask a question (e.g., 'What is the attendance criteria for B.Tech?')..."):
    # Display user input immediately
    st.session_state.chat_transcript.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Invoke graph using the persistent thread_id
    execution_config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent and querying FAISS..."):
            try:
                # Run the graph workflow
                graph_output = app.invoke(
                    {"messages": [HumanMessage(content=user_prompt)]},
                    config=execution_config
                )

                # Extract assistant response
                bot_reply = graph_output["messages"][-1].content
                st.markdown(bot_reply)

                # Append to transcript
                st.session_state.chat_transcript.append({"role": "assistant", "content": bot_reply})

                # Extract RouteDecision and Programme to update sidebar telemetry
                route_data = graph_output.get("route")
                st.session_state.agent_telemetry = {
                    "category": getattr(route_data, "category", "general"),
                    "search_query": getattr(route_data, "search_query", "None"),
                    "programme": graph_output.get("programme") or "Universal / Not Detected"
                }

                # Rerun to sync sidebar immediately
                st.rerun()

            except Exception as exc:
                st.error(f"Graph execution failed: `{exc}`")