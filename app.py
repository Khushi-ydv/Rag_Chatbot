import os
from pathlib import Path
from typing import TypedDict, Annotated

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ============================================================
# 1. SETUP
# ============================================================

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.error("Please add GROQ_API_KEY to your .env file.")
    st.stop()

# LangSmith tracing: LangChain/LangGraph auto-trace when LANGSMITH_TRACING=true
# and LANGSMITH_API_KEY are set in .env. Disable tracing if the key is missing.
if os.getenv("LANGSMITH_TRACING", "").lower() == "true" and not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "false"


# ============================================================
# 2. CREATE DEMO COMPANY POLICY DOCUMENTS
# ============================================================

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


leave_policy = """
ACME CORPORATION - LEAVE POLICY

Employees receive 24 paid leaves per calendar year.

Annual leave can be carried forward up to a maximum of 10 days.

Employees can request leave through the internal HR portal.

For planned leave of more than 3 consecutive working days,
employees should submit their request at least 7 days in advance.

Emergency leave should be communicated to the reporting manager
as soon as reasonably possible.

Employees who have completed less than 6 months at the company
are eligible for a maximum of 1 paid leave per month.

Unused leave above the 10-day carry-forward limit expires
at the end of the calendar year.
"""


workplace_policy = """
ACME CORPORATION - WORKPLACE POLICY

Employees are expected to work from 9:30 AM to 6:30 PM
from Monday to Friday.

Employees may work remotely up to 2 days per week
with manager approval.

Employees should remain available through the company's
official communication channels during working hours.

Company laptops must not be shared with unauthorized individuals.

Confidential company information must not be uploaded to
personal cloud storage or external AI tools without approval.

Employees should report security incidents to the IT team
immediately.

Employees are expected to complete mandatory security and
compliance training within the specified deadlines.
"""


(DATA_DIR / "leave_policy.txt").write_text(leave_policy)
(DATA_DIR / "workplace_policy.txt").write_text(workplace_policy)


# ============================================================
# 3. LOAD DOCUMENTS
# ============================================================

documents = []

for file in DATA_DIR.glob("*.txt"):

    documents.append(
        Document(
            page_content=file.read_text(),
            metadata={
                "source": file.name
            }
        )
    )


# ============================================================
# 4. CHUNK DOCUMENTS
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)


# ============================================================
# 5. LOCAL EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 6. VECTOR DATABASE
# ============================================================

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# ============================================================
# 7. RAG TOOL
# ============================================================

@tool
def search_company_policies(query: str) -> str:
    """
    Search the company's internal policy documents.

    Use this tool when the user asks about:
    - leave policy
    - working hours
    - remote work
    - workplace rules
    - security policies
    - company policies
    """

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant policy information was found."

    results = []

    for doc in docs:

        results.append(
            f"SOURCE: {doc.metadata['source']}\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(results)


# ============================================================
# 8. LEAVE BALANCE TOOL
# ============================================================

@tool
def get_leave_balance() -> str:
    """
    Get the current employee's remaining paid leave balance.
    """

    return """
    Employee: Demo User
    Remaining paid leaves: 14 days
    """


tools = [
    search_company_policies,
    get_leave_balance
]


tool_map = {
    tool.name: tool
    for tool in tools
}


# ============================================================
# 9. GROQ LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 10. LANGGRAPH STATE
# ============================================================

class State(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# 11. AGENT NODE
# ============================================================

def agent_node(state: State):

    system_message = HumanMessage(
        content="""
You are a Company Policy AI Agent.

You have two tools.

TOOL 1:
search_company_policies

Use this for questions about company policies,
rules, leave policies, working hours, remote work,
security and workplace policies.

TOOL 2:
get_leave_balance

Use this when the user asks about their personal
remaining leave balance.

You may use both tools if necessary.

Do not invent company policy information.

Answer clearly and concisely.
"""
    )

    messages = [
        system_message
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [
            response
        ]
    }


# ============================================================
# 12. TOOL NODE
# ============================================================

def tool_node(state: State):

    last_message = state["messages"][-1]

    results = []

    for tool_call in last_message.tool_calls:

        tool_name = tool_call["name"]

        tool_args = tool_call["args"]

        selected_tool = tool_map[tool_name]

        result = selected_tool.invoke(
            tool_args
        )

        results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

    return {
        "messages": results
    }


# ============================================================
# 13. ROUTER
# ============================================================

def should_continue(state: State):

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage):

        if last_message.tool_calls:

            return "tools"

    return END


# ============================================================
# 14. BUILD LANGGRAPH
# ============================================================

graph = StateGraph(State)

graph.add_node(
    "agent",
    agent_node
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_edge(
    START,
    "agent"
)

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

graph.add_edge(
    "tools",
    "agent"
)

app = graph.compile()


# ============================================================
# 15. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Company Policy Agent",
    page_icon="🤖"
)

st.title(
    "🤖 Company Policy AI Agent"
)

st.write(
    "Ask questions about company policies "
    "or your personal leave balance."
)

st.caption(
    "Groq + RAG + Tool Calling + LangGraph"
)


question = st.chat_input(
    "Ask something..."
)


if question:

    st.chat_message(
        "user"
    ).write(question)

    initial_state = {
        "messages": [
            HumanMessage(
                content=question
            )
        ]
    }

    with st.spinner(
        "Agent is thinking..."
    ):

        result = app.invoke(
            initial_state,
            config={
                "run_name": "rag-chatbot",
                "tags": ["streamlit", "rag"],
                "metadata": {"question": question},
            },
        )

    messages = result["messages"]

    final_answer = ""

    used_tools = []

    sources = []

    for message in messages:

        # --------------------------------------------
        # Tool calls
        # --------------------------------------------

        if isinstance(
            message,
            ToolMessage
        ):

            if "SOURCE:" in message.content:

                used_tools.append(
                    "RAG"
                )

                for line in message.content.splitlines():

                    if line.startswith(
                        "SOURCE:"
                    ):

                        sources.append(
                            line.replace(
                                "SOURCE:",
                                ""
                            ).strip()
                        )

            else:

                used_tools.append(
                    "Leave Balance Tool"
                )

        # --------------------------------------------
        # Final AI response
        # --------------------------------------------

        elif isinstance(
            message,
            AIMessage
        ):

            if message.content:

                final_answer = (
                    message.content
                )


    # --------------------------------------------
    # Display answer
    # --------------------------------------------

    st.chat_message(
        "assistant"
    ).write(
        final_answer
    )


    # --------------------------------------------
    # Display tools
    # --------------------------------------------

    if used_tools:

        st.divider()

        st.caption(
            "Tools used: "
            + ", ".join(
                dict.fromkeys(
                    used_tools
                )
            )
        )


    # --------------------------------------------
    # Display sources
    # --------------------------------------------

    if sources:

        st.caption(
            "Sources:"
        )

        for source in set(sources):

            st.write(
                f"📄 {source}"
            )