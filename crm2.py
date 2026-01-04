import streamlit as st
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_openai_tools_agent
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain.agents import AgentExecutor
from sentence_transformers import SentenceTransformer
import faiss
import psycopg2
import pandas as pd



class AgentState(TypedDict):
    user_input: str
    response: str

# -------------------
# Database connection
# -------------------
def connect_db():
    return psycopg2.connect(
        dbname="CRM",
        user="postgres",
        password="12345",
        host="localhost",
        port="5432"
    )

def fetch_crm_data():
    conn = connect_db()
    df = pd.read_sql("SELECT * FROM crm_leads;", conn)
    conn.close()
    df['content'] = (
        df['lead_name'].fillna('') + " " +
        df['customer'].fillna('') + " " +
        df['hot_lead'].fillna('') + " " +
        df['type'].fillna('') + " " +
        df['lead_stage'].fillna('')
    )
    return df

# -------------------
# FAISS Setup
# -------------------
@st.cache_resource
def build_faiss_index():
    df = fetch_crm_data()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df['content'].tolist(), convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return df, model, index

crm_df, embed_model, faiss_index = build_faiss_index()

def search_crm(query, top_n=5):
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    D, I = faiss_index.search(query_vec, top_n)
    results = crm_df.iloc[I[0]].copy()
    results['score'] = 1 - (D[0] / 2)
    return results

# -------------------
# LLM
# -------------------
llm = ChatOllama(model="llama3.2")

# -------------------
# Tools
# -------------------
@tool(return_direct=True)
def greeting_tool(user_input: str) -> str:
    """Respond to greetings like hi, hello, etc. Use this when user says hello or similar greetings."""
    prompt = f"""You are a friendly chatbot. Respond warmly to greetings.
    User: {user_input}
    Response:"""
    result = llm.invoke(prompt)
    return result.content if hasattr(result, 'content') else str(result)

@tool(return_direct=True)
def rag_tool(user_input: str) -> str:
    """Search CRM database for relevant information to the query."""
    search_results = search_crm(user_input, top_n=5)
    if search_results.empty:
        return "No relevant CRM data found."
    
    # Format retrieved info
    response = "Here are the most relevant CRM records:\n\n"
    for _, row in search_results.iterrows():
        response += f"- **Lead:** {row['lead_name']} | **Customer:** {row['customer']} | **Stage:** {row['lead_stage']} | **Revenue:** {row['lead_revenue']} | **Score:** {row['score']:.2f}\n"
    return response

# -------------------
# Agent setup
# -------------------
tools1= [greeting_tool, rag_tool]



# template = '''
# You are a helpful CRM AI Assistant. You can:
# - Respond to greetings
# - Retrieve CRM information from a database via RAG

# You have access to the following tools:
# {tools1}

# Use the following format:

# Question: the input question
# Thought: reasoning
# Action: one of [{tool_names}]
# Action Input: the input to the action
# Observation: the tool's result
# ... repeat as needed
# Thought: I now know the final answer
# Final Answer: the final answer

# Begin!

# Question: {input}
# {agent_scratchpad}
# '''


# system_prompt = """You are a CRM Assistant AI...

# You have access to these tools:
# {tools}

# Always be helpful and professional.

# Question: {input}
# {agent_scratchpad}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a CRM assistant..."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

#prompt1 = PromptTemplate.from_template(system_prompt)
agent=create_openai_tools_agent(llm=llm,tools=tools1,prompt=prompt)
#agent = create_openai_tools_agent(model=llm, tools=tools1, prompt=prompt1)


#prompt1= PromptTemplate.from_template(template)
#agent=create_react_agent(model=llm,tools=tools1,prompt=prompt1)
agent_executor = AgentExecutor(agent=agent, tools=tools1, verbose=True)

# -------------------
# LangGraph workflow
# -------------------


def route_with_agent(state: AgentState) -> dict:
    try:
        response = agent_executor.invoke({"input": state["user_input"]})
        final_output = response.get("output","No response.")
        return {"response": final_output}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

def format_response(state: AgentState) -> dict:
    """Pass-through node to finalize response."""
    return {"response": state["response"]}

workflow = StateGraph(AgentState)
workflow.add_node("agent_router", route_with_agent)
workflow.add_node("response_formatter", format_response)
workflow.set_entry_point("agent_router")
workflow.add_edge("agent_router", "response_formatter")
workflow.add_edge("response_formatter", "__end__")
app = workflow.compile()

# -------------------
# Streamlit UI
# -------------------
st.title("CRM Agentic Chatbot")
if "messages" not in st.session_state:
    st.session_state.messages = [] 

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about CRM data or say hello"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    inputs = {"user_input": prompt, "response": ""}
    with st.spinner("Processing..."):
        result = app.invoke(inputs)
        response = result.get("response", "No response.")
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
