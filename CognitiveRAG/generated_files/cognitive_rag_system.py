import os
from typing import List, Literal, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_neo4j.graphs import Neo4jGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from sklearn.ensemble import IsolationForest
import numpy as np

# Load environment variables
load_dotenv()

# --- Models and Retrievers ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(persist_directory="./vector_store_chroma", embedding_function=embeddings)
vector_retriever = vector_store.as_retriever()
graph = Neo4jGraph()

# --- Novelty Detection Setup ---
novelty_detector = IsolationForest(contamination='auto', random_state=42)
try:
    initial_embeddings = np.array(vector_store.get(include=['embeddings'])['embeddings'])
    if len(initial_embeddings) > 0:
        novelty_detector.fit(initial_embeddings)
except Exception as e:
    print(f"Could not initialize novelty detector. It will be skipped. Error: {e}")
    initial_embeddings =


# --- Agent State ---
class AgentState(TypedDict):
    messages: List
    initial_context: List
    refined_context: List
    sub_queries: List[str]
    contradictions_found: bool
    novel_docs: List
    analogical_context: List
    generation: str
    recursion_depth: int

# --- Agent Nodes ---

def query_decomposer(state: AgentState):
    """Decomposes the user query into sub-queries."""
    print("---NODE: Decomposing Query---")
    user_query = state["messages"][-1].content
    prompt = f"""Decompose the user query into 3 distinct sub-queries. If simple, return the original query.
    Query: "{user_query}"
    Return a Python list of strings. Example: ['query1', 'query2']"""
    response = llm.invoke(prompt)
    sub_queries = eval(response.content)
    return {"sub_queries": sub_queries, "recursion_depth": 1}

def hybrid_retriever(state: AgentState):
    """Performs hybrid retrieval from vector store and knowledge graph."""
    print(f"---NODE: Hybrid Retrieval (Depth: {state['recursion_depth']})---")
    documents =
    for query in state["sub_queries"]:
        docs = vector_retriever.invoke(query)
        documents.extend(docs)
        try:
            graph_data = graph.query(
                "CALL db.index.fulltext.queryNodes('chunk_fulltext', $query, {limit: 2}) YIELD node RETURN node.text AS text",
                params={"query": query}
            )
            for record in graph_data:
                documents.append(Document(page_content=record['text']))
        except Exception as e:
            print(f"Graph retrieval failed for query '{query}': {e}")
    return {"initial_context": documents}

def contradiction_checker(state: AgentState):
    """Checks for contradictions in the retrieved context."""
    print("---NODE: Checking for Contradictions---")
    context_text = "\n".join([doc.page_content for doc in state["initial_context"]])
    prompt = f"""Analyze the following text for direct contradictions.
    Text: "{context_text}"
    Respond with only 'YES' or 'NO'."""
    response = llm.invoke(prompt)
    contradictions = "YES" in response.content
    return {"contradictions_found": contradictions, "refined_context": state["initial_context"]}

def novelty_detector_node(state: AgentState):
    """Uses anomaly detection to find novel information."""
    print("---NODE: Detecting Novelty---")
    if len(state["refined_context"]) == 0 or len(initial_embeddings) == 0:
        return {"novel_docs":}
        
    context_embeddings = embeddings.embed_documents([doc.page_content for doc in state["refined_context"]])
    scores = novelty_detector.predict(context_embeddings)
    
    novel_docs = [doc for doc, score in zip(state["refined_context"], scores) if score == -1]
    print(f"Found {len(novel_docs)} novel documents.")
    return {"novel_docs": novel_docs}

def analogical_reasoner(state: AgentState):
    """Generates cross-domain analogies to find 'unknown unknowns'."""
    print("---NODE: Analogical Reasoning---")
    user_query = state["messages"][-1].content
    prompt = f"""The user is asking about: '{user_query}'.
    What is a structurally similar problem or concept from a completely different domain?
    For example, for 'cancer as DNA corruption', an analogy is 'error correction in software'.
    Generate one such analogy and then create 2 search queries based on it.
    Return a Python list of strings. Example: ['analogous query 1', 'analogous query 2']"""
    response = llm.invoke(prompt)
    analogical_queries = eval(response.content)
    
    analogical_docs =
    for query in analogical_queries:
        docs = vector_retriever.invoke(query)
        analogical_docs.extend(docs)
    
    return {"analogical_context": analogical_docs}

def synthesizer(state: AgentState):
    """Generates the final answer."""
    print("---NODE: Synthesizing Final Answer---")
    user_query = state["messages"][-1].content
    context_text = "\n".join([doc.page_content for doc in state["refined_context"]])
    novelty_text = "\n".join([doc.page_content for doc in state["novel_docs"]])
    analogy_text = "\n".join([doc.page_content for doc in state.get("analogical_context",)])

    prompt = f"""You are a research assistant. Answer the user's query based on the provided context.
    Acknowledge if contradictions were found.
    If novel or analogical information was discovered, integrate it to provide a more comprehensive answer.
    
    Query: "{user_query}"
    
    Primary Context: "{context_text}"
    
    Contradictions Found: {state['contradictions_found']}
    
    ---
    Potentially Novel Information (may be related):
    "{novelty_text}"
    
    ---
    Insights from Analogical Domains:
    "{analogy_text}"
    """
    response = llm.invoke(prompt)
    return {"generation": response.content}

# --- Conditional Edges (Saturation Controller) ---

def saturation_controller(state: AgentState):
    """Decision point: determines if more refinement is needed."""
    print("---EDGE: Saturation Controller---")
    if state['recursion_depth'] >= 3:
        print("Max recursion depth reached. Proceeding to synthesis.")
        return "synthesize"
    
    # Simple logic: if contradictions or no novel docs, try analogical reasoning
    if state['contradictions_found'] or not state['novel_docs']:
         print("Contradictions or no novelty. Trying analogical reasoning.")
         return "analogize"

    print("Sufficient context found. Proceeding to synthesis.")
    return "synthesize"

# --- Build the Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("decompose_query", query_decomposer)
workflow.add_node("retrieve_context", hybrid_retriever)
workflow.add_node("check_contradictions", contradiction_checker)
workflow.add_node("detect_novelty", novelty_detector_node)
workflow.add_node("analogize", analogical_reasoner)
workflow.add_node("synthesize_answer", synthesizer)

workflow.set_entry_point("decompose_query")
workflow.add_edge("decompose_query", "retrieve_context")
workflow.add_edge("retrieve_context", "check_contradictions")
workflow.add_edge("check_contradictions", "detect_novelty")
workflow.add_edge("analogize", "synthesize_answer")
workflow.add_conditional_edges(
    "detect_novelty",
    saturation_controller,
    {"synthesize": "synthesize_answer", "analogize": "analogize"}
)
workflow.add_edge("synthesize_answer", END)

# --- Compile and Run ---
memory = SqliteSaver.from_conn_string(":memory:")
app = workflow.compile(checkpointer=memory)

# Example run
config = {"configurable": {"thread_id": "user_1"}}
user_input = "What are the main challenges of multi-hop reasoning in RAG and how can they be addressed using graph structures?"
for event in app.stream({"messages": [HumanMessage(content=user_input)]}, config=config):
    for key, value in event.items():
        if key!= "__end__":
            print(f"--- Event: {key} ---")
            # print(value) # Uncomment for very verbose output

# Print final state
final_state = app.get_state(config)
print("\n--- FINAL RESPONSE ---")
print(final_state.values['generation'])
