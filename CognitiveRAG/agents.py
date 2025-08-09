# CognitiveRAG/agents.py
from typing import List, Dict, Any, TypedDict
from langchain_core.documents import Document
from . import config
from .retriever import retriever
from .llm_provider import get_llm

# --- Agent State ---
class RAGState(TypedDict):
    original_query: str
    current_query: str
    context: List[Document]
    answer: str
    plan: List[str]
    recursion_depth: int
    feedback: Dict[str, Any]

# --- Agent Components ---

class PlannerAgent:
    """
    Analyzes the user query and creates a multi-step plan.
    """
    def __init__(self):
        self.llm = get_llm(config.PLANNER_MODEL)

    def create_plan(self, query: str) -> List[str]:
        prompt = f"""You are a research planner. Analyze the user's query and break it down into a series of simple, logical steps.
If the query is simple, the plan can be a single step.
For complex questions, create up to 3 steps.
User Query: "{query}"

Return the plan as a Python list of strings. Example: ['Define key terms.', 'Search for main arguments.', 'Summarize findings.']
"""
        try:
            response = self.llm.invoke(prompt)
            # The response content is often a string representation of a list
            plan = eval(response.content)
            if isinstance(plan, list):
                return plan
        except Exception as e:
            print(f"Warning: Could not parse plan from LLM response: {e}")
            # Fallback for simple string response
            try:
                return [response.content]
            except:
                pass
        return [query] # Default fallback

class SynthesisAgent:
    """
    Generates a final answer based on the gathered context.
    """
    def __init__(self):
        self.llm = get_llm(config.SYNTHESIS_MODEL)

    def synthesize(self, query: str, context: List[Document]) -> str:
        if not context:
            return "I don't have enough information to answer your question. Please try providing more context or documents."
            
        context_str = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source', 'N/A')}\n\n{doc.page_content}" for doc in context])
        prompt = f"""You are a research assistant. Your task is to answer the user's query based *only* on the provided context.
Cite your sources using the [Source: filename] format.
If the context does not contain the answer, state that clearly. Do not use outside knowledge.

User Query: "{query}"

Context:
{context_str}

Answer:
"""
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

class ReflectionAgent:
    """
    Evaluates the generated answer and provides feedback for refinement.
    """
    def __init__(self):
        self.llm = get_llm(config.REFLECTION_MODEL)

    def reflect(self, query: str, answer: str, context: List[Document]) -> Dict[str, Any]:
        prompt = f"""You are a quality assurance agent. Evaluate the provided answer based on the query and context.
Check for the following:
1. Faithfulness: Is the answer fully supported by the context?
2. Relevance: Does the answer directly address the user's query?
3. Completeness: Are there any gaps in the answer that the context could fill?

User Query: "{query}"
Answer: "{answer}"

Provide feedback in a dictionary format with keys 'faithful', 'relevant', 'complete' (booleans) and 'suggestions' (a string for improvement).
Example: {{'faithful': True, 'relevant': False, 'complete': True, 'suggestions': 'The answer is factually correct but does not address the second part of the user's question.'}}
"""
        try:
            response = self.llm.invoke(prompt)
            feedback = eval(response.content)
            if isinstance(feedback, dict):
                return feedback
        except Exception as e:
            print(f"Warning: Could not parse reflection feedback: {e}")
            return {'faithful': True, 'relevant': True, 'complete': True, 'suggestions': 'Could not parse feedback.'}
        return {'faithful': True, 'relevant': True, 'complete': True, 'suggestions': ''}


class Orchestrator:
    """
    Manages the multi-agent workflow for answering a query.
    """
    def __init__(self):
        try:
            self.planner = PlannerAgent()
            self.synthesis = SynthesisAgent()
            self.reflection = ReflectionAgent()
        except Exception as e:
            print(f"Warning: Could not initialize all agents: {e}")
            # Initialize minimal fallback
            self.planner = None
            self.synthesis = None
            self.reflection = None

    def run(self, query: str) -> RAGState:
        state = RAGState(
            original_query=query,
            current_query=query,
            context=[],
            answer="",
            plan=[],
            recursion_depth=0,
            feedback={}
        )

        if not self.planner:
            state['answer'] = "System error: Could not initialize agents properly."
            return state

        try:
            state['plan'] = self.planner.create_plan(query)
            print(f"Plan: {state['plan']}")

            for step in state['plan']:
                state['current_query'] = step
                print(f"Executing step: {state['current_query']}")
                
                # Retrieve context for the current step
                try:
                    retrieved_docs = retriever.retrieve(state['current_query'], top_k=config.VECTOR_TOP_K)
                    state['context'].extend(retrieved_docs)
                    # De-duplicate context
                    state['context'] = list({doc.page_content: doc for doc in state['context']}.values())
                except Exception as e:
                    print(f"Warning: Could not retrieve documents for step '{step}': {e}")

            # Initial synthesis
            print("Synthesizing initial answer...")
            state['answer'] = self.synthesis.synthesize(state['original_query'], state['context'])

            # Reflection and refinement loop
            if self.reflection:
                while state['recursion_depth'] < config.MAX_RECURSION_DEPTH:
                    print(f"Reflection loop depth: {state['recursion_depth'] + 1}")
                    state['feedback'] = self.reflection.reflect(state['original_query'], state['answer'], state['context'])
                    print(f"Feedback: {state['feedback']}")

                    if state['feedback'].get('faithful') and state['feedback'].get('relevant') and state['feedback'].get('complete'):
                        print("Answer is satisfactory. Exiting loop.")
                        break
                    
                    # If not satisfactory, refine the query based on suggestions
                    refinement_query = f"{state['original_query']} - addressing feedback: {state['feedback'].get('suggestions', '')}"
                    print(f"Refining query: {refinement_query}")
                    
                    try:
                        refined_docs = retriever.retrieve(refinement_query, top_k=2) # Fetch less docs for refinement
                        state['context'].extend(refined_docs)
                        state['context'] = list({doc.page_content: doc for doc in state['context']}.values())

                        print("Re-synthesizing answer...")
                        state['answer'] = self.synthesis.synthesize(state['original_query'], state['context'])
                        state['recursion_depth'] += 1
                    except Exception as e:
                        print(f"Warning: Could not refine query: {e}")
                        break

        except Exception as e:
            print(f"Error in orchestrator run: {e}")
            state['answer'] = f"An error occurred while processing your query: {str(e)}"

        return state
