import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # --- Setup ---
    # This assumes you have a set of questions, ground truth answers,
    # and the results from your RAG pipeline (generated answer and retrieved contexts).

    # For demonstration, we'll create a small, sample dataset.
    # In a real scenario, this would come from a test set or production logs.
    eval_data = {
        'question':,
        'answer':,
        'contexts':,
           ,
        'ground_truth':
    }
    dataset = Dataset.from_dict(eval_data)

    # --- Initialize Ragas ---
    # Ragas uses an LLM as a judge for some of its metrics
    ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    ragas_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Define the metrics we want to calculate
    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ]

    # --- Run Evaluation ---
    print("Running RAG evaluation with Ragas...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    print("\n--- Evaluation Results ---")
    print(result)

    # Convert to a pandas DataFrame for better visualization
    df_results = result.to_pandas()
    print("\n--- Detailed Results DataFrame ---")
    print(df_results.head())

if __name__ == "__main__":
    main()
