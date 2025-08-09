import os
from dotenv import load_dotenv
from bertopic import BERTopic
from langchain_community.document_loaders import DirectoryLoader
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# --- Environment Setup ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# --- Constants ---
SOURCE_DIRECTORY = "./source_documents"

class Neo4jThemeIntegrator:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def add_themes_and_relations(self, documents, topics, topic_info):
        with self._driver.session() as session:
            for doc_idx, topic_id in enumerate(topics):
                if topic_id == -1:  # Ignore outliers
                    continue

                topic_name = topic_info['Name'].iloc
                doc_source_id = documents[doc_idx].metadata.get('source', 'Unknown')

                session.write_transaction(
                    self._create_theme_and_link, doc_source_id, topic_id, topic_name
                )
        print("Finished integrating themes.")

    @staticmethod
    def _create_theme_and_link(tx, doc_source_id, topic_id, topic_name):
        # Create Theme node and relate it to the Document node
        # Assumes Document nodes were created with an 'id' property matching the source path
        query = (
            "MERGE (t:Theme {id: $topic_id, name: $topic_name}) "
            "WITH t "
            "MATCH (d:Document {id: $doc_source_id}) "
            "MERGE (d)-->(t)"
        )
        tx.run(query, topic_id=topic_id, topic_name=topic_name, doc_source_id=doc_source_id)

def main():
    # --- Load Documents ---
    print("Loading documents for thematic analysis...")
    loader = DirectoryLoader(SOURCE_DIRECTORY, glob="**/*.txt")
    documents = loader.load()
    docs_content = [doc.page_content for doc in documents]
    if not docs_content:
        print("No documents found for analysis.")
        return

    # --- Perform Topic Modeling with BERTopic ---
    print("Performing topic modeling with BERTopic... (This may take a while)")
    topic_model = BERTopic(language="english", verbose=True)
    topics, _ = topic_model.fit_transform(docs_content)

    # --- Get Topic Information ---
    topic_info = topic_model.get_topic_info()
    print("\nIdentified Topics:")
    print(topic_info)

    # --- Integrate Themes into Knowledge Graph ---
    print("\nIntegrating themes into Neo4j Knowledge Graph...")
    try:
        integrator = Neo4jThemeIntegrator(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        integrator.add_themes_and_relations(documents, topics, topic_info)
        integrator.close()
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        print("Please ensure your Neo4j Docker container is running and credentials are correct.")

    print("\n--- Thematic Analysis and Integration Complete ---")

if __name__ == "__main__":
    main()
