import os
import chromadb
from chromadb.utils import embedding_functions
from src.logging import logging
from src.config import DISTANCE_METRIC


class Chroma_database():

    def __init__(self, distance_metric: str = DISTANCE_METRIC):
        logging.info(f"Initialising chromadb | distance={distance_metric}")
        self.distance_metric = distance_metric
        self.client = chromadb.PersistentClient(
            path=os.environ.get(
                "CHROMADB_PATH",
                os.path.join(os.getcwd(), "chromadb")
            )
        )
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="collectiontesting",
            embedding_function=self.sentence_transformer_ef,
            metadata={"hnsw:space": distance_metric}
        )

    def data_pre_processing(self, chunks, filename):
        try:
            logging.info("Processing the data")
            metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]
            ids       = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
            return ids, chunks, metadatas
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            raise

    def add_to_collection(self, ids, texts, metadatas, filename):
        if not texts:
            return
        batch_size = 200
        try:
            for i in range(0, len(texts), batch_size):
                end_idx = min(i + batch_size, len(texts))
                self.collection.add(
                    documents=texts[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
            logging.info("Data added without any errors")
        except Exception as e:
            logging.error(f"Error adding {filename}: {e}")
            raise

    def delete_by_filename(self, filename: str):
        """Purge all existing chunks for this file before re-ingesting."""
        try:
            existing = self.collection.get(where={"source": {"$eq": filename}})
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
                logging.info(f"Deleted {len(existing['ids'])} stale chunks for '{filename}'")
        except Exception as e:
            logging.error(f"Error deleting chunks for '{filename}': {e}")
            raise

    def process_and_add_documents(self, chunks, filename: str):
        # Always purge stale chunks first — prevents ghost results from old runs
        self.delete_by_filename(filename)
        ids, texts, metadatas = self.data_pre_processing(chunks, filename)
        self.add_to_collection(ids, texts, metadatas, filename)

    def formated_context_with_sources(self, results, filename):
        try:
            if not results:
                return "", []
            context = "\n\n".join(item["documents"] for item in results)
            sources = [
                f"{item['metadatas']['source']} (chunk {item['metadatas']['chunk']})"
                for item in results
            ]
            return context, sources
        except Exception as e:
            logging.exception(f"Error formatting {filename}: {e}")
            raise

    def retrive_text(self, query: str, filename: str, n_results: int = 3):
        try:
            # Count chunks belonging to this file only
            file_chunks = self.collection.get(where={"source": {"$eq": filename}})
            file_count  = len(file_chunks["ids"])

            if file_count == 0:
                logging.warning(f"No chunks found in collection for '{filename}'")
                return "", []

            n_results = min(n_results, file_count)

            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"source": {"$eq": filename}},
            )

            logging.info(f"Data retrieved from chromadb for '{filename}'\n data: {results}")

            filtered = [
                {
                    "ids":       results["ids"][0][i],
                    "documents": results["documents"][0][i],
                    "metadatas": results["metadatas"][0][i],
                    "distances": results["distances"][0][i],
                }
                for i in range(len(results["ids"][0]))
            ]

            return self.formated_context_with_sources(filtered, filename)

        except Exception as e:
            logging.error(f"Error retrieving '{filename}': {e}")
            raise