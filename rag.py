from hashlib import sha256
import os
import chromadb

from chromadb.api.types import EmbeddingFunction, Documents, Embeddings, QueryResult
from typing import Any, Dict, List, cast
from data.document import Document
import torch.nn.functional as F
from datetime import datetime
from numpy import array, linalg, ndarray


def compute_similarity_scores(embedding: ndarray, embedding_array: ndarray) -> ndarray:
    """
    Compute the cosine similarity scores between a given embedding and an array of embeddings.

    Args:
        embedding (numpy.ndarray): The embedding vector.
        embedding_array (numpy.ndarray): The array of embedding vectors.

    Returns:
        numpy.ndarray: The array of similarity scores.

    Raises:
        ValueError: If the embedding or embedding_array is invalid.
    """
    query_norm = linalg.norm(embedding)
    collection_norm = linalg.norm(embedding_array, axis=1)
    valid_indices = (query_norm != 0) & (collection_norm != 0)
    similarity_scores = array([-1.0] * embedding_array.shape[0])

    if valid_indices.any():
        similarity_scores[valid_indices] = embedding.dot(
            embedding_array[valid_indices].T
        ) / (query_norm * collection_norm[valid_indices])
    else:
        raise ValueError(f"Invalid vectors: {embedding_array} or {embedding}")
    return similarity_scores


def query_results_to_records(results: QueryResult) -> List[Dict[str, Any]]:
    """
    Convert query results to memory records.

    Args:
        results (QueryResult): The query results.

    Returns:
        list: The list of memory records.
    """
    try:
        if isinstance(results["ids"][0], str):
            for k, v in results.items():
                results[k] = [v]
    except IndexError:
        return []
    memory_records = [
        {
            "document_name": metadata["document_name"],
            "id": metadata["id"],
            "text": document,
            "embedding": embedding,
            "key": id,
        }
        for id, document, embedding, metadata in zip(
            results["ids"][0],
            results["documents"][0],
            results["embeddings"][0],
            results["metadatas"][0],
        )
    ]
    return memory_records


class LocalEmbeddingFunction(EmbeddingFunction[Documents]):
    # Since we do dynamic imports we have to type this as Any
    models: Dict[str, Any] = {}

    # If you have a beefier machine, try "gtr-t5-large".
    # for a full list of options: https://huggingface.co/sentence-transformers, https://www.sbert.net/docs/pretrained_models.html
    def __init__(
            self,
            model_name="nomic-ai/nomic-embed-text-v1.5",
            device="cuda",
            matryoshka_dim=512,
            **kwargs,
    ):
        """Initialize SentenceTransformerEmbeddingFunction.

        Args:
            model_name (str, optional): Identifier of the SentenceTransformer model, defaults to "nomic-ai/nomic-embed-text-v1.5"
                    note that trust_remote_code=True is needed for the default model
            device (str, optional): Device used for computation, defaults to "gpu"
            normalize_embeddings (bool, optional): Whether to normalize returned vectors, defaults to False
            **kwargs: Additional arguments to pass to the SentenceTransformer model.
        """
        if model_name not in self.models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ValueError(
                    "The sentence_transformers python package is not installed. Please install it with `pip install sentence_transformers`"
                )
            self.models[model_name] = SentenceTransformer(
                model_name, device=device, **kwargs
            )
        self._model = self.models[model_name]
        self._matryoshka_dim = matryoshka_dim

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self._model.encode(input, convert_to_tensor=True)
        if isinstance(input, str) == 1:
            embeddings = embeddings.unsqueeze(0)
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        embeddings = embeddings[:, :self._matryoshka_dim]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return cast(Embeddings, embeddings.tolist())


class Memory:
    """
    A class representing a memory database.

    Attributes:
        _collections (dict): A dictionary to store the collections in the database.
        _embedder (LocalEmbeddingFunction): An instance of the LocalEmbeddingFunction class for embedding documents.
        _chroma_client (PersistentClient): An instance of the PersistentClient class for interacting with the database.
        _chunk_size (int): The size of each chunk when splitting documents.
        _overlap (int): The overlap between consecutive chunks.

    Methods:
        __init__: Initializes a new instance of the Memory class.
        _get_collection: Retrieves or creates a collection in the database.
        save_documents_to_db: Saves documents to the specified collection in the database.
    """

    def __init__(
            self,
            db_path="embedding_database",
            chunk_size=512,
            overlap=10,
    ):
        """
        Initializes a new instance of the Memory class.

        Args:
            db_path (str): The path to the database directory. Defaults to "embedding_database".
            chunk_size (int): The size of each chunk when splitting documents. Defaults to 512.
            overlap (int): The overlap between consecutive chunks. Defaults to 10.

        Returns:
            None
        """
        self._collections = None
        memories_dir = os.path.join(os.getcwd(), db_path)
        if not os.path.exists(memories_dir):
            os.makedirs(memories_dir)
        self._embedder = LocalEmbeddingFunction(device="cuda", trust_remote_code=True)
        self._chroma_client = chromadb.PersistentClient(path=memories_dir)
        self._chunk_size = chunk_size
        self._overlap = overlap

    def create_collection(self, collection_name):
        """
        Creates a new collection in the database if it is not already present.

        Args:
            collection_name (str): The name of the collection to create.

        Returns:
            None
        """
        self._get_collection(collection_name)

    def _get_collection(self, collection_name):
        """
        Retrieves or creates a collection in the database.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            chromadb.Collection: The collection object.
        """
        if self._collections is None:
            self._collections = {}
        if collection_name not in self._collections:
            self._collections[collection_name] = self._chroma_client.get_or_create_collection(collection_name,
                                                                                              embedding_function=self._embedder)
        return self._collections[collection_name]

    def save_documents_to_db(self, collection_name, documents):
        """
        Save the given documents to the specified collection in the database.

        Args:
            collection_name (str): The name of the collection in the database.
            documents (List[Document]): A list of Document objects to be saved.

        Returns:
            None
        """
        collection = self._get_collection(collection_name)
        chunks = []
        metadatas = []
        if isinstance(documents, Document):
            documents = [documents]
        for doc in documents:
            chunked_doc = [doc.text[i:i + self._chunk_size] for i in
                           range(0, len(doc.text), self._chunk_size - self._overlap)]
            for chunk in chunked_doc:
                metadata = {
                    "document_name": doc.name,
                    "id": sha256(
                        (chunk + datetime.now().isoformat()).encode()
                    ).hexdigest(),
                }
                metadatas.append(metadata)
                chunks.append(chunk)
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=[m["id"] for m in metadatas],
        )

    def get_memories_data(
            self,
            user_input,
            collection_name,
            limit_to_document,
            limit,
            min_relevance_score=0.0,
    ):
        """
        Retrieve memories data based on the given user input and search criteria.

        Args:
            user_input (str): The user input used for searching memories.
            collection_name (str): The name of the collection to search in.
            limit_to_document (str): The document name to limit the search to.
            limit (int): The maximum number of results to return.
            min_relevance_score (float, optional): The minimum relevance score for a result to be included.
                Defaults to 0.0.

        Returns:
            List[dict]: A list of dictionaries representing the top memories data matching the search criteria.
        """
        if not user_input:
            return []
        collection = self._get_collection(collection_name)
        if collection.count() == 0:
            return []
        embedding = array(self._embedder(user_input))
        results = collection.query(
            query_embeddings=embedding.tolist(),
            n_results=limit,
            where={"document_name": {"$eq": limit_to_document}} if limit_to_document else {},
            include=["embeddings", "metadatas", "documents"],
        )
        embedding_array = array(results["embeddings"][0])
        if len(embedding_array) == 0:
            return []
        embedding_array = embedding_array.reshape(embedding_array.shape[0], -1)
        if len(embedding.shape) == 2:
            embedding = embedding.reshape(
                embedding.shape[1],
            )
        similarity_score = compute_similarity_scores(
            embedding=embedding, embedding_array=embedding_array
        )
        record_list = []
        for record, score in zip(query_results_to_records(results), similarity_score):
            record["relevance_score"] = score
            record_list.append(record)
        sorted_results = sorted(
            record_list, key=lambda x: x["relevance_score"], reverse=True
        )
        filtered_results = [
            x for x in sorted_results if x["relevance_score"] >= min_relevance_score
        ]
        top_results = filtered_results[:limit]
        return top_results

    def get_memories(
            self,
            user_input,
            collection_name,
            limit_to_document=None,
            limit=5,
            min_relevance_score=0.0,
    ):
        """
        Retrieves memories based on the given parameters.

        Args:
            user_input (str): The user input used for retrieving memories.
            collection_name (str): The name of the collection to search in.
            limit_to_document (str): The document to limit the search to.
            limit (int): The maximum number of memories to retrieve.
            min_relevance_score (float, optional): The minimum relevance score for retrieved memories. Defaults to 0.0.
                It's recommended to keep this value 0.0 because the score is calculated manually.

        Returns:
            List[str]: A list of retrieved memories.

        """
        results = self.get_memories_data(
            user_input=user_input,
            collection_name=collection_name,
            limit_to_document=limit_to_document,
            limit=limit,
            min_relevance_score=min_relevance_score,
        )
        response = []
        if results:
            for result in results:
                metadata = (
                    result["text"]
                    if "text" in result
                    else ""
                )
                external_source = (
                    result["document_name"]
                    if "document_name" in result
                    else None
                )
                if external_source:
                    metadata = f"Sourced from {external_source}:\n{metadata}"
                if metadata not in response and metadata != "":
                    response.append(metadata)
        return response
