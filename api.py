from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from omnivec import Omnivec
import numpy as np

# Motto
"""
OmniVec: Versatile Embeddings for Smarter AI
"""

app = FastAPI(title="OmniVec API", description="API for generating dense, sparse, and late interaction embeddings.")
omnivec = Omnivec()


def custom_ndarray_encoder(obj):
    """
    Custom JSON encoder for handling NumPy arrays and nested lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [custom_ndarray_encoder(x) for x in obj]
    return obj


class SparseVector(BaseModel):
    indices: list[int]  # Sparse vector indices
    values: list[float]  # Sparse vector values

class EmbeddingResponse(BaseModel):
    """
    Response model for embeddings.
    """
    dense: Optional[list[list[float]]] = None  # Dense vector representation
    sparse: Optional[list[SparseVector]] = None  # Sparse vector representation
    late: Optional[list[list[list[float]]]] = None  # Late interaction representation

class EmbeddingRequest(BaseModel):
    """
    Request model for generating embeddings.
    """
    corpus: list[str]  # List of text passages or queries to encode
    batch_size: int = 128  # Number of samples processed per batch
    max_length: int = 1024  # Maximum token length for input sequences
    dense: bool = True  # Whether to generate dense embeddings
    sparse: bool = True  # Whether to generate sparse embeddings
    late_interaction: bool = False  # Whether to generate late interaction embeddings


@app.exception_handler(Exception)
async def my_custom_error_handler(request: Request, exc: Exception):
    """
    Custom exception handler that returns a JSON response with error details.
    """
    return JSONResponse(
        status_code=418,  # Can be modified based on the error type
        content={"message": str(exc)},
    )


@app.post("/v1/embedding/passages/", response_model=EmbeddingResponse)
def encode_passages(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Encode a list of passages into dense, sparse, and/or late interaction embeddings.

    - **corpus**: List of text passages to encode.
    - **batch_size**: Number of passages processed per batch.
    - **max_length**: Maximum sequence length per passage.
    - **dense**: Generate dense embeddings if True.
    - **sparse**: Generate sparse embeddings if True.
    - **late_interaction**: Generate late interaction embeddings if True.
    """
    data = omnivec.encode_passages(request.corpus, request.batch_size, request.max_length, request.dense,
                                   request.sparse, request.late_interaction)
    return EmbeddingResponse(
        dense=custom_ndarray_encoder(data['dense']),
        late=custom_ndarray_encoder(data['late']),
        sparse=data['sparse']
    )


@app.post("/v1/embedding/queries/", response_model=EmbeddingResponse)
def encode_queries(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Encode a list of queries into dense, sparse, and/or late interaction embeddings.

    - **corpus**: List of query texts to encode.
    - **batch_size**: Number of queries processed per batch.
    - **max_length**: Maximum sequence length per query.
    - **dense**: Generate dense embeddings if True.
    - **sparse**: Generate sparse embeddings if True.
    - **late_interaction**: Generate late interaction embeddings if True.
    """
    data = omnivec.encode_queries(request.corpus, request.batch_size, request.max_length, request.dense, request.sparse,
                                  request.late_interaction)
    return EmbeddingResponse(
        dense=custom_ndarray_encoder(data['dense']),
        late=custom_ndarray_encoder(data['late']),
        sparse=data['sparse']
    )


class RerankingRequest(BaseModel):
    """
    Request model for computing similarity scores between a query and passages.
    """
    query: str  # Query text for similarity scoring
    passages: list[str]  # List of passages to be compared to the query
    batch_size: int = 128  # Number of passages processed per batch
    max_length: int = 1024  # Maximum sequence length for input
    normalize: bool = True  # Whether to normalize similarity scores


class RerankingResponse(BaseModel):
    """
    Response model for reranking scores.
    """
    scores: list[float]  # List of similarity scores for each passage


@app.post("/v1/reranking/compute", response_model=RerankingResponse)
def compute_score(request: RerankingRequest) -> RerankingResponse:
    """
    Compute similarity scores between a query and a list of passages.

    - **query**: Query text.
    - **passages**: List of text passages to compare.
    - **batch_size**: Number of passages processed per batch.
    - **max_length**: Maximum sequence length.
    - **normalize**: Normalize similarity scores if True.
    """
    scores = omnivec.compute_score(request.query, request.passages, request.batch_size, request.max_length,
                                   request.normalize)
    return RerankingResponse(scores=custom_ndarray_encoder(scores))