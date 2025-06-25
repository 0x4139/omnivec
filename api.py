from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from chunker import Chunker
from omnivec import Omnivec
import numpy as np

# Motto
"""
OmniVec: Versatile Embeddings for Smarter AI
"""

app = FastAPI(title="OmniVec API", description="API for generating dense, sparse, and late interaction embeddings.",
              openapi_version="3.0.0")
app.openapi_version = "3.0.0"
omnivec = Omnivec()
chunker = Chunker()


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
    dense: list[list[float]]  # Dense vector representation
    sparse: list[SparseVector]  # Sparse vector representation
    late: list[list[list[float]]]  # Late interaction representation


class EmbeddingRequest(BaseModel):
    """
    Request model for generating embeddings.
    """
    corpus: list[str] = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit."]  # List of text passages or queries to encode
    batch_size: int = 128  # Number of samples processed per batch
    max_length: int = 1024  # Maximum token length for input sequences


class OmnivecException(Exception):
    def __init__(self, name: str, message: str):
        self.name = name
        self.message = message


class ErrorResponse(BaseModel):
    name: str
    message: str


@app.exception_handler(OmnivecException)
async def custom_exception_handler(request: Request, exc: OmnivecException):
    return JSONResponse(
        status_code=418,
        content=ErrorResponse(name=exc.name, message=exc.message).model_dump()
    )


@app.post("/v1/embedding/passages/", response_model=EmbeddingResponse)
def encode_passages(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Encode a list of passages into dense, sparse, and/or late interaction embeddings.

    - **corpus**: List of text passages to encode.
    - **batch_size**: Number of passages processed per batch.
    - **max_length**: Maximum sequence length per passage.
    """
    try:
        data = omnivec.encode_passages(request.corpus, request.batch_size, request.max_length)
        return EmbeddingResponse(
            dense=custom_ndarray_encoder(data['dense']),
            late=custom_ndarray_encoder(data['late']),
            sparse=data['sparse']
        )
    except Exception as e:
        raise OmnivecException("EmbeddingError", str(e))


@app.post("/v1/embedding/queries/", response_model=EmbeddingResponse)
def encode_queries(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Encode a list of queries into dense, sparse, and/or late interaction embeddings.

    - **corpus**: List of query texts to encode.
    - **batch_size**: Number of queries processed per batch.
    - **max_length**: Maximum sequence length per query.
    """
    try:
        data = omnivec.encode_queries(request.corpus, request.batch_size, request.max_length)
        return EmbeddingResponse(
            dense=custom_ndarray_encoder(data['dense']),
            late=custom_ndarray_encoder(data['late']),
            sparse=data['sparse'])
    except Exception as e:
        raise OmnivecException("EmbeddingError", str(e))


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
    try:
        scores = omnivec.compute_score(request.query, request.passages, request.batch_size, request.max_length,
                                       request.normalize)
        return RerankingResponse(scores=custom_ndarray_encoder(scores))
    except Exception as e:
        raise OmnivecException("RerankingError", str(e))


class ChunkingRequest(BaseModel):
    document: str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    capacity: int = 1500
    trim: bool = True
    overlap: int = 150


class ChunkingResponse(BaseModel):
    chunks: list[str]


@app.post("/v1/chunking/markdown", response_model=ChunkingResponse)
def chunk_markdown(request: ChunkingRequest):
    try:
        return ChunkingResponse(
            chunks=chunker.markdown(text=request.document, capacity=request.capacity, trim=request.trim, overlap=request.overlap))
    except Exception as e:
        raise OmnivecException("ChunkingError", str(e))

@app.post("/v1/chunking/text", response_model=ChunkingResponse)
def chunk_text(request:ChunkingRequest):
    try:
        return ChunkingResponse(
            chunks=chunker.text(text=request.document, capacity=request.capacity, trim=request.trim,
                                    overlap=request.overlap))
    except Exception as e:
        raise OmnivecException("ChunkingError", str(e))
