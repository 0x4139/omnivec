# OmniVec API - Versatile Embeddings for Smarter AI

## Motivation
**OmniVec** is designed to provide a **versatile embedding solution** for AI-powered applications. It supports **dense embeddings**, **sparse embeddings**, and **late interaction embeddings**, making it an adaptable tool for search, retrieval, and ranking tasks. With a built-in **reranking** module, OmniVec enables high-accuracy similarity scoring, improving performance across NLP and machine learning workflows.


## Description
OmniVec is a FastAPI-based solution for embedding text and reranking passages using state-of-the-art models. It leverages **FlagEmbedding** for embedding generation and **FlagReranker** for similarity computation. The API is designed to be flexible and efficient, making it ideal for large-scale deployments in search engines, recommendation systems, and AI applications.

## API Overview
The API exposes the following endpoints:

### 1. **Embedding API**
#### `POST /v1/embedding/passages/`
Encodes a list of text passages into **dense**, **sparse**, and/or **late interaction** embeddings.

- **Request Body:**
  ```json
  {
    "corpus": ["Example passage 1", "Example passage 2"],
    "batch_size": 128,
    "max_length": 1024,
    "dense": true,
    "sparse": true,
    "late_interaction": false
  }
  ```
- **Response:**
  ```json
  {
    "dense": [[...], [...]],
    "sparse": [{...}, {...}],
    "late": [[[...]], [[...]]]
  }
  ```

#### `POST /v1/embedding/queries/`
Encodes a list of **queries** into **dense**, **sparse**, and/or **late interaction** embeddings.

- **Request Body:** Same as `POST /v1/embedding/passages/`
- **Response:** Same as `POST /v1/embedding/passages/`

### 2. **Reranking API**
#### `POST /v1/reranking/compute`
Computes similarity scores between a **query** and a list of **passages**.

- **Request Body:**
  ```json
  {
    "query": "Example query",
    "passages": ["Passage 1", "Passage 2"],
    "batch_size": 128,
    "max_length": 1024,
    "normalize": true
  }
  ```
- **Response:**
  ```json
  {
    "scores": [0.85, 0.72]
  }
  ```

## Running with Docker
To quickly start OmniVec using Docker, run the following command:

```sh
docker volume create omnivec
docker run --gpus all -p 8000:8000 -v omnivec:/app/cache 0x4139/omnivec
```

Once the container is running, you can access the Open API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)

## Environment Variables
The following environment variables can be customized:

### **Embedding Model Configurations**
| Variable | Description | Default |
|----------|------------|---------|
| `EMBEDDING_MODEL_MODEL_NAME` | Model used for generating embeddings | `BAAI/bge-m3` |
| `EMBEDDING_MODEL_NORMALIZE_EMBEDDINGS` | Whether to normalize embeddings | `true` |
| `EMBEDDING_MODEL_USE_FP16` | Use FP16 for performance optimization | `true` |
| `EMBEDDING_MODEL_QUERY_INSTRUCTION_FOR_RETRIEVAL` | Query instruction format | `null` |
| `EMBEDDING_MODEL_QUERY_INSTRUCTION_FORMAT` | Query instruction template | `{}`{}" |
| `EMBEDDING_MODEL_INFERENCE_DEVICES` | Devices for inference (e.g., CPU/GPU) | `cuda:0` |
| `EMBEDDING_MODEL_POOLING_METHOD` | Pooling method for embedding | `cls` |
| `EMBEDDING_MODEL_CACHE_DIR` | Directory for caching model weights | `./cache` |
| `EMBEDDING_MODEL_BATCH_SIZE` | Batch size for embedding generation | `128` |
| `EMBEDDING_MODEL_QUERY_MAX_LENGTH` | Max token length for queries | `1024` |
| `EMBEDDING_MODEL_PASSAGE_MAX_LENGTH` | Max token length for passages | `1024` |
| `EMBEDDING_MODEL_RETURN_DENSE` | Whether to return dense embeddings | `true` |
| `EMBEDDING_MODEL_RETURN_SPARSE` | Whether to return sparse embeddings | `true` |
| `EMBEDDING_MODEL_RETURN_COLBERT_VECS` | Whether to return ColBERT vectors (late interaction) | `false` |

### **Reranker Model Configurations**
| Variable | Description | Default |
|----------|------------|---------|
| `RERANKER_MODEL_NAME_OR_PATH` | Model used for reranking | `BAAI/bge-reranker-v2-m3` |
| `RERANKER_USE_FP16` | Use FP16 for optimization | `false` |
| `RERANKER_QUERY_INSTRUCTION_FOR_RERANK` | Query instruction for reranking | `null` |
| `RERANKER_QUERY_INSTRUCTION_FORMAT` | Query instruction template | `{}`{}" |
| `RERANKER_PASSAGE_INSTRUCTION_FOR_RERANK` | Passage instruction for reranking | `null` |
| `RERANKER_PASSAGE_INSTRUCTION_FORMAT` | Passage instruction template | `{}`{}" |
| `RERANKER_BATCH_SIZE` | Batch size for reranking | `128` |
| `RERANKER_QUERY_MAX_LENGTH` | Max token length for queries | `1024` |
| `RERANKER_MAX_LENGTH` | Max token length for passages | `1024` |
| `RERANKER_NORMALIZE` | Normalize similarity scores | `true` |

## License
OmniVec is released under the **MIT License**.