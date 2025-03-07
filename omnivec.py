import os

from FlagEmbedding import BGEM3FlagModel, FlagReranker


class Omnivec:
    def __init__(self):
        self.__embedding_model = BGEM3FlagModel(
            model_name_or_path=os.getenv("EMBEDDING_MODEL_MODEL_NAME", "BAAI/bge-m3"),
            normalize_embeddings=os.getenv("EMBEDDING_MODEL_NORMALIZE_EMBEDDINGS", True),
            use_fp16=os.getenv("EMBEDDING_MODEL_USE_FP16", True),
            query_instruction_for_retrieval=os.getenv("EMBEDDING_MODEL_QUERY_INSTRUCTION_FOR_RETRIEVAL", None),
            query_instruction_format=os.getenv("EMBEDDING_MODEL_QUERY_INSTRUCTION_FORMAT", "{}{}"),
            devices=os.getenv("EMBEDDING_MODEL_INFERENCE_DEVICES", ['cuda:0']),
            pooling_method=os.getenv("EMBEDDING_MODEL_POOLING_METHOD", "cls"),
            trust_remote_code=os.getenv("EMBEDDING_MODEL_TRUST_REMOTE_CODE", True),
            cache_dir=os.getenv("EMBEDDING_MODEL_CACHE_DIR", "./cache"),
            colbert_dim=int(os.getenv("EMBEDDING_MODEL_COLBERT_DIM", -1)),
            batch_size=int(os.getenv("EMBEDDING_MODEL_BATCH_SIZE", 128)),
            query_max_length=int(os.getenv("EMBEDDING_MODEL_QUERY_MAX_LENGTH", 1024)),
            passage_max_length=int(os.getenv("EMBEDDING_MODEL_PASSAGE_MAX_LENGTH", 1024)),
            return_dense=os.getenv("EMBEDDING_MODEL_RETURN_DENSE", True) in ('true', '1', 't'),
            return_sparse=os.getenv("EMBEDDING_MODEL_RETURN_SPARSE", True) in ('true', '1', 't'),
            return_colbert_vecs=os.getenv("EMBEDDING_MODEL_RETURN_COLBERT_VECS", False) in ('true', '1', 't'),
        )
        self.__reranking_model = FlagReranker(
            model_name_or_path=os.getenv("RERANKER_MODEL_NAME_OR_PATH", "BAAI/bge-reranker-v2-m3"),
            use_fp16=os.getenv("RERANKER_USE_FP16", 'False').lower() in ('true', '1', 't'),
            query_instruction_for_rerank=os.getenv("RERANKER_QUERY_INSTRUCTION_FOR_RERANK", None),
            query_instruction_format=os.getenv("RERANKER_QUERY_INSTRUCTION_FORMAT", "{}{}"),
            passage_instruction_for_rerank=os.getenv("RERANKER_PASSAGE_INSTRUCTION_FOR_RERANK", None),
            passage_instruction_format=os.getenv("RERANKER_PASSAGE_INSTRUCTION_FORMAT", "{}{}"),
            trust_remote_code=os.getenv("RERANKER_TRUST_REMOTE_CODE", False) in ('true', '1', 't'),
            cache_dir=os.getenv("RERANKER_CACHE_DIR", "./cache"),
            devices=os.getenv("RERANKER_DEVICES", None),
            batch_size=int(os.getenv("RERANKER_BATCH_SIZE", 128)),
            query_max_length=int(os.getenv("RERANKER_QUERY_MAX_LENGTH", 1024)) if os.getenv(
                "RERANKER_QUERY_MAX_LENGTH") else None,
            max_length=int(os.getenv("RERANKER_MAX_LENGTH", 1024)),
            normalize=os.getenv("RERANKER_NORMALIZE", 'False') in ('true', '1', 't')
        )

    def encode_passages(self, corpus: list[str] | str, batch_size: int = 128, max_length: int = 1024, dense: bool = True,
                        sparse: bool = True, late_interaction: bool = False) -> dict:
        data = self.__embedding_model.encode_corpus(
            corpus=corpus,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=dense,
            return_sparse=sparse,
            return_colbert_vecs=late_interaction
        )
        sparse=[]

        for item in data['lexical_weights']:
            item_indices = []
            item_values = []
            for key, value in item.items():
                item_indices.append(int(key))
                item_values.append(value)
            sparse.append({
                "indices": item_indices,
                "values": item_values
            })

        return {
            "dense": data['dense_vecs'],
            "sparse": sparse,
            "late": data['colbert_vecs']
        }

    def encode_queries(self, queries: list[str] | str, batch_size: int = 128, max_length: int = 1024, dense: bool = True,
                       sparse: bool = True, late_interaction: bool = False) -> dict:
        data = self.__embedding_model.encode_queries(
            queries=queries,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=dense,
            return_sparse=sparse,
            return_colbert_vecs=late_interaction
        )
        sparse = []

        for item in data['lexical_weights']:
            item_indices = []
            item_values = []
            print(item)
            for key, value in item.items():
                item_indices.append(int(key))
                item_values.append(value)
            sparse.append({
                "indices": item_indices,
                "values": item_values
            })

        return {
            "dense": data['dense_vecs'],
            "sparse": sparse,
            "late": data['colbert_vecs']
        }

    def compute_score(self, query:str,passages:list[str], batch_size: int = 128,
                      max_length: int = 1024, normalize: bool = True):
        sentencepairs= []
        for passage in passages:
           sentencepairs.append((query,passage))

        return  self.__reranking_model.compute_score(
            sentence_pairs=sentencepairs,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize)

