import json
import unittest
from typing import Any

from backend import config
from backend.schemas import KnowledgeBaseReference
from backend.services.knowledge_base_service import KnowledgeBaseService


class StubRerankService(KnowledgeBaseService):
    def __init__(
        self,
        *,
        response_payload: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.response_payload = response_payload or {}
        self.error = error
        self.last_payload: dict[str, Any] | None = None

    def _call_rerank_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.last_payload = payload
        if self.error is not None:
            raise self.error
        return self.response_payload


class KnowledgeBaseRerankTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_config = {
            "RERANK_ENABLED": config.RERANK_ENABLED,
            "RERANK_TOP_N": config.RERANK_TOP_N,
            "RERANK_SCORE_THRESHOLD": config.RERANK_SCORE_THRESHOLD,
            "RERANK_INSTRUCT": config.RERANK_INSTRUCT,
        }

    def tearDown(self) -> None:
        for key, value in self.original_config.items():
            setattr(config, key, value)

    def test_rerank_orders_candidates_by_relevance_score(self) -> None:
        config.RERANK_ENABLED = True
        config.RERANK_TOP_N = 2
        config.RERANK_SCORE_THRESHOLD = None
        service = StubRerankService(
            response_payload={
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.91},
                        {"index": 0, "relevance_score": 0.32},
                    ]
                }
            }
        )
        candidates = [
            self._candidate("doc-a", "A", 0.8),
            self._candidate("doc-b", "B", 0.7),
            self._candidate("doc-c", "C", 0.6),
        ]

        selected, rerank_log = service._select_retrieval_candidates(
            "question", candidates, top_k=1
        )

        self.assertEqual(["doc-b", "doc-a"], [item["document_id"] for item in selected])
        self.assertEqual([0.91, 0.32], [item["score"] for item in selected])
        self.assertEqual("succeeded", rerank_log["status"])
        self.assertEqual(3, len(rerank_log["pre_rerank_chunks"]))
        self.assertEqual(["doc-b:0", "doc-a:0"], [
            item["chunk_id"] for item in rerank_log["post_rerank_chunks"]
        ])
        self.assertNotIn("text", rerank_log["pre_rerank_chunks"][0])
        self.assertNotIn("text", rerank_log["post_rerank_chunks"][0])
        self.assertEqual(
            {"knowledge_base_id", "knowledge_base_name", "chunk_id"},
            set(rerank_log["pre_rerank_chunks"][0].keys()),
        )
        self.assertEqual(
            {"knowledge_base_id", "knowledge_base_name", "chunk_id"},
            set(rerank_log["post_rerank_chunks"][0].keys()),
        )
        self.assertEqual({"text": "question"}, service.last_payload["input"]["query"])
        self.assertEqual({"text": "B"}, service.last_payload["input"]["documents"][1])

    def test_rerank_failure_falls_back_to_vector_top_k_per_knowledge_base(self) -> None:
        config.RERANK_ENABLED = True
        config.RERANK_TOP_N = 2
        service = StubRerankService(error=RuntimeError("timeout"))
        candidates = [
            self._candidate("kb-1-a", "A1", 0.8, knowledge_base_id="kb-1"),
            self._candidate("kb-1-b", "A2", 0.7, knowledge_base_id="kb-1"),
            self._candidate("kb-2-a", "B1", 0.6, knowledge_base_id="kb-2"),
            self._candidate("kb-2-b", "B2", 0.5, knowledge_base_id="kb-2"),
        ]

        selected, rerank_log = service._select_retrieval_candidates(
            "question", candidates, top_k=1
        )

        self.assertEqual(["kb-1-a", "kb-2-a"], [item["document_id"] for item in selected])
        self.assertEqual("failed", rerank_log["status"])
        self.assertEqual("RuntimeError", rerank_log["error_type"])
        self.assertEqual(
            ["kb-1-a:0", "kb-2-a:0"],
            [item["chunk_id"] for item in rerank_log["post_rerank_chunks"]],
        )
        self.assertNotIn("text", rerank_log["pre_rerank_chunks"][0])
        self.assertNotIn("text", rerank_log["post_rerank_chunks"][0])

    def test_context_char_limit_truncates_final_chunks(self) -> None:
        service = KnowledgeBaseService()
        candidates = [
            self._candidate("doc-a", "abcdef", 0.8),
            self._candidate("doc-b", "gh", 0.7),
        ]

        retrieved_chunks = service._candidates_to_retrieved_chunks(
            candidates, context_char_limit=4
        )

        self.assertEqual(1, len(retrieved_chunks))
        self.assertEqual("abcd", retrieved_chunks[0]["text"])
        self.assertEqual(0.8, retrieved_chunks[0]["score"])

    def test_rerank_score_threshold_filters_low_score_results(self) -> None:
        config.RERANK_ENABLED = True
        config.RERANK_TOP_N = 2
        config.RERANK_SCORE_THRESHOLD = 0.5
        service = StubRerankService(
            response_payload={
                "output": {
                    "results": [
                        {"index": 0, "relevance_score": 0.4},
                        {"index": 1, "relevance_score": 0.8},
                    ]
                }
            }
        )
        candidates = [
            self._candidate("doc-a", "A", 0.8),
            self._candidate("doc-b", "B", 0.7),
        ]

        selected, rerank_log = service._select_retrieval_candidates(
            "question", candidates, top_k=1
        )

        self.assertEqual(["doc-b"], [item["document_id"] for item in selected])
        self.assertEqual("succeeded", rerank_log["status"])

    def test_hybrid_retrieval_log_item_accepts_dense_and_sparse_hits(self) -> None:
        service = KnowledgeBaseService()
        knowledge_base_ref = KnowledgeBaseReference(id="kb-1", name="KB")
        dense_hit = type(
            "DenseHit",
            (),
            {
                "score": 0.82,
                "payload": {
                    "document_id": "doc-a",
                    "original_filename": "doc-a.md",
                    "chunk_index": 0,
                },
            },
        )()
        sparse_hit = {
            "score": 1.23,
            "document_id": "doc-b",
            "original_filename": "doc-b.md",
            "chunk_index": 1,
        }

        log_item = service._build_retrieval_log_item(
            knowledge_base_ref,
            status="searched",
            dense_hits=[dense_hit],
            sparse_hits=[sparse_hit],
            fused_candidate_count=2,
            sparse_cache_status="hit",
            sparse_status="searched",
        )

        self.assertEqual(1, log_item["dense_hit_count"])
        self.assertEqual(1, log_item["sparse_hit_count"])
        self.assertEqual(2, log_item["fused_candidate_count"])
        self.assertEqual("hit", log_item["sparse_cache_status"])
        self.assertNotIn("hits", log_item)
        self.assertNotIn("sparse_hits", log_item)

    def test_rerank_disabled_log_keeps_candidates_as_post_rerank_chunks(self) -> None:
        config.RERANK_ENABLED = False
        service = KnowledgeBaseService()
        candidates = [
            self._candidate("doc-a", "A", 0.8),
            self._candidate("doc-b", "B", 0.7),
        ]

        selected, rerank_log = service._select_retrieval_candidates(
            "question", candidates, top_k=1
        )

        self.assertEqual(["doc-a", "doc-b"], [item["document_id"] for item in selected])
        self.assertEqual("disabled", rerank_log["status"])
        self.assertEqual(
            rerank_log["pre_rerank_chunks"], rerank_log["post_rerank_chunks"]
        )
        self.assertNotIn("text", rerank_log["post_rerank_chunks"][0])

    def test_retrieval_result_log_only_contains_chunk_metadata(self) -> None:
        service = KnowledgeBaseService()
        rerank_log = {
            "enabled": True,
            "status": "succeeded",
            "candidate_count": 1,
            "selected_count": 1,
            "pre_rerank_chunks": [
                service._serialize_chunk_log_item(self._candidate("doc-a", "A", 0.8))
            ],
            "post_rerank_chunks": [
                service._serialize_chunk_log_item(self._candidate("doc-a", "A", 0.8))
            ],
        }

        with self.assertLogs("uvicorn.error", level="INFO") as captured:
            service._log_retrieval_result(
                "question",
                [{"knowledge_base_id": "kb-1", "status": "searched"}],
                started_at=0.0,
                rerank_log=rerank_log,
            )

        payload = json.loads(captured.output[0].split("知识库检索结果:\n", 1)[1])
        self.assertEqual("question", payload["question"])
        self.assertIn("pre_rerank_chunks", payload["rerank"])
        self.assertIn("post_rerank_chunks", payload["rerank"])
        self.assertNotIn("retrieved_chunk_count", payload)
        self.assertNotIn("text", payload["rerank"]["pre_rerank_chunks"][0])
        self.assertNotIn("text", payload["rerank"]["post_rerank_chunks"][0])
        self.assertEqual(
            {"knowledge_base_id", "knowledge_base_name", "chunk_id"},
            set(payload["rerank"]["pre_rerank_chunks"][0].keys()),
        )
        self.assertEqual(
            {"knowledge_base_id", "knowledge_base_name", "chunk_id"},
            set(payload["rerank"]["post_rerank_chunks"][0].keys()),
        )

    def _candidate(
        self,
        document_id: str,
        text: str,
        score: float,
        *,
        knowledge_base_id: str = "kb-1",
    ) -> dict[str, Any]:
        return {
            "knowledge_base_id": knowledge_base_id,
            "knowledge_base_name": "KB",
            "document_id": document_id,
            "original_filename": f"{document_id}.md",
            "chunk_index": 0,
            "score": score,
            "vector_score": score,
            "text": text,
        }


if __name__ == "__main__":
    unittest.main()
