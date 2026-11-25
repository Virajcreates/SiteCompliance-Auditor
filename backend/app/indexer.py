"""Hybrid Pinecone + Local JSON/NumPy EHS Vector Store & Embedder for SieveAudit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from google import genai
from google.genai import types
import numpy as np
from pinecone import Pinecone, ServerlessSpec

from backend.app.config import (
    EMBEDDING_DIMENSION,
    GEMINI_EMBEDDING_MODEL,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    settings,
)
from backend.app.schemas import AuditedWorkEpisodeRecord


class ComplianceEmbedder:
    """Generates dense vector embeddings for EHS episodes using Gemini Embedding SDK or local hash fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = GEMINI_EMBEDDING_MODEL,
        dimension: int = EMBEDDING_DIMENSION,
        client: Any | None = None,
        use_mock_embeddings: bool = False,
    ) -> None:
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name
        self.dimension = dimension
        self.client = client
        self.use_mock_embeddings = use_mock_embeddings

    @staticmethod
    def deterministic_hash_embedding(text: str, dimension: int = 768) -> list[float]:
        """Tokenizes words and bigrams (lowercased), hashes them into an L2-normalized float vector.

        Guarantees accurate cosine similarity ranking for domain keywords (scaffolding, harness, hardhat, excavator)
        in offline and mock execution environments.
        """
        words = re.findall(r"\w+", text.lower())
        if not words:
            vec = np.zeros(dimension, dtype=np.float32)
            vec[0] = 1.0
            return vec.tolist()

        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        tokens = words + bigrams

        vec = np.zeros(dimension, dtype=np.float32)
        for token in tokens:
            h = hashlib.sha256(token.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % dimension
            sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
            vec[idx] += sign

        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec.tolist()

    def embed_text(
        self, text: str, task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[float]:
        """Embeds a single string into a dense float vector."""
        if self.use_mock_embeddings or (self.client is None and not self.api_key):
            return self.deterministic_hash_embedding(text, self.dimension)

        try:
            if self.client is None:
                self.client = genai.Client(api_key=self.api_key)
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.dimension,
                ),
            )
            if response.embeddings and len(response.embeddings) > 0:
                vals = response.embeddings[0].values
                return list(vals)
        except Exception:
            pass

        return self.deterministic_hash_embedding(text, self.dimension)


class SieveAuditVectorIndex:
    """Hybrid Pinecone + Local JSON vector store with domain-specific EHS filtering."""

    def __init__(
        self,
        embedder: ComplianceEmbedder | None = None,
        pinecone_api_key: str | None = None,
        index_name: str = PINECONE_INDEX_NAME,
        dimension: int = EMBEDDING_DIMENSION,
        local_index_path: str | Path | None = None,
        pinecone_client: Any | None = None,
    ) -> None:
        self.embedder = embedder or ComplianceEmbedder()
        self.dimension = dimension
        self.index_name = index_name
        self.local_index_path = Path(
            local_index_path or (settings.DATA_DIR / "vector_index.json")
        ).resolve()
        self._local_records: dict[str, dict[str, Any]] = {}
        self._load_local_index()

        self.pinecone_api_key = pinecone_api_key or settings.PINECONE_API_KEY
        self._pc_client = pinecone_client
        self._pc_index: Any | None = None

        if self._pc_client is not None or (
            self.pinecone_api_key and self.pinecone_api_key.strip()
        ):
            try:
                if self._pc_client is None:
                    self._pc_client = Pinecone(api_key=self.pinecone_api_key)

                # Check if serverless index already exists
                existing_indexes: list[str] = []
                idx_list = self._pc_client.list_indexes()
                for idx in idx_list:
                    if hasattr(idx, "name"):
                        existing_indexes.append(idx.name)
                    elif isinstance(idx, dict) and "name" in idx:
                        existing_indexes.append(idx["name"])
                    elif isinstance(idx, str):
                        existing_indexes.append(idx)

                if self.index_name not in existing_indexes:
                    self._pc_client.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                    )

                self._pc_index = self._pc_client.Index(self.index_name)
            except Exception:
                # If Pinecone serverless init fails or is offline, continue with local JSON mirror
                self._pc_index = None

    def _load_local_index(self) -> None:
        if self.local_index_path.is_file():
            try:
                data = json.loads(
                    self.local_index_path.read_text(encoding="utf-8")
                )
                if isinstance(data, dict):
                    self._local_records = data
            except Exception:
                self._local_records = {}

    def _save_local_index(self) -> None:
        self.local_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_index_path.write_text(
            json.dumps(self._local_records, indent=2), encoding="utf-8"
        )

    def upsert_episodes(
        self,
        records: list[AuditedWorkEpisodeRecord],
        namespace: str = PINECONE_NAMESPACE,
    ) -> dict[str, Any]:
        """Embeds and indexes audited episodes into local JSON store and Pinecone (if configured)."""
        vector_ids: list[str] = []
        vectors_to_upsert: list[dict[str, Any]] = []

        for record in records:
            vid = f"{record.video_id}_ep_{record.episode_index:03d}"
            vector_ids.append(vid)

            emb_text = record.to_embedding_text()
            vec = self.embedder.embed_text(
                emb_text, task_type="RETRIEVAL_DOCUMENT"
            )
            meta = record.to_vector_metadata()

            # Ensure violation_types list is always present for instant filtering
            if "violation_types" not in meta:
                meta["violation_types"] = [
                    str(v.violation_type) for v in record.safety_violations
                ]

            entry = {
                "id": vid,
                "values": vec,
                "metadata": meta,
                "namespace": namespace,
            }
            self._local_records[vid] = entry
            vectors_to_upsert.append(
                {"id": vid, "values": vec, "metadata": meta}
            )

        self._save_local_index()

        backend_used = "local"
        if self._pc_index is not None:
            try:
                self._pc_index.upsert(
                    vectors=vectors_to_upsert, namespace=namespace
                )
                backend_used = "pinecone"
            except Exception:
                backend_used = "local_fallback"

        return {
            "upserted_count": len(records),
            "vector_ids": vector_ids,
            "backend": backend_used,
        }

    @staticmethod
    def _matches_filters(
        meta: dict[str, Any],
        video_id: str | None,
        site_zone: str | None,
        ncr_only: bool,
        violation_type: str | None,
        work_category: str | None,
        min_severity: str | None,
    ) -> bool:
        severity_ranks = {"low": 1, "medium": 2, "high": 3, "critical": 4}

        if video_id and meta.get("video_id") != video_id:
            return False

        if site_zone and meta.get("site_zone") != site_zone:
            return False

        if ncr_only:
            status = meta.get("episode_compliance_status")
            sev = str(meta.get("overall_severity", "")).lower()
            if status != "non_compliant_ncr" and sev not in ("high", "critical"):
                return False

        if violation_type:
            raw_viols = meta.get("violation_types", [])
            if isinstance(raw_viols, str):
                try:
                    raw_viols = json.loads(raw_viols)
                except Exception:
                    raw_viols = [
                        v.strip() for v in raw_viols.split(",") if v.strip()
                    ]
            if violation_type not in raw_viols:
                return False

        if work_category and meta.get("work_category") != work_category:
            return False

        if min_severity:
            req_rank = severity_ranks.get(min_severity.lower(), 1)
            item_sev = str(meta.get("overall_severity", "low")).lower()
            item_rank = severity_ranks.get(item_sev, 1)
            if item_rank < req_rank:
                return False

        return True

    def search(
        self,
        query: str,
        top_k: int = 5,
        video_id: str | None = None,
        site_zone: str | None = None,
        ncr_only: bool = False,
        violation_type: str | None = None,
        work_category: str | None = None,
        min_severity: str | None = None,
        namespace: str = PINECONE_NAMESPACE,
    ) -> list[dict[str, Any]]:
        """Performs semantic similarity search with multi-attribute EHS compliance filtering."""
        self._load_local_index()
        query_vec = self.embedder.embed_text(
            query, task_type="RETRIEVAL_QUERY"
        )
        query_arr = np.array(query_vec, dtype=np.float32)
        q_norm = float(np.linalg.norm(query_arr)) + 1e-9

        scored_candidates: list[tuple[float, str, dict[str, Any]]] = []

        # If Pinecone index is active, attempt Pinecone query
        if self._pc_index is not None:
            try:
                res = self._pc_index.query(
                    vector=query_vec,
                    top_k=top_k * 4,
                    include_metadata=True,
                    namespace=namespace,
                )
                matches = getattr(res, "matches", []) or []
                for m in matches:
                    mid = getattr(m, "id", None) or (
                        m.get("id") if isinstance(m, dict) else None
                    )
                    mscore = getattr(m, "score", 0.0) or (
                        m.get("score", 0.0) if isinstance(m, dict) else 0.0
                    )
                    mmeta = getattr(m, "metadata", {}) or (
                        m.get("metadata", {}) if isinstance(m, dict) else {}
                    )
                    if mid and self._matches_filters(
                        mmeta,
                        video_id,
                        site_zone,
                        ncr_only,
                        violation_type,
                        work_category,
                        min_severity,
                    ):
                        scored_candidates.append((float(mscore), str(mid), mmeta))
            except Exception:
                scored_candidates = []

        # If no candidates from Pinecone or local mode, compute via local JSON index
        if not scored_candidates and self._local_records:
            for vid, rec in self._local_records.items():
                if rec.get("namespace", namespace) != namespace:
                    continue
                meta = rec.get("metadata", {})
                if not self._matches_filters(
                    meta,
                    video_id,
                    site_zone,
                    ncr_only,
                    violation_type,
                    work_category,
                    min_severity,
                ):
                    continue

                val_arr = np.array(rec["values"], dtype=np.float32)
                v_norm = float(np.linalg.norm(val_arr)) + 1e-9
                sim = float(np.dot(query_arr, val_arr) / (q_norm * v_norm))
                scored_candidates.append((sim, vid, meta))

        # Sort descending by similarity score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_candidates[:top_k]

        results: list[dict[str, Any]] = []
        for score, vid, meta in top_candidates:
            meta_copy = dict(meta)

            # Deserialize JSON-encoded observation and violation records
            try:
                ppe_obs = json.loads(meta_copy.get("ppe_observations_json", "[]"))
            except Exception:
                ppe_obs = []

            try:
                safety_viols = json.loads(
                    meta_copy.get("safety_violations_json", "[]")
                )
            except Exception:
                safety_viols = []

            hit: dict[str, Any] = {
                "id": vid,
                "score": round(score, 4),
                "metadata": meta_copy,
                "ppe_observations": ppe_obs,
                "safety_violations": safety_viols,
            }

            # Populate top-level fields for ergonomic accessibility
            for k, v in meta_copy.items():
                if k not in hit:
                    hit[k] = v

            results.append(hit)

        return results

    def delete_video(
        self, video_id: str, namespace: str = PINECONE_NAMESPACE
    ) -> int:
        """Purges all indexed episode vectors associated with a given video_id."""
        matching_ids = [
            vid
            for vid, rec in self._local_records.items()
            if rec.get("metadata", {}).get("video_id") == video_id
            or vid.startswith(f"{video_id}_")
        ]
        count = len(matching_ids)
        for vid in matching_ids:
            self._local_records.pop(vid, None)

        self._save_local_index()

        if self._pc_index is not None and matching_ids:
            try:
                self._pc_index.delete(ids=matching_ids, namespace=namespace)
            except Exception:
                pass

        return count

