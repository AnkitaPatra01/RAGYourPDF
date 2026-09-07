import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchAny, MatchValue, Range, PayloadSchemaType)

load_dotenv()

class QdrantStorage:
    def __init__(self, url: str = None, collection: str = None, dim: int = None):
        url = url or os.getenv("QDRANT_URL")
        collection = collection or os.getenv("QDRANT_COLLECTION")
        dim = dim or int(os.getenv("QDRANT_VECTOR_DIM"))
        api_key = os.getenv("QDRANT_API_KEY")
        client_kwargs = {
            "url": url,
            "timeout": 30,
        }

        if api_key:
            client_kwargs["api_key"] = api_key

        self.client = QdrantClient(**client_kwargs)

        self.collection = collection

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                )
            )

        collection_info = self.client.get_collection(
            collection_name=self.collection
        )

        payload_schema = collection_info.payload_schema or {}

        if "expires_at" not in payload_schema:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="expires_at",
                field_schema=PayloadSchemaType.FLOAT
            )

        if "source" not in payload_schema:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD
            )

        if "session_id" not in payload_schema:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="session_id",
                field_schema=PayloadSchemaType.KEYWORD
            )

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i]
            )
            for i in range(len(ids))
        ]

        self.client.upsert(
            collection_name=self.collection,
            points=points
        )


    def search(
        self,
        query_vector,
        top_k: int = 5,
        source_ids: list[str] | None = None
    ):

        now = datetime.now(
            timezone.utc
        ).timestamp()

        must_conditions = [
            FieldCondition(
                key="expires_at",
                range=Range(gt=now)
            )
        ]

        if source_ids:

            must_conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchAny(
                        any=source_ids
                    )
                )
            )

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=Filter(
                must=must_conditions
            ),
            with_payload=True,
            limit=top_k
        ).points

        contexts = []

        sources = set()

        for result in results:
            payload = result.payload or {}

            text = payload.get(
                "text",
                ""
            )

            source = payload.get(
                "source",
                ""
            )

            if text:

                contexts.append(text)

                if source:
                    sources.add(source)

        return {
            "contexts": contexts,
            "sources": list(sources)
        }


    def refresh_session(
        self,
        session_id: str,
        expires_at: float
    ):

        self.client.set_payload(
            collection_name=self.collection,
            payload={
                "expires_at": expires_at
            },
            points=Filter(
                must=[
                    FieldCondition(
                        key="session_id",
                        match=MatchValue(
                            value=session_id
                        )
                    )
                ]
            )
        )


    def delete_sources(
        self,
        source_ids: list[str]
    ):

        if not source_ids:
            return

        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchAny(
                            any=source_ids
                        )
                    )
                ]
            )
        )


    def delete_session(
        self,
        session_id: str
    ):

        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="session_id",
                        match=MatchValue(
                            value=session_id
                        )
                    )
                ]
            )
        )


    def delete_all(self):

        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter()
        )


    def delete_expired_sessions(self):

        now = datetime.now(
            timezone.utc
        ).timestamp()

        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="expires_at",
                        range=Range(
                            lt=now
                        )
                    )
                ]
            )
        )