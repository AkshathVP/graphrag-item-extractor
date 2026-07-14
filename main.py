import asyncio
import json
import os
from dotenv import load_dotenv
from src.ingestion import AsyncTranscriptIngestor
from src.graph_rag import GraphRAGEngine
from src.visualize import export_interactive_graph
from src.models import MeetingExtraction, ActionItem, Entity, Relationship

load_dotenv()

def get_fallback_data() -> list[MeetingExtraction]:
    return [
        MeetingExtraction(
            meeting_id="MEETING_2026_001",
            action_items=[
                ActionItem(task_id="TASK_101", description="Complete OAuth2 token validation", assignee="David", deadline="Friday", dependencies=["Auth Migration"]),
                ActionItem(task_id="TASK_102", description="Integrate login spinner", assignee="Sarah", deadline="Next Monday", dependencies=["TASK_101"])
            ],
            entities=[
                Entity(name="David", label="PERSON"), Entity(name="Sarah", label="PERSON"),
                Entity(name="Security Team", label="TEAM"), Entity(name="Frontend Team", label="TEAM"),
                Entity(name="Auth Migration", label="PROJECT")
            ],
            relationships=[
                Relationship(source="David", target="Security Team", relation="PART_OF"),
                Relationship(source="Sarah", target="Frontend Team", relation="PART_OF"),
                Relationship(source="TASK_101", target="Auth Migration", relation="PART_OF")
            ]
        ),
        MeetingExtraction(
            meeting_id="MEETING_2026_002",
            action_items=[
                ActionItem(task_id="TASK_103", description="Monitor AWS load balancer", assignee="Alex", deadline="Tuesday", dependencies=["AWS Cluster Migration"]),
                ActionItem(task_id="TASK_104", description="Build vector indexing pipeline", assignee="Backend Team", deadline="End of Month", dependencies=[])
            ],
            entities=[
                Entity(name="Alex", label="PERSON"), Entity(name="DevOps Team", label="TEAM"),
                Entity(name="Backend Team", label="TEAM"), Entity(name="AWS Cluster Migration", label="PROJECT")
            ],
            relationships=[
                Relationship(source="Alex", target="DevOps Team", relation="PART_OF"),
                Relationship(source="TASK_103", target="DevOps Team", relation="ASSIGNED_TO")
            ]
        )
    ]

async def main():
    print("=== GraphRAG Item Extractor Pipeline Initialized ===")
    
    transcript_path = os.path.join("data", "sample_transcripts.json")
    if os.path.exists(transcript_path):
        with open(transcript_path, "r") as f:
            transcripts = json.load(f)
    else:
        transcripts = []

    api_key = os.getenv("GROQ_API_KEY")
    if api_key and transcripts:
        print(f"Connecting to Groq Asynchronous Inference (Llama 3.3)... Processing {len(transcripts)} transcripts.")
        ingestor = AsyncTranscriptIngestor(api_key=api_key)
        extractions = await ingestor.process_batch(transcripts)
    else:
        print("[Notice] No GROQ_API_KEY found or empty data. Running with high-accuracy fallback dataset to demonstrate graph architecture...")
        extractions = get_fallback_data()

    engine = GraphRAGEngine()
    engine.build_index(extractions)

    export_interactive_graph(engine.graph, "interactive_knowledge_graph.html")

    print("\n=== Executing Hybrid GraphRAG Queries ===")
    queries = [
        "What tasks are assigned to David and what are their dependencies?",
        "What is blocking the Frontend Team?",
        "Who is responsible for the vector indexing pipeline?"
    ]

    for q in queries:
        print(f"\nQuery: '{q}'")
        result = engine.hybrid_query(q, top_k=2, depth=1)
        print(f" -> Retrieval Latency: {result['retrieval_latency_ms']} ms (Target: <800ms)")
        print(f" -> Action Items Found: {[item['task_id'] + ': ' + item['description'] for item in result['retrieved_action_items']]}")
        print(f" -> Graph Context Enriched: {result['enriched_graph_context'][:3]}")

if __name__ == "__main__":
    asyncio.run(main())
