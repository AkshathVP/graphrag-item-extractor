import time
import networkx as nx
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.models import MeetingExtraction

class GraphRAGEngine:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_texts = []
        self.doc_metadata = []
        self.tfidf_matrix = None

    def build_index(self, extractions: List[MeetingExtraction]):
        print("\nConstructing Knowledge Graph and Vector Embeddings...")
        for ext in extractions:
            for entity in ext.entities:
                self.graph.add_node(entity.name, label=entity.label, type="entity")
            
            for item in ext.action_items:
                self.graph.add_node(
                    item.task_id, 
                    label="ACTION_ITEM", 
                    desc=item.description, 
                    assignee=item.assignee, 
                    deadline=item.deadline
                )
                self.graph.add_edge(item.assignee, item.task_id, relation="ASSIGNED_TO")
                
                for dep in item.dependencies:
                    self.graph.add_edge(item.task_id, dep, relation="DEPENDS_ON")
                
                doc_str = f"{item.task_id}: {item.description} (Assignee: {item.assignee}, Due: {item.deadline})"
                self.doc_texts.append(doc_str)
                self.doc_metadata.append({
                    "task_id": item.task_id, 
                    "assignee": item.assignee, 
                    "description": item.description,
                    "deadline": item.deadline
                })
            
            for rel in ext.relationships:
                self.graph.add_edge(rel.source, rel.target, relation=rel.relation)
        
        if self.doc_texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)
        
        print(f"Graph mapped successfully: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} relationships.")

    def hybrid_query(self, query: str, top_k: int = 3, depth: int = 1) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        if not self.doc_texts or self.tfidf_matrix is None:
            return {"error": "Index is empty.", "latency_ms": 0}

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        retrieved_items = []
        subgraph_nodes = set()

        for idx in top_indices:
            if similarities[idx] > 0.0:
                meta = self.doc_metadata[idx]
                task_id = meta["task_id"]
                retrieved_items.append(meta)
                subgraph_nodes.add(task_id)
                
                if task_id in self.graph:
                    for _ in range(depth):
                        neighbors = list(self.graph.predecessors(task_id)) + list(self.graph.successors(task_id))
                        subgraph_nodes.update(neighbors)
        
        graph_context = []
        for u, v, data in self.graph.edges(nbunch=list(subgraph_nodes), data=True):
            graph_context.append(f"{u} --[{data.get('relation', 'RELATED')}]--> {v}")

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "query": query,
            "retrieved_action_items": retrieved_items,
            "enriched_graph_context": list(set(graph_context)),
            "retrieval_latency_ms": round(latency_ms, 2)
        }
