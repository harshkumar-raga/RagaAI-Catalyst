from typing import List, Dict, Any

class SpanAttributes:
    def __init__(self, name):
        self.name = name
        self.tags = []
        self.metadata = {}
        self.metrics = []
        self.feedback = None
        self.trace_attributes = ['tags', 'metadata', 'metrics']

    def add_tags(self, tags: str|List[str]):
        if isinstance(tags, str):
            tags = [tags]
        self.tags.extend(tags)

    def add_metadata(self, metadata):
        self.metadata.update(metadata)

    def add_metrics(
        self, 
        name: str, 
        score: float|int, 
        reasoning: str='', 
        cost: float=None, 
        latency: float=None, 
        metadata: Dict[str, Any]={}, 
        config: Dict[str, Any]={}
        ):
        self.metrics.append({
            "name": name,
            "score": score,
            "reason": reasoning, 
            "source": 'user', 
            "cost": cost,
            "latency": latency, 
            "metadata": metadata,
            "mappings": [],
            "config": config
        })
    
    def add_feedback(self, feedback: Any):
        self.feedback = feedback