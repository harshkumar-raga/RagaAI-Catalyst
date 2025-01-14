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

    def add_metrics(self, metric_name: str, metric_value: float|int, metric_reasoning: str):
        self.metrics.append({
            "name": metric_name,
            "value": metric_value,
            "reasoning": metric_reasoning
        })
    
    def add_feedback(self, feedback: Any):
        self.feedback = feedback