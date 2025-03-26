from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAgent(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.memory: Dict[str, Any] = {}

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def update_memory(self, key: str, value: Any) -> None:
        self.memory[key] = value

    def get_memory(self, key: str) -> Optional[Any]:
        return self.memory.get(key)
