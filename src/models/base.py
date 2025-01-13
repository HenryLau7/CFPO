from abc import ABC, abstractmethod

class LLM_Model(ABC):
    def __init__(self, model_path=None, max_tokens=512, stop = '', repetition_penalty=1.0):
        self.model_path = model_path
        self.max_token = max_tokens
        self.stop = stop
        self.repetition_penalty = repetition_penalty

    @abstractmethod
    def inference(self, ex, prompt):
        pass
