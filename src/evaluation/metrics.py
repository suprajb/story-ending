from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional

class EvaluationMetric(ABC):
    """Base class for all evaluation metrics"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the metric"""
        pass
    
    @abstractmethod
    def evaluate(self, beginning: str, ending: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a story ending based on this metric
        
        Args:
            beginning: The beginning of the story
            ending: The generated ending
            kwargs: Additional parameters for evaluation
            
        Returns:
            Dictionary containing:
                - score: float between 0.0 and 1.0
                
        """
        pass
    
    @abstractmethod
    def batch_evaluate(self, beginnings: List[str], endings: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Batch evaluation of multiple endings
        
        Args:
            beginnings: List of story beginnings
            endings: List of generated endings
            kwargs: Additional parameters for evaluation
            
        Returns:
            List of dictionaries with evaluation results
        """
        pass

class FaithfulnessMetric(EvaluationMetric):
    """
    Measures how well the ending maintains consistency with the beginning
    """
    
    @property
    def name(self) -> str:
        return "faithfulness"
    
    def evaluate(self, beginning: str, ending: str, **kwargs) -> Dict[str, Any]:
        """
        Placeholder for faithfulness evaluation
       
        """
        # PLACEHOLDER IMPLEMENTATION
        return {
            "score": 0.5,  # Default placeholder score
            
        }
    
    def batch_evaluate(self, beginnings: List[str], endings: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Evaluate faithfulness for multiple stories"""
        # PLACEHOLDER IMPLEMENTATION
        return [self.evaluate(beginning, ending, **kwargs) 
                for beginning, ending in zip(beginnings, endings)]


class CreativityMetric(EvaluationMetric):
    """
    Measures how creative, original, or surprising the ending is.
    """
    
    @property
    def name(self) -> str:
        return "creativity"
    
    def evaluate(self, beginning: str, ending: str, **kwargs) -> Dict[str, Any]:
        """
        Placeholder for creativity evaluation
       
        """
        # PLACEHOLDER IMPLEMENTATION
        return {
            "score": 0.5,  # Default placeholder score
            
        }
    
    def batch_evaluate(self, beginnings: List[str], endings: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Evaluate creativity for multiple stories"""
        # PLACEHOLDER IMPLEMENTATION
        return [self.evaluate(beginning, ending, **kwargs) 
                for beginning, ending in zip(beginnings, endings)]


class CharacterCoverageMetric(EvaluationMetric):
    """
    Evaluates how well the ending addresses and resolves character arcs
    established in the beginning.
    """
    
    @property
    def name(self) -> str:
        return "character_coverage"
    
    def evaluate(self, beginning: str, ending: str, **kwargs) -> Dict[str, Any]:
        """
        Placeholder for character coverage evaluation
      
        """
        # PLACEHOLDER IMPLEMENTATION
        return {
            "score": 0.5,  # Default placeholder score
            
        }
    
    def batch_evaluate(self, beginnings: List[str], endings: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Evaluate character coverage for multiple stories"""
        # PLACEHOLDER IMPLEMENTATION
        return [self.evaluate(beginning, ending, **kwargs) 
                for beginning, ending in zip(beginnings, endings)]


def get_all_metrics() -> Dict[str, EvaluationMetric]:
    """Return all available evaluation metrics"""
    return {
        "faithfulness": FaithfulnessMetric(),
        "creativity": CreativityMetric(),
        "character_coverage": CharacterCoverageMetric()
    }
