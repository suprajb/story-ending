from typing import Dict, List, Any, Optional, Union
from .metrics import get_all_metrics, EvaluationMetric
import importlib

class StoryEvaluator:
    """
    Evaluates generated story endings using multiple metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluator with a configuration
        
        Args:
            config: Dictionary containing evaluation configuration
        """
        self.config = config or {}
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict[str, EvaluationMetric]:
 
        # use all metrics by default
        if "metrics" not in self.config:
            return get_all_metrics()
            
        #  dynamically load custom metrics from configuration if implemented
        metrics = {}
        for metric_name, metric_config in self.config.get("metrics", {}).items():
            module_path = metric_config.get("module", "src.evaluation.metrics")
            class_name = metric_config.get("class")
            
           
            if not class_name:
                continue
                
            try:
                module = importlib.import_module(module_path)
                metric_class = getattr(module, class_name)
                metrics[metric_name] = metric_class()
            except (ImportError, AttributeError) as e:
                print(f"Failed to load metric {metric_name}: {e}")
        
        return metrics
    
    def evaluate_ending(self, beginning: str, ending: str, 
                       metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate a single story ending using all or specified metrics
        
        Args:
            beginning: The beginning of the story
            ending: The generated ending to evaluate
            metric_names: Optional list of metric names to use (uses all if None)
            
        Returns:
            Dictionary of metric names to evaluation results
        """
        # filter metrics based on provided names
        metrics_to_use = self.metrics
        if metric_names:
            metrics_to_use = {name: self.metrics[name] for name in metric_names 
                             if name in self.metrics}
        
    
        results = {}
        for name, metric in metrics_to_use.items():
            try:
                results[name] = metric.evaluate(beginning, ending)
            except Exception as e:
                print(f"Error evaluating using metric {name}: {e}")
                results[name] = {
                    "score": 0.0,
                    "error": str(e),
                    "explanation": "Error during evaluation"
                }
        
        # calculate average score
        # TODO: check if average makes sense for all metrics
        scores = [result["score"] for result in results.values() if "score" in result]
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        
        results["average"] = {
            "score": average_score,
            "explanation": "Average of all metrics"
        }
        
        return results

    def evaluate_story_endings(self, story: Dict[str, Any], 
                              metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate all generated endings for a story
        
        Args:
            story: Dictionary containing story data with generated endings
            metric_names: Optional list of specific metrics to use
                
        Returns:
            Dictionary with evaluation results for all endings
        """
        beginning = story.get("beginning", "")
        all_endings = story.get("endings", {})
        
        evaluation_results = {}
        

        for model_name, endings in all_endings.items():
            model_results = []
            for ending in endings:
                result = self.evaluate_ending(beginning, ending, metric_names)
                model_results.append(result)
            
            evaluation_results[model_name] = model_results
        
        return evaluation_results
