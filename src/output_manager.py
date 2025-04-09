import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import datetime
import glob

class OutputManager:
    """Manages saving and organizing generated story endings"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        
        # timestamp for run directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir()
        
        # create summary file
        # TODO: reorganise summary for better readability
        self.summary_file = self.run_dir / "summary.json"
        self.summary = {
            "timestamp": timestamp,
            "stories": []
        }
        
        # keep track of stories for later evaluation
        self.current_stories = []
    
    def save_generated_endings(self, story_results: Dict[str, Any]) -> None:
        """
        Save generated endings for a story
        
        Args:
            story_results: Dictionary containing story data and generated endings
        """
        story_id = story_results.get("id")
        if not story_id:
            raise ValueError("Story results must include an 'id' field")
        
        #  later evaluation
        self.current_stories.append(story_results)
        
        
        story_file = self.run_dir / f"{story_id}.json"
        with open(story_file, 'w') as f:
            json.dump(story_results, f, indent=2)
        
        # update summary
        self.summary["stories"].append({
            "id": story_id,
            "models": list(story_results.get("endings", {}).keys()),
            "file": story_file.name
        })
        
       
        with open(self.summary_file, 'w') as f:
            json.dump(self.summary, f, indent=2)
        
        print(f"Saved results for story {story_id} to {story_file}")
    
    def save_evaluation_results(self, story_id: str, evaluation_results: Dict[str, Any]) -> None:
        """
        Save evaluation results for a story
        
        Args:
            story_id: Identifier of the story
            evaluation_results: Dictionary of evaluation results
        """
        
        eval_dir = self.run_dir / "evaluations"
        eval_dir.mkdir(exist_ok=True)
        
        
        eval_file = eval_dir / f"{story_id}_eval.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # update summary
        eval_summary_file = eval_dir / "eval_summary.json"
        
        # if summary file exists, load it
        if eval_summary_file.exists():
            with open(eval_summary_file, 'r') as f:
                eval_summary = json.load(f)
        else:
            eval_summary = {
                "stories": {},
                "metrics_by_model": {}
            }
        
        # add this story to the stories section
        eval_summary["stories"][story_id] = {
            "file": eval_file.name
        }
        
        # reorganise metrics by model
        metrics_by_model = eval_summary.get("metrics_by_model", {})
        
        
        for model_name, endings_evals in evaluation_results.items():
            if not endings_evals:
                continue
            
            # each model gets its own entry
            if model_name not in metrics_by_model:
                metrics_by_model[model_name] = {
                    "stories_count": 0,
                    "metrics": {}
                }
            
            model_entry = metrics_by_model[model_name]
            model_entry["stories_count"] += 1
            
            #  metrics across all endings per story for this model 
            story_metrics = {}
            
            for ending_eval in endings_evals:
                for metric_name, metric_data in ending_eval.items():
                    if metric_name == "average":
                        continue
                    
                    
                    if metric_name not in story_metrics:
                        story_metrics[metric_name] = []
                    
                   
                    if "score" in metric_data:
                        story_metrics[metric_name].append(metric_data["score"])
            
            # TODO: calculate more statistics if needed, check if average makes sense for all metrics
            for metric_name, scores in story_metrics.items():
                if not scores:
                    continue
                    
                # avg
                story_avg = sum(scores) / len(scores)
                
                if metric_name not in model_entry["metrics"]:
                    model_entry["metrics"][metric_name] = {
                        "total_score": 0.0,
                        "story_count": 0,
                        "average": 0.0
                    }
                
               
                metric_entry = model_entry["metrics"][metric_name]
                metric_entry["total_score"] += story_avg
                metric_entry["story_count"] += 1
                metric_entry["average"] = metric_entry["total_score"] / metric_entry["story_count"]
        
       
        eval_summary["metrics_by_model"] = metrics_by_model
        
        # save to separate file
        with open(eval_summary_file, 'w') as f:
            json.dump(eval_summary, f, indent=2)
            
        print(f"Saved evaluation results for story {story_id} to {eval_file}")
    
    def get_current_stories(self) -> List[Dict[str, Any]]:
        """Get stories generated in the current run"""
        return self.current_stories
    
    def load_generated_stories(self, run_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load previously generated stories from a run directory
        
        Args:
            run_dir: Optional path to run directory, uses latest if not specified
            
        Returns:
            List of story dictionaries with generated endings
        """
        
        if run_dir:
            target_dir = Path(run_dir)
        else:
            # we use latest run
            run_dirs = sorted(self.output_dir.glob("run_*"))
            if not run_dirs:
                return []
            target_dir = run_dirs[-1]
        
        
        summary_file = target_dir / "summary.json"
        if not summary_file.exists():
            return []
            
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        
        stories = []
        for story_info in summary.get("stories", []):
            story_file = target_dir / story_info.get("file", "")
            if story_file.exists():
                with open(story_file, 'r') as f:
                    story = json.load(f)
                    stories.append(story)
        
        return stories
