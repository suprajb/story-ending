import argparse
import yaml
from pathlib import Path
from src.data_handler import DataHandler
from src.generation_manager import GenerationManager
from src.output_manager import OutputManager
from src.evaluation.evaluator import StoryEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Story Ending Generation Framework")
    parser.add_argument("--config", type=str, default="config/default.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing story data")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save generated endings")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate generated endings")
    parser.add_argument("--evaluation_only", action="store_true",
                        help="Only run evaluation on existing outputs")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of model names to use (e.g., 'gpt-3-5,llama-2'). If not provided, all models in the config will be used.")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available models in the config and exit")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # List available models if requested
    if args.list_models:
        print("Available models in config:")
        for model_name, model_config in config.get("models", {}).items():
            model_type = model_config.get("class", "Unknown")
            model_actual = model_config.get("model_name", "unnamed")
            print(f"  - {model_name}: {model_type} ({model_actual})")
        return
    
    # Initialize components
    data_handler = DataHandler(args.data_dir)
    output_manager = OutputManager(args.output_dir)
    
    # filter models
    # TODO: should we have an arg for this or just use the config?
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
        print(f"Using selected models: {', '.join(model_names)}")
        
        filtered_models = {}
        for model_name in model_names:
            if model_name in config.get("models", {}):
                filtered_models[model_name] = config["models"][model_name]
            else:
                print(f"Warning: Model '{model_name}' not found in config")
        
        # replace config in-memory 
        config["models"] = filtered_models
    
    # if evaluation_only, skip generation
    if not args.evaluation_only:
        generation_manager = GenerationManager(config, output_manager)
        stories = data_handler.load_stories()
        generation_manager.generate_endings(stories)
        print(f"Generation complete. Results saved to {args.output_dir}")
    
    # skip evaluation if not requested
    if not (args.evaluate or args.evaluation_only):
        return
        
    # load data
    stories = []
    if args.evaluation_only:
        stories = output_manager.load_generated_stories()
        if not stories:
            print("No generated stories found for evaluation.")
            return
    else:
        stories = output_manager.get_current_stories()
    
    # evaluate
    evaluator = StoryEvaluator(config.get("evaluation", {}))
    for story in stories:
        evaluation_results = evaluator.evaluate_story_endings(story)
        output_manager.save_evaluation_results(story["id"], evaluation_results)
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
