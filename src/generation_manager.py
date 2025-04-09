from typing import Dict, List, Any
import importlib
from .model_wrapper import ModelWrapper
from .output_manager import OutputManager

class GenerationManager:
    """Manages the generation of story endings across multiple LLMs"""
    
    def __init__(self, config: Dict[str, Any], output_manager: OutputManager):
        self.config = config
        self.output_manager = output_manager
        self.models = self._initialize_models()
        
    def _initialize_models(self) -> Dict[str, ModelWrapper]:
        """Initialize model wrappers based on configuration"""
        models = {}
        
        # Check if test_mode is set globally
        global_test_mode = self.config.get("test_mode", False)
        
        for model_name, model_config in self.config.get("models", {}).items():
            # Apply global test_mode setting if not specified in the model config
            if global_test_mode and "test_mode" not in model_config:
                model_config["test_mode"] = global_test_mode
            
            # load the appropriate wrapper class
            module_path = model_config.get("module", "src.model_wrapper")
            class_name = model_config.get("class", "OpenAIWrapper")
            
            try:
                module = importlib.import_module(module_path)
                wrapper_class = getattr(module, class_name)
                models[model_name] = wrapper_class(model_config)
                print(f"Initialized model: {model_name} using {class_name}")
            except (ImportError, AttributeError) as e:
                print(f"Failed to initialize model {model_name}: {e}")
                
        return models
    
    def _create_prompt(self, story: Dict[str, Any]) -> str:
        """Create a prompt for story ending generation"""
        template = self.config.get("prompt_template", 
                                   "Complete this story:\n\n{beginning}\n\nEnding:")
        return template.format(**story)
    
    def generate_endings(self, stories: List[Dict[str, Any]]) -> None:
        """Generate endings for all stories using all configured models"""
        num_endings = self.config.get("num_endings", 3)
        
        for story in stories:
            story_id = story.get("id")
            prompt = self._create_prompt(story)
            
            story_results = {
                "id": story_id,
                "beginning": story.get("beginning"),
                "endings": {}
            }
            
            for model_name, model in self.models.items():
                try:
                    endings = model.generate(
                        prompt, 
                        num_endings=num_endings,
                        **self.config.get("generation_params", {})
                    )
                    
                    story_results["endings"][model_name] = endings
                    print(f"Generated {len(endings)} endings for story {story_id} using {model_name}")
                except Exception as e:
                    print(f"Error generating endings for story {story_id} with {model_name}: {e}")
            
            # save results for this story
            self.output_manager.save_generated_endings(story_results)
