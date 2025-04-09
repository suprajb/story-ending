# Story Ending Generation

A Python framework for generating and evaluating story endings using state-of-the-art language models.

## Installation

```shell
git clone <repository-url>
cd seg
pip install -r requirements.txt
```

## API Keys Setup

```shell
# For OpenAI models
export OPENAI_API_KEY="your-openai-key"

# For Google Gemini models
export GOOGLE_API_KEY="your-google-key"
```

## Usage

Set `test_mode:True` in your config (default config is `config/default.yaml`) for debugging and testing, this returns a random ending from a list regardless of the story

Set `test_mode:False` after setting up your API keys for actual generation API calls

```bash
# Generate endings using specified models
python main.py --config config/default.yaml --models gpt-3-5,gemini-2-0-flash

# Generate and evaluate
python main.py --config config/default.yaml --evaluate
```

## Sample Config
```yaml
# Basic configuration
test_mode: false
num_endings: 3
prompt_template: "Complete this story with a compelling ending:\n\n{beginning}\n\nEnding:"
system_prompt: "You are a creative story writer"

# Models configuration
models:
  gpt-3-5:
    module: "src.model_wrapper"
    class: "OpenAIWrapper"
    model_name: "gpt-3.5-turbo"
  
  gemini-2-0-flash:
    module: "src.model_wrapper"
    class: "GeminiWrapper"
    model_name: "gemini-2.0-flash"

# Generation parameters
generation_params:
  temperature: 0.7
  max_tokens: 1000
  top_p: 0.9

# Evaluation configuration
evaluation:
  metrics:
    faithfulness:
      module: "src.evaluation.metrics"
      class: "FaithfulnessMetric"
```
## Command Line

```
--config CONFIG       Path to configuration file (default: config/default.yaml)
--data_dir DATA_DIR   Directory containing story data (default: data)
--output_dir OUTPUT_DIR
                      Directory to save generated endings (default: outputs)
--evaluate            Evaluate generated endings
--evaluation_only     Only run evaluation on existing outputs
--models MODELS       Comma-separated list of model names to use (e.g., 'gpt-3-5,llama-2')
--list-models         List all available models in the config and exit
```

## Output Structure
Generated endings and evaluation results are saved in the outputs directory with the following structure:

```shell
outputs/
└── run_20250408_143012/             # Timestamped run directory
    ├── summary.json                 # Run summary
    ├── story_001.json               # Generated endings
    └── evaluations/                 # Created when using --evaluate
        ├── eval_summary.json        # Evaluation summary
        └── story_001_eval.json      # Story-specific evaluation
```            

## Adding Custom Models

1. Create a new wrapper class that inherits from `ModelWrapper`:

```python
# src/custom_wrapper.py
from src.model_wrapper import ModelWrapper

class MyCustomWrapper(ModelWrapper):
    def __init__(self, model_config):
        super().__init__(model_config)
        # Custom initialization
        
    def generate(self, prompt, num_endings=1, **kwargs):
        # Custom generation logic
        return ["Generated ending 1", "Generated ending 2"]
```
2. Update your config to use the custom wrapper:
```yaml
models:
  my-custom-model:
    module: "src.custom_wrapper"
    class: "MyCustomWrapper"
    model_name: "custom-model-name"
    # Additional configuration
```

## Adding Custom Metrics
1. Create a new metric class that inherits from `EvaluationMetric`:

```python
# src/evaluation/custom_metric.py
from src.evaluation.metric import EvaluationMetric

class MyCustomMetric(EvaluationMetric):
    def evaluate(self, story, ending):
        # Custom evaluation logic
        return 0.75  # Score between 0.0 and 1.0
```
2. Update your config to use this metric

```yaml
evaluation:
  metrics:
    my_custom_metric:
      module: "src.evaluation.custom_metric"
      class: "MyCustomMetric"
```

### Data Organization

