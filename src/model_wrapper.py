from abc import ABC, abstractmethod
from typing import List, Dict, Any
import os
import time
from tqdm import tqdm

from openai import OpenAI


from google import genai

class ModelWrapper(ABC):
    """Abstract base class for LLM wrappers"""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        # system prompt for all model wrappers
        self.system_prompt = model_config.get("system_prompt", "You are a creative story writer")
        
        # get test_mode from config
        self.test_mode = model_config.get("test_mode", False)
        if self.test_mode:
            print(f"Model initialized in test mode, will use test endings instead of real generation")
    
    # TEMPORARY: Add method to get random test endings
    # 
    def get_test_endings(self, num_endings: int = 1) -> List[str]:
        """Generate random endings for testing"""
        test_endings = [
            "The journey ended as it began, with a sense of wonder and discovery. Though challenges had been faced, lessons had been learned, and friendships forged that would last a lifetime.",
            
            "With one final twist of fate, everything changed. What seemed like defeat transformed into an unexpected victory, proving that sometimes the most difficult paths lead to the most rewarding destinations.",
            
            "Looking back at where it all began, they couldn't help but smile. The circle was complete, and though nothing had gone according to plan, perhaps that had been the plan all along.",
            
            "And so it was decided that some mysteries are better left unsolved, some doors better left unopened. The greatest wisdom sometimes lies in knowing when to walk away."
        ]
        
        # Return random endings from our test set
        import random
        return [random.choice(test_endings) for _ in range(num_endings)]
        
    @abstractmethod
    def generate(self, prompt: str, num_endings: int = 1, **kwargs) -> List[str]:
        """
        Generate endings for a given story beginning.
        
        Args:
            prompt: The story beginning/prompt
            num_endings: Number of different endings to generate
            kwargs: Additional generation parameters
            
        Returns:
            List of generated endings
        """
        pass
        
    @abstractmethod
    def batch_generate(self, prompts: List[str], num_endings: int = 1, **kwargs) -> List[List[str]]:
        """
        Generate endings for multiple story beginnings.
        
        Args:
            prompts: List of story beginnings/prompts
            num_endings: Number of different endings to generate for each prompt
            kwargs: Additional generation parameters
            
        Returns:
            List of lists, where each inner list contains endings for the corresponding prompt
        """
        pass

class OpenAIWrapper(ModelWrapper):
    """Wrapper for OpenAI models like GPT-4, GPT-3.5-turbo"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model_name = model_config.get("model_name", "gpt-3.5-turbo")
        
        self.api_key = model_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        
        # Only validate API key and initialize client if not in test mode
        if not self.test_mode:
            if not self.api_key:
                raise ValueError("OpenAI API key not provided in config or environment variables")
                
            # Initialize OpenAI client
            self.client = OpenAI(api_key=self.api_key)
        
    def generate(self, prompt: str, num_endings: int = 1, **kwargs) -> List[str]:
        """Generate story endings using OpenAI API or test mode"""
        # TEMPORARY: Use test endings in test mode
        if self.test_mode:
            print(f"TEST MODE: Generating {num_endings} test endings for {self.model_name}")
            return self.get_test_endings(num_endings)
        
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 200)
        top_p = kwargs.get("top_p", 1.0)
        
        print(f"Generating {num_endings} endings with {self.model_name}, temp={temperature}")
        
        endings = []
        for i in range(num_endings):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=1  # generate one completion per request TODO: check if batch is possible
                )
                
                
                ending = response.choices[0].message.content.strip()
                endings.append(ending)
                
                # avoid rate limiting
                # TODO: figure out a better way to handle this
                if i < num_endings - 1:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Error calling OpenAI API: {e}")
                # TODO: consider retrying the request, don't fail silently
                endings.append(f"[Error generating ending {i+1}]")
                
        return endings
        
    def batch_generate(self, prompts: List[str], num_endings: int = 1, **kwargs) -> List[List[str]]:
        """Generate endings for multiple prompts"""
        results = []
        for prompt in tqdm(prompts, desc=f"Generating with {self.model_name}"):
            results.append(self.generate(prompt, num_endings, **kwargs))
        return results

class GeminiWrapper(ModelWrapper):
    """Wrapper for Google's Gemini models like Gemini 2.5 Pro and 1.5 Pro"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model_name = model_config.get("model_name", "gemini-2.0-flash")
        
        # API key from config or environment variable
        self.api_key = model_config.get("api_key") or os.environ.get("GOOGLE_API_KEY")
        
        # Only validate API key and initialize client if not in test mode
        if not self.test_mode:
            if not self.api_key:
                raise ValueError("Google API key not provided in config or environment variables")
                
            
            self.client = genai.Client(api_key=self.api_key)
    
    def generate(self, prompt: str, num_endings: int = 1, **kwargs) -> List[str]:
        """Generate story endings using Google Gemini API or test mode"""
        # TEMPORARY: Use test endings in test mode
        if self.test_mode:
            print(f"TEST MODE: Generating {num_endings} test endings for {self.model_name}")
            return self.get_test_endings(num_endings)
        
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 200)
        top_p = kwargs.get("top_p", 1.0)
        
        print(f"Generating {num_endings} endings with {self.model_name}, temp={temperature}")
        
        endings = []
        for i in range(num_endings):
            try:
                generation_config = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "top_p": top_p,
                }
                
                system_instruction = self.system_prompt
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        **generation_config
                    )
                )
                
                ending = response.text.strip()
                endings.append(ending)
                
                # TODO: handle rate limiting
                if i < num_endings - 1:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                # Add a placeholder if the API call fails
                endings.append(f"[Error generating ending {i+1}]")
                
        return endings
        
    def batch_generate(self, prompts: List[str], num_endings: int = 1, **kwargs) -> List[List[str]]:
        """Generate endings for multiple prompts"""
        results = []
        for prompt in tqdm(prompts, desc=f"Generating with {self.model_name}"):
            results.append(self.generate(prompt, num_endings, **kwargs))
        return results

class VLLMWrapper(ModelWrapper):
    """Wrapper for running models using vLLM for optimized inference"""

    #TODO: test out params on unity, this is a reference implementation
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model_name = model_config.get("model_name", "meta-llama/Llama-2-7b")
        self.tensor_parallel_size = model_config.get("tensor_parallel_size", 1)
        self.max_model_len = model_config.get("max_model_len", 2048)
        self.dtype = model_config.get("dtype", "half") # 'half' or 'float' or 'bfloat16'
        self.gpu_memory_utilization = model_config.get("gpu_memory_utilization", 0.9)
        self.quantization = model_config.get("quantization", None) # None or 'awq' or 'gptq'
        
        # Only initialize the vLLM engine if not in test mode
        if not self.test_mode:
            self._initialize_engine()
        
    def _initialize_engine(self):
        """Initialize the vLLM engine for efficient inference"""
        try:
            from vllm import LLM, SamplingParams
            
            print(f"Initializing vLLM engine for {self.model_name}...")
            print(f"  Tensor parallel size: {self.tensor_parallel_size}")
            print(f"  GPU memory utilization: {self.gpu_memory_utilization}")
            
            
            self.engine = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                gpu_memory_utilization=self.gpu_memory_utilization,
                quantization=self.quantization
            )
            
            # store SamplingParams class for later use
            self.SamplingParams = SamplingParams
            
        except ImportError:
            raise ImportError("vLLM package not installed. Install with 'pip install vllm'")
        
    def generate(self, prompt: str, num_endings: int = 1, **kwargs) -> List[str]:
        """Generate story endings using vLLM or test mode"""
        #TODO: test out generation on unity, this is a reference implementation
        # TEMPORARY: Use test endings in test mode
        if self.test_mode:
            print(f"TEST MODE: Generating {num_endings} test endings for {self.model_name}")
            return self.get_test_endings(num_endings)
        
        from vllm import SamplingParams
        
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 200)
        top_p = kwargs.get("top_p", 0.9)
        
        print(f"Generating {num_endings} endings with vLLM ({self.model_name})")
        
        # format system prompt for LLaMA models
        if "llama" in self.model_name.lower():
           
            full_prompt = f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        else:
            # generic format for other models
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
        
        endings = []
        
        try:
            # create sampling parameters for all endings at once
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=num_endings  # generate multiple outputs in parallel
            )
            
            # generate all endings in one batch for efficiency
            outputs = self.engine.generate(full_prompt, sampling_params)
            
            for output in outputs:
                generated_text = output.outputs[0].text.strip()
                endings.append(generated_text)
                
        except Exception as e:
            print(f"Error generating with vLLM: {e}")
            # add placeholders for any missing endings
            # TODO: don't fail silently
            while len(endings) < num_endings:
                endings.append(f"[Error generating ending {len(endings)+1}]")
        
        return endings
        
    def batch_generate(self, prompts: List[str], num_endings: int = 1, **kwargs) -> List[List[str]]:
        #TODO: implement batch generation for vLLM
        pass

