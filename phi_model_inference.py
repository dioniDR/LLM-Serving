#!/usr/bin/env python3
"""
Phi-1.5 ONNX Model Inference Script

This script provides a clean implementation for text generation using the Phi-1.5 ONNX model.
It handles the model's specific requirements for input/output processing.
"""

import argparse
import logging
import os
import numpy as np
import onnxruntime
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class PhiOnnxModel:
    """Class to handle inference with the Phi-1.5 ONNX model."""
    
    def __init__(self, model_path: str):
        """
        Initialize the Phi-1.5 ONNX model.
        
        Args:
            model_path (str): Path to the directory containing the ONNX model files
        """
        self.model_path = model_path
        self.onnx_model_path = os.path.join(model_path, "model.onnx")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set up ONNX session
        logger.info(f"Loading ONNX model from {self.onnx_model_path}")
        self.session = onnxruntime.InferenceSession(
            self.onnx_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        
        # Get model metadata
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        logger.info(f"Model inputs: {self.input_names}")
        logger.info(f"Model outputs: {self.output_names}")
        
    def generate(self, 
                prompt: str, 
                max_length: int = 100, 
                temperature: float = 0.7, 
                top_p: float = 0.9,
                top_k: int = 50) -> str:
        """
        Generate text based on the input prompt.
        
        Args:
            prompt (str): The input prompt to generate text from
            max_length (int, optional): Maximum length of generated text. Defaults to 100.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            top_p (float, optional): Top-p sampling parameter. Defaults to 0.9.
            top_k (int, optional): Top-k sampling parameter. Defaults to 50.
            
        Returns:
            str: The generated text including the prompt
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Initialize past key values as None for the first iteration
        past_key_values = None
        position_ids = None
        
        # Start with the tokenized prompt as the generated sequence
        generated_sequence = input_ids[0].tolist()
        
        # Keep track of tokens for logging
        tokens_generated = 0
        
        # Generate tokens until we reach max_length or an EOS token
        for _ in range(max_length):
            # Prepare inputs for the model - handling position_ids and past_key_values
            model_inputs = self._prepare_inputs(
                input_ids, attention_mask, position_ids, past_key_values
            )
            
            # Run inference
            outputs = self.session.run(self.output_names, model_inputs)
            
            # Extract logits and past_key_values from outputs
            logits = outputs[0]
            
            # Update past key values for next iteration if present in outputs
            if len(outputs) > 1:
                past_key_values = {}
                for i, name in enumerate(self.output_names[1:], 1):
                    if name.startswith("present"):
                        past_key_values[name] = outputs[i]
            
            # Get next token through sampling
            next_token_id = self._sample_token(
                logits[:, -1, :], temperature, top_p, top_k
            )
            
            # Append the next token to the generated sequence
            generated_sequence.append(next_token_id)
            tokens_generated += 1
            
            # Update input_ids and attention_mask for next iteration
            input_ids = np.array([[next_token_id]], dtype=np.int64)
            attention_mask = np.array([[1]], dtype=np.int64)
            
            # Update position_ids for next iteration
            if position_ids is None:
                position_ids = np.array([[len(generated_sequence) - 1]], dtype=np.int64)
            else:
                position_ids = np.array([[position_ids[0, 0] + 1]], dtype=np.int64)
            
            # Log progress
            if tokens_generated % 10 == 0:
                logger.info(f"Generated {tokens_generated} tokens")
            
            # Stop if we've generated an EOS token
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
        # Decode the generated sequence
        generated_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
        return generated_text
    
    def _prepare_inputs(self, 
                       input_ids: np.ndarray,
                       attention_mask: np.ndarray,
                       position_ids: Optional[np.ndarray] = None,
                       past_key_values: Optional[Dict] = None) -> Dict:
        """
        Prepare inputs for the model based on what the ONNX model expects.
        
        Args:
            input_ids (np.ndarray): Input token IDs
            attention_mask (np.ndarray): Attention mask
            position_ids (np.ndarray, optional): Position IDs. Defaults to None.
            past_key_values (Dict, optional): Past key values. Defaults to None.
            
        Returns:
            Dict: Dictionary of inputs for the model
        """
        # Basic inputs that are always required
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        # Add position_ids if needed
        if "position_ids" in self.input_names:
            if position_ids is None:
                seq_length = input_ids.shape[1]
                position_ids = np.arange(seq_length, dtype=np.int64).reshape(1, -1)
            inputs["position_ids"] = position_ids
        
        # Add past key values if needed
        if past_key_values:
            for name, value in past_key_values.items():
                # Map 'present' output names to 'past' input names if needed
                input_name = name.replace("present", "past") if "present" in name else name
                if input_name in self.input_names:
                    inputs[input_name] = value
        
        return inputs
    
    def _sample_token(self, 
                     logits: np.ndarray, 
                     temperature: float = 1.0, 
                     top_p: float = 1.0,
                     top_k: int = 0) -> int:
        """
        Sample the next token from the logits.
        
        Args:
            logits (np.ndarray): Logits from the model output
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            top_p (float, optional): Top-p sampling parameter. Defaults to 1.0.
            top_k (int, optional): Top-k sampling parameter. Defaults to 0.
            
        Returns:
            int: The sampled token ID
        """
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < np.sort(logits)[-top_k]
            logits[indices_to_remove] = -float('Inf')
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits = np.sort(logits)[::-1]
            sorted_indices = np.argsort(logits)[::-1]
            cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits)))
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
            sorted_indices_to_remove[0] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')
        
        # Convert logits to probabilities
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        # Sample from the distribution
        next_token_id = np.random.choice(logits.shape[0], p=probs)
        return int(next_token_id)

def main():
    """Main function to parse arguments and run text generation."""
    parser = argparse.ArgumentParser(description="Generate text using Phi-1.5 ONNX model")
    parser.add_argument("--model_path", type=str, default="models/phi-1.5-onnx",
                        help="Path to the directory containing the ONNX model files")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Input prompt for text generation")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    
    args = parser.parse_args()
    
    try:
        # Initialize the model
        model = PhiOnnxModel(args.model_path)
        
        # Generate text
        logger.info(f"Generating text with prompt: {args.prompt}")
        logger.info(f"Parameters: max_length={args.max_length}, temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
        
        generated_text = model.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )
        
        # Print the generated text
        print("\n" + "="*50)
        print("GENERATED TEXT:")
        print("="*50)
        print(generated_text)
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error during text generation: {e}", exc_info=True)
        
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phi-1.5 ONNX Model Inference Script

This script provides functionality to run text generation inference using the Phi-1.5 model
in ONNX format. It optimizes the model for inference and handles the generation process
including proper management of attention caching for efficient generation.

Usage:
    python phi_model_inference.py --prompt "Your prompt here" --max_length 100 --temperature 0.7
"""

import os
import time
import argparse
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any, Optional, Tuple, Union
from transformers import AutoTokenizer, PreTrainedTokenizer
import json


class PhiOnnxModel:
    """
    Wrapper class for Phi-1.5 ONNX model inference with optimized runtime.
    """
    
    def __init__(self, model_path: str, use_cuda: bool = False):
        """
        Initialize the Phi ONNX model.
        
        Args:
            model_path: Path to the directory containing the ONNX model files
            use_cuda: Whether to use CUDA for inference acceleration
        """
        self.model_path = model_path
        self.onnx_model_path = os.path.join(model_path, "model.onnx")
        
        # Load model configuration
        with open(os.path.join(model_path, "config.json"), "r") as f:
            self.config = json.load(f)
        
        # Extract model dimensions from config
        self.hidden_size = self.config["hidden_size"]
        self.num_attention_heads = self.config["num_attention_heads"]
        self.num_hidden_layers = self.config["num_hidden_layers"]
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = self.config["max_position_embeddings"]
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Configure ONNX Runtime session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Set up provider options based on CUDA availability
        if use_cuda and ort.get_device() == "GPU":
            providers = [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
            print("Using CUDA for inference")
        else:
            providers = ["CPUExecutionProvider"]
            print("Using CPU for inference")
        
        # Create ONNX Runtime session
        print(f"Loading ONNX model from {self.onnx_model_path}")
        self.session = ort.InferenceSession(
            self.onnx_model_path, 
            sess_options=sess_options, 
            providers=providers
        )
        
        # Get input and output names
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"Model inputs: {self.input_names}")
        print(f"Model outputs: {self.output_names}")
    
    def _get_empty_past_key_values(self, batch_size: int, sequence_length: int) -> Dict[str, np.ndarray]:
        """
        Create empty past key values tensors for the first inference step.
        
        Args:
            batch_size: Batch size for inference
            sequence_length: Sequence length of the input
            
        Returns:
            Dictionary of empty past key value tensors
        """
        past_key_values = {}
        
        # The pattern for past key value names varies by model version
        # These are common patterns for Phi models
        for i in range(self.num_hidden_layers):
            # Keys for past attention
            past_key_values[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, self.num_attention_heads, 0, self.head_dim), 
                dtype=np.float32
            )
            
            # Values for past attention
            past_key_values[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, self.num_attention_heads, 0, self.head_dim), 
                dtype=np.float32
            )
        
        return past_key_values
    
    def _prepare_inputs_for_generation(
        self, 
        input_ids: np.ndarray, 
        attention_mask: np.ndarray,
        past_key_values: Optional[Dict[str, np.ndarray]] = None,
        position_ids: Optional[np.ndarray] = None,
        is_first_step: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Prepare inputs for the ONNX model.
        
        Args:
            input_ids: Token IDs for input text
            attention_mask: Attention mask for input tokens
            past_key_values: Past key values from previous inference steps
            position_ids: Position IDs for input tokens
            is_first_step: Whether this is the first step of generation
            
        Returns:
            Dictionary of inputs for the model
        """
        batch_size, seq_length = input_ids.shape
        
        # If position_ids is not provided, create it
        if position_ids is None:
            if is_first_step:
                # For the first step, create position IDs from 0 to seq_length-1
                position_ids = np.arange(seq_length, dtype=np.int64).reshape(1, -1)
            else:
                # For subsequent steps, use the position after the last one
                position_ids = np.array([[attention_mask.sum() - 1]], dtype=np.int64)
        
        # Initialize inputs dictionary with required inputs
        model_inputs = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
            "position_ids": position_ids.astype(np.int64),
        }
        
        # Add past key values if provided
        if past_key_values is not None:
            model_inputs.update(past_key_values)
        else:
            # If past_key_values is not provided, create empty ones
            model_inputs.update(self._get_empty_past_key_values(batch_size, seq_length))
        
        return model_inputs
    
    def _update_attention_mask_and_position_ids(
        self, 
        attention_mask: np.ndarray, 
        position_ids: np.ndarray, 
        next_token_id: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update attention mask and position IDs for the next token.
        
        Args:
            attention_mask: Current attention mask
            position_ids: Current position IDs
            next_token_id: Token ID for the next predicted token
            
        Returns:
            Tuple of (new input IDs, new attention mask, new position IDs)
        """
        # Set input_ids to just the newly generated token
        new_input_ids = next_token_id.reshape(1, 1)
        
        # Extend attention mask for the new token
        new_attention_mask = np.concatenate(
            [attention_mask, np.ones((1, 1), dtype=np.int64)], 
            axis=1
        )
        
        # Update position IDs to point to the position after the last attended position
        new_position_ids = np.array([[position_ids[0, -1] + 1]], dtype=np.int64)
        
        return new_input_ids, new_attention_mask, new_position_ids
    
    def _extract_past_key_values(self, outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract past key values from model outputs.
        
        Args:
            outputs: Model outputs from inference
            
        Returns:
            Dictionary of past key values for next inference step
        """
        past_key_values = {}
        
        # Extract all present key/values from outputs to use as past key/values in next step
        for i in range(self.num_hidden_layers):
            if f"present.{i}.key" in outputs:
                past_key_values[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
                past_key_values[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]
            else:
                # Handle models with different output naming patterns
                for key in outputs:
                    if f"present_{i}_key" in key or f"present_{i}.key" in key:
                        past_key_values[f"past_key_values.{i}.key"] = outputs[key]
                    if f"present_{i}_value" in key or f"present_{i}.value" in key:
                        past_key_values[f"past_key_values.{i}.value"] = outputs[key]
        
        return past_key_values
    
    def generate(
        self, 
        prompt: str, 
        max_length: int = 100, 
        temperature: float = 0.7, 
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text based on the input prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to sample or use greedy decoding
            num_return_sequences: Number of sequences to return
            
        Returns:
            List of generated text sequences
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="np", padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Initialize generation parameters
        batch_size = input_ids.shape[0]
        generated_sequences = []
        
        # Generate sequences one by one
        for _ in range(num_return_sequences):
            # Make a copy of the input tensors to avoid modifying the originals
            curr_input_ids = input_ids.copy()
            curr_attention_mask = attention_mask.copy()
            past_key_values = None
            
            # Generate tokens up to max_length
            for i in range(max_length):
                # Print progress every 10 tokens
                if i % 10 == 0:
                    print(f"Generating token {i}/{max_length}...")
                
                # Prepare inputs for the model
                is_first_step = (i == 0)
                if is_first_step:
                    # For the first step, use the full input sequence
                    position_ids = np.arange(curr_input_ids.shape[1], dtype=np.int64).reshape(1, -1)
                    model_inputs = self._prepare_inputs_for_generation(
                        curr_input_ids, 
                        curr_attention_mask,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        is_first_step=True
                    )
                else:
                    # For subsequent steps, only use the last generated token
                    new_position_id = np.array([[curr_attention_mask.sum() - 1]], dtype=np.int64)
                    model_inputs = self._prepare_inputs_for_generation(
                        curr_input_ids[:, -1:], 
                        curr_attention_mask,
                        past_key_values=past_key_values,
                        position_ids=new_position_id,
                        is_first_step=False
                    )
                
                # Run inference
                outputs = self.session.run(None, model_inputs)
                output_dict = {name: outputs[i] for i, name in enumerate(self.output_names)}
                
                # Extract logits and past key values
                logits = output_dict["logits"]
                past_key_values = self._extract_past_key_values(output_dict)
                
                # Get the logits for the last token
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty > 1.0:
                    for batch_idx in range(batch_size):
                        for token_idx in set(curr_input_ids[batch_idx].tolist()):
                            if next_token_logits[batch_idx, token_idx] < 0:
                                next_token_logits[batch_idx, token_idx] *= repetition_penalty
                            else:
                                next_token_logits[batch_idx, token_idx] /= repetition_penalty
                
                # Apply token selection strategy (top_k, top_p, or greedy)
                if do_sample:
                    # Top-k sampling
                    if top_k > 0:
                        indices_to_remove = np.argpartition(next_token_logits, -top_k, axis=-1)[:, :-top_k]
                        for batch_idx in range(batch_size):
                            next_token_logits[batch_idx, indices_to_remove[batch_idx]] = -float('inf')
                    
                    # Convert logits to probabilities
                    probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits), axis=-1, keepdims=True)
                    
                    # Top-p (nucleus) sampling
                    if top_p < 1.0:
                        sorted_indices = np.argsort(probs, axis=-1)[:, ::-1]
                        sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)
                        cumulative_probs = np.cumsum(sorted_probs, axis=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep the first token above threshold
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].copy()
                        sorted_indices_to_remove[:, 0] = False
                        
                        # Scatter sorted indices to original indices
                        indices_to_remove = np.zeros_like(probs, dtype=bool)
                        for batch_idx in range(batch_size):

#!/usr/bin/env python3
"""
Phi-1.5 ONNX Model Inference Script (Optimized Version)

This script loads the Microsoft Phi-1.5 model in ONNX format and uses 
onnxruntime.transformers for optimized inference. The model should be located 
in the 'models/phi-1.5-onnx' directory.

Requirements:
- onnxruntime
- onnxruntime-transformers
- transformers
- numpy

Usage:
    python phi_model_inference.py --prompt "Your prompt here" --max_length 150
"""

import os
import sys
import argparse
import logging
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer, AutoConfig
    from onnxruntime.transformers.optimizer import optimize_model
    from onnxruntime.transformers.fusion_options import FusionOptions
except ImportError as e:
    logger.error(f"Required package not found: {e}")
    logger.error("Please install the required packages: pip install onnxruntime onnxruntime-transformers transformers")
    sys.exit(1)


class PhiOnnxModel:
    """
    A wrapper class for the Phi-1.5 ONNX model to handle inference.
    """
    
    def __init__(self, model_path: str = "models/phi-1.5-onnx"):
        """
        Initialize the Phi-1.5 ONNX model.
        
        Args:
            model_path: Path to the directory containing the ONNX model and tokenizer files
        """
        self.model_path = model_path
        self.onnx_model_path = os.path.join(model_path, "model.onnx")
        self.tokenizer_path = model_path
        
        # Check if model files exist
        if not os.path.exists(self.onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found at {self.onnx_model_path}")
        
        if not os.path.exists(os.path.join(model_path, "tokenizer.json")):
            raise FileNotFoundError(f"Tokenizer files not found in {model_path}")
        
        # Load tokenizer and model
        try:
            logger.info(f"Loading tokenizer from {self.tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            
            logger.info(f"Loading ONNX model from {self.onnx_model_path}")
            # Create ONNX inference session with CPU provider
            self.session = ort.InferenceSession(
                self.onnx_model_path, 
                providers=["CPUExecutionProvider"]
            )
            
            # Get model inputs and outputs
            self.model_inputs = [input.name for input in self.session.get_inputs()]
            self.model_outputs = [output.name for output in self.session.get_outputs()]
            
            # Log detailed information about inputs and outputs
            logger.info(f"Model loaded successfully. Input names: {self.model_inputs}")
            logger.info(f"Model output names: {self.model_outputs}")
            
            # Get more details about inputs
            logger.info("Input details:")
            for input in self.session.get_inputs():
                if hasattr(input, 'shape'):
                    logger.info(f"  {input.name}: shape={input.shape}, type={input.type}")
                else:
                    logger.info(f"  {input.name}: type={input.type}")
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {str(e)}")
            raise
    
    def generate(self, 
                prompt: str, 
                max_length: int = 100, 
                temperature: float = 0.7,
                top_p: float = 0.9,
                do_sample: bool = True) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: Input text to generate from
            max_length: Maximum length of the generated text (including prompt)
            temperature: Controls randomness in sampling (lower is more deterministic)
            top_p: Controls diversity via nucleus sampling
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Generated text including the prompt
        """
        try:
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="np")
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            
            # Initialize for autoregressive generation
            generated_ids = input_ids.copy()
            current_length = input_ids.shape[1]
            past_key_values = None
            
            logger.info(f"Starting generation with prompt: {prompt}")
            
            # Get the model's expected past_key_value input names
            past_kv_input_names = [name for name in self.model_inputs 
                                  if "past" in name.lower() and "key_values" in name.lower()]
            
            # Get the model's output names for past key values (present outputs)
            past_kv_output_names = [name for name in self.model_outputs 
                                   if "present" in name.lower() or ("past" in name.lower() and "key_values" in name.lower())]
            
            logger.info(f"Past KV input names: {past_kv_input_names}")
            logger.info(f"Past KV output names: {past_kv_output_names}")
            
            # Generate tokens one by one
            for _ in range(max_length - input_ids.shape[1]):
                # Prepare inputs for the model
                if past_key_values is None:
                    # First forward pass - use the entire input sequence
                    position_ids = np.arange(current_length, dtype=np.int64).reshape(1, -1)
                    model_inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids
                    }
                    
                    # Initialize past key values with zeros if model expects them
                    if past_kv_input_names:
                        batch_size = input_ids.shape[0]
                        seq_len = input_ids.shape[1]
                        # Initialize empty past key values based on model expectations
                        # This is typically required for the first run in some ONNX models
                        for name in past_kv_input_names:
                            # Try to get the expected shape from the input
                            input_info = next((inp for inp in self.session.get_inputs() if inp.name == name), None)
                            if input_info and hasattr(input_info, 'shape'):
                                shape = input_info.shape
                                # Replace dynamic dimensions with appropriate values
                                shape = [batch_size if dim == -1 or dim == 'batch_size' else 
                                        seq_len if dim == -1 or dim == 'sequence_length' else 
                                        int(dim) if isinstance(dim, str) and dim.isdigit() else dim
                                        for dim in shape]
                                model_inputs[name] = np.zeros(shape, dtype=np.float32)
                else:
                    # Subsequent passes - use the last token and past key values
                    last_token = generated_ids[:, -1:]
                    # For subsequent tokens, position_id is the length of the sequence
                    position_ids = np.array([[current_length-1]], dtype=np.int64)
                    # Update attention mask to include the new token
                    new_attention_mask = np.ones((1, current_length), dtype=np.int64)
                    
                    model_inputs = {
                        "input_ids": last_token,
                        "attention_mask": new_attention_mask,
                        "position_ids": position_ids
                    }
                    
                    # Add past key values to inputs - maintain exact mapping from outputs to inputs
                    if past_key_values and past_kv_input_names and past_kv_output_names:
                        # Map past key values from outputs to corresponding inputs
                        for i, (output_name, kv_tensor) in enumerate(past_key_values):
                            # Find the corresponding input name
                            for input_name in past_kv_input_names:
                                # Try to match output index to input index
                                if f"{i}" in input_name or (i < len(past_kv_input_names) and input_name == past_kv_input_names[i]):
                                    model_inputs[input_name] = kv_tensor
                                    break
                # Filter inputs based on what the model expects
                filtered_inputs = {k: v for k, v in model_inputs.items() if k in self.model_inputs}
                
                # Print filtered inputs for debugging the first time and when past_key_values are added
                if _ == 0 or _ == 1:
                    logger.info(f"Iteration {_} - Filtered inputs keys: {filtered_inputs.keys()}")
                    for k, v in filtered_inputs.items():
                        if isinstance(v, np.ndarray):
                            logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                        else:
                            logger.info(f"  {k}: type={type(v)}")
                
                # Run inference
                # Run inference
                outputs = self.session.run(None, filtered_inputs)
                
                # Get logits and past key values
                # First output is typically the logits
                if "logits" in self.model_outputs:
                    logits_idx = self.model_outputs.index("logits")
                    logits = outputs[logits_idx][:, -1, :]
                else:
                    # Default to first output if names don't match expectations
                    logits = outputs[0][:, -1, :]
                
                # Extract past key values from outputs
                past_key_values = []
                
                # Match outputs to their names
                for i, name in enumerate(self.model_outputs):
                    if i < len(outputs) and ("present" in name.lower() or 
                                           ("past" in name.lower() and "key_values" in name.lower())):
                        past_key_values.append((name, outputs[i]))
                
                # If no identified past key values, try using non-logits outputs with their indices
                if not past_key_values and len(outputs) > 1:
                    for i in range(1, len(outputs)):
                        if i < len(self.model_outputs):
                            past_key_values.append((self.model_outputs[i], outputs[i]))
                        else:
                            past_key_values.append((f"output_{i}", outputs[i]))
                # Increment the current length for position_ids
                current_length += 1
                
                # Apply temperature
                if temperature > 0:
                    logits = logits / temperature
                
                # Get probabilities
                probs = self._softmax(logits)
                
                # Apply top_p sampling
                if do_sample and top_p < 1.0:
                    sorted_probs, sorted_indices = self._sort_by_probs(probs[0])
                    cumulative_probs = np.cumsum(sorted_probs)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first one above the threshold
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                    sorted_indices_to_remove[0] = False
                    
                    # Get indices to remove
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    
                    # Create a mask for probs
                    mask = np.ones_like(probs[0])
                    mask[indices_to_remove] = 0
                    
                    probs[0] = probs[0] * mask
                    # Renormalize
                    probs[0] = probs[0] / (probs[0].sum() + 1e-8)
                
                # Sample or greedy select next token
                if do_sample:
                    next_token_id = np.random.multinomial(1, probs[0]).argmax()
                else:
                    next_token_id = probs[0].argmax()
                
                # Add the new token to the generated sequence
                next_token_id = np.array([[next_token_id]], dtype=np.int64)
                generated_ids = np.concatenate([generated_ids, next_token_id], axis=1)
                
                # Stop if EOS token is generated
                if next_token_id[0, 0] == self.tokenizer.eos_token_id:
                    break
            
            # Decode the generated token IDs to text
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return prompt + "\n[ERROR: Generation failed]"
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to the input array."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _sort_by_probs(self, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sort probabilities in descending order and return the sorted values and indices."""
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        return sorted_probs, sorted_indices


def main():
    """Main function to demonstrate model usage."""
    parser = argparse.ArgumentParser(description="Phi-1.5 ONNX Model Inference")
    parser.add_argument(
        "--prompt", type=str, default="Once upon a time,",
        help="Input prompt for the model"
    )
    parser.add_argument(
        "--max_length", type=int, default=100,
        help="Maximum length of generated text including prompt"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Temperature for sampling (lower is more deterministic)"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9,
        help="Top-p sampling parameter (nucleus sampling)"
    )
    parser.add_argument(
        "--no_sample", action="store_true",
        help="Use greedy decoding instead of sampling"
    )
    parser.add_argument(
    )
    parser.add_argument(
        "--model_path", type=str, default="models/phi-1.5-onnx",
        help="Path to the directory containing the ONNX model and tokenizer files"
    )
    
    args = parser.parse_args()
    
    try:
        # Create and load the model
        model = PhiOnnxModel(model_path=args.model_path)
        
        # Generate text
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerating text...")
        
        generated_text = model.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=not args.no_sample
        )
        
        print("\nGenerated text:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

