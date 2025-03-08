class PhiOnnxModel:
    """
    A wrapper class for the Phi-1.5 ONNX model that uses onnxruntime.transformers
    for optimized inference.
    """
    
    def __init__(self, model_path: str = "models/phi-1.5-onnx", optimize: bool = True):
        """
        Initialize the Phi-1.5 ONNX model with optimizations.
        
        Args:
            model_path: Path to the directory containing the ONNX model and tokenizer files
            optimize: Whether to apply ONNX Runtime optimizations
        """
        self.model_path = model_path
        self.onnx_model_path = os.path.join(model_path, "model.onnx")
        self.tokenizer_path = model_path
        self.optimized_model_path = os.path.join(model_path, "model_optimized.onnx")
        
        # Check if model files exist
        if not os.path.exists(self.onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found at {self.onnx_model_path}")
        
        if not os.path.exists(os.path.join(model_path, "tokenizer.json")):
            raise FileNotFoundError(f"Tokenizer files not found in {model_path}")
        
        # Load configuration
        self.config_path = os.path.join(model_path, "config.json")
        if os.path.exists(self.config_path):
            self.config = AutoConfig.from_pretrained(model_path)
            logger.info(f"Loaded model config: {self.config.model_type}")
        else:
            logger.warning(f"Config file not found at {self.config_path}, using default settings")
            self.config = None
        
        # Load tokenizer
        try:
            logger.info(f"Loading tokenizer from {self.tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            
            # Optimize model if requested
            if optimize:
                self._optimize_model()
                model_to_load = self.optimized_model_path
            else:
                model_to_load = self.onnx_model_path
                
            logger.info(f"Loading ONNX model from {model_to_load}")
            # Create ONNX inference session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Initialize session with optimized model
            self.session = ort.InferenceSession(
                model_to_load, 
                session_options,
                providers=["CPUExecutionProvider"]
            )
            
            # Get model inputs and outputs
            self.model_inputs = [input.name for input in self.session.get_inputs()]
            self.model_outputs = [output.name for output in self.session.get_outputs()]
            
            # Log model information
            logger.info(f"Model loaded successfully with {len(self.model_inputs)} inputs and {len(self.model_outputs)} outputs")
            logger.info(f"Input names: {self.model_inputs}")
            logger.info(f"Output names: {self.model_outputs}")
            
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {str(e)}")
            raise

    def _optimize_model(self):
        """
        Optimize the ONNX model using onnxruntime.transformers if not already optimized.
        """
        if os.path.exists(self.optimized_model_path):
            logger.info(f"Using existing optimized model at {self.optimized_model_path}")
            return
            
        logger.info(f"Optimizing ONNX model...")
        start_time = time.time()
        
        # Set up optimization options
        fusion_options = FusionOptions("bert")
        fusion_options.enable_gelu = True
        fusion_options.enable_layer_norm = True
        fusion_options.enable_attention = True
        
        # Get model type from config for optimization
        model_type = "bert"  # default
        if self.config and hasattr(self.config, 'model_type'):
            model_type = self.config.model_type
            
        # Optimize the model
        optimize_model(
            input=self.onnx_model_path,
            output=self.optimized_model_path,
            model_type=model_type,
            num_heads=self.config.num_attention_heads if self.config else 32,
            hidden_size=self.config.hidden_size if self.config else 2560,
            optimization_options=fusion_options
        )
        
        logger.info(f"Model optimization completed in {time.time() - start_time:.2f} seconds")
