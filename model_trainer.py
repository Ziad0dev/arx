from advanced_ai_analyzer import logger, CONFIG
import numpy as np
import random # Needed if curriculum/selection involves randomness
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from advanced_ai_analyzer_learning import LearningSystem, SimpleClassifier, EnhancedClassifier
import logging
import time
import os
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Advanced model trainer with multi-GPU support and mixed precision training"""
    
    def __init__(self, engine):
        """Initialize model trainer
        
        Args:
            engine: Main research engine instance
        """
        self.engine = engine
        self.learner = engine.learner
        
        # Training settings
        self.use_mixed_precision = CONFIG.get('use_mixed_precision', True)
        self.gradient_accumulation_steps = CONFIG.get('gradient_accumulation_steps', 4)
        self.max_grad_norm = CONFIG.get('max_grad_norm', 1.0)
        
        # Multi-GPU settings
        self.use_distributed = CONFIG.get('use_distributed_training', False)
        self.world_size = CONFIG.get('distributed_world_size', torch.cuda.device_count())
        self.backend = CONFIG.get('distributed_backend', 'nccl' if torch.cuda.is_available() else 'gloo')
        
        # Current model tracking
        self.current_model = None
        self.current_model_type = None
        
        # Performance history
        self.training_history = {}
        
        if self.use_mixed_precision and not torch.cuda.is_available():
            logger.warning("Mixed precision training requires CUDA. Disabling mixed precision.")
            self.use_mixed_precision = False
            
        if self.use_distributed and self.world_size <= 1:
            logger.warning("Distributed training requires multiple GPUs. Disabling distributed training.")
            self.use_distributed = False
            
        # Initialize mixed precision if available and enabled
        self.scaler = None
        if self.use_mixed_precision:
            try:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
                logger.info("Mixed precision training enabled with GradScaler")
            except ImportError:
                logger.warning("Mixed precision training not available. Using standard precision.")
                self.use_mixed_precision = False

        # State Attributes (moved or managed here)
        self.model_types = CONFIG.get('model_types', ['enhanced', 'simple']) # Get from config
        self.current_model = CONFIG.get('initial_model_type', 'enhanced') # Initial model from config
        self.learning_rate_schedule = {} # Potentially track LR per model or domain
        # adaptive_exploration is read from engine config/state currently

    def select_best_model(self):
        """Select the best performing model type based on evaluation history."""
        performance_history = self.engine.performance_history
        if len(performance_history) < len(self.model_types): # Need at least one eval per model type
            return self.current_model # Not enough data, keep current

        # Get latest performance for each model type
        model_performances = {}
        try:
            for eval_data in reversed(performance_history):
                model_type = eval_data.get('model_type')
                metrics = eval_data.get('metrics')
                if model_type not in model_performances and metrics and 'overall_f1' in metrics:
                    model_performances[model_type] = metrics['overall_f1']

                # Stop once we have the latest F1 for all model types
                if len(model_performances) == len(self.model_types):
                    break

            if not model_performances: # No valid performance data found
                 return self.current_model

            # Select best model based on F1 score
            best_model = max(model_performances.items(), key=lambda item: item[1])[0]

            if best_model != self.current_model:
                logger.info(f"Switching to better performing model: {best_model} (F1: {model_performances[best_model]:.4f}) from {self.current_model} (F1: {model_performances.get(self.current_model, 0):.4f})")
                self.current_model = best_model
            else:
                logger.debug(f"Keeping current model: {self.current_model}")

        except Exception as e:
             logger.error(f"Error during model selection: {e}. Keeping current model {self.current_model}.")

        return self.current_model

    def apply_curriculum_learning(self):
        """Apply curriculum learning adjustments based on iteration count."""
        # This method currently modifies the engine's exploration_rate directly.
        # It should perhaps return desired parameters or adjustments instead.
        # For now, let's keep the logic similar but note this dependency.

        iteration = self.engine.iteration
        stage = min(3, iteration // CONFIG.get('curriculum_stage_length', 5))

        if stage == 0:
            logger.info("Curriculum Stage 0: Focusing on core concepts and clear categories")
            # Modify strategist's exploration rate? Or return a desired rate?
            # Let's assume strategist has a method to set exploration rate
            if hasattr(self.engine.query_strategist, 'set_exploration_rate'):
                 self.engine.query_strategist.set_exploration_rate(CONFIG.get('stage0_exploration', 0.3))
        elif stage == 1:
            logger.info("Curriculum Stage 1: Incorporating diverse papers and concepts")
            if hasattr(self.engine.query_strategist, 'set_exploration_rate'):
                 self.engine.query_strategist.set_exploration_rate(CONFIG.get('stage1_exploration', 0.4))
        elif stage == 2:
            logger.info("Curriculum Stage 2: Focusing on research themes and connections")
            if hasattr(self.engine.query_strategist, 'set_exploration_rate'):
                 self.engine.query_strategist.set_exploration_rate(CONFIG.get('stage2_exploration', 0.3))
        else: # stage == 3
            logger.info("Curriculum Stage 3: Targeting knowledge gaps and cutting-edge research")
            if hasattr(self.engine.query_strategist, 'set_exploration_rate'):
                 self.engine.query_strategist.set_exploration_rate(CONFIG.get('stage3_exploration', 0.2))

        # Sync engine's exploration rate for transparency (optional)
        # self.engine.exploration_rate = self.engine.query_strategist.exploration_rate


    def calculate_adaptive_learning_rate(self):
        """Calculate adaptive learning rate based on iteration and performance."""
        base_lr = CONFIG.get('base_learning_rate', 1e-4)
        iteration = self.engine.iteration
        performance_history = self.engine.performance_history

        # Adjust based on iteration number (decaying effect)
        iteration_decay_factor = CONFIG.get('lr_iteration_decay_rate', 100.0)
        iteration_factor = max(0.1, 1.0 - (iteration / iteration_decay_factor))

        # Adjust based on recent performance (F1 score)
        performance_factor = 1.0
        if performance_history:
             # Use the F1 score of the *current* model if available, otherwise overall latest
             latest_f1 = 0.5 # Default assumption
             for eval_data in reversed(performance_history):
                  if eval_data.get('model_type') == self.current_model and eval_data.get('metrics'):
                       latest_f1 = eval_data['metrics'].get('overall_f1', 0.5)
                       break
             else: # If no specific eval for current model found, use latest overall
                   latest_eval = performance_history[-1].get('metrics', {})
                   latest_f1 = latest_eval.get('overall_f1', 0.5)

             # Scale factor based on F1: Higher F1 -> slightly lower LR, Lower F1 -> slightly higher LR
             # Example scaling: (can be tuned)
             performance_factor = 1.0 + (0.5 - latest_f1) * 0.5 # Max increase 25%, max decrease 25%
             performance_factor = max(0.75, min(1.25, performance_factor))
        
        adaptive_lr = base_lr * iteration_factor * performance_factor
        min_lr = CONFIG.get('min_learning_rate', 1e-6)
        adaptive_lr = max(min_lr, adaptive_lr) # Ensure LR doesn't go below minimum

        logger.info(f"Using adaptive learning rate: {adaptive_lr:.2e} (Base: {base_lr:.1e}, IterFactor: {iteration_factor:.2f}, PerfFactor: {performance_factor:.2f})")
        return adaptive_lr

    def train_current_model(self):
        """Train the currently selected model using adaptive LR."""
        model_to_train = self.current_model
        learning_rate = self.calculate_adaptive_learning_rate()

        logger.info(f"Training {model_to_train} model with adaptive LR {learning_rate:.2e}...")
        try:
            # Assuming learner.train takes model_type and learning_rate
            self.learner.train(model_type=model_to_train, learning_rate=learning_rate)
            return True
        except Exception as e:
             logger.error(f"Training failed for model {model_to_train}: {e}")
             return False

    def evaluate_current_model(self):
        """Evaluate the currently selected model."""
        model_to_evaluate = self.current_model
        logger.info(f"Evaluating {model_to_evaluate} model...")
        try:
            # Assuming learner.evaluate takes model_type
            eval_results = self.learner.evaluate(model_type=model_to_evaluate)
            if eval_results is None:
                 logger.warning(f"Evaluation returned None for model {model_to_evaluate}. Using defaults.")
                 return {'overall_f1': 0.0, 'accuracy': 0.0, 'status': 'evaluation_failed'}
            return eval_results
        except Exception as e:
             logger.error(f"Evaluation failed for model {model_to_evaluate}: {e}")
             return {'overall_f1': 0.0, 'accuracy': 0.0, 'status': f'evaluation_error: {e}'}

    def train_and_evaluate(self, model_type='transformer'):
        """Train and evaluate model
        
        Args:
            model_type (str): Model type to train
            
        Returns:
            dict: Evaluation results
        """
        # Start distributed training if enabled
        if self.use_distributed:
            logger.info(f"Starting distributed training on {self.world_size} GPUs")
            mp.spawn(
                self._distributed_train_worker,
                args=(self.world_size, model_type),
                nprocs=self.world_size,
                join=True
            )
            # After distributed training completes, load the best checkpoint
            self.learner.load_model(model_type)
            self.current_model = model_type
            self.current_model_type = model_type
            return self.learner.evaluate(model_type)
        else:
            # Regular training
            start_time = time.time()
            self.current_model = model_type
            self.current_model_type = model_type
            
            # Get data from learner
            X_train, X_val, y_train, y_val = self.learner.prepare_data()
            if X_train is None or y_train is None:
                logger.error("No valid training data available")
                return None
                
            # Train the model
            logger.info(f"Training {model_type} model")
            self.learner.train(model_type)
            
            # Evaluate the model
            eval_results = self.learner.evaluate(model_type)
            
            # Log training time
            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time:.2f} seconds")
            
            return eval_results
    
    def _distributed_train_worker(self, rank, world_size, model_type):
        """Worker function for distributed training
        
        Args:
            rank (int): Process rank
            world_size (int): Total number of processes
            model_type (str): Model type to train
        """
        # Set up distributed process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        logger.info(f"Initializing process group on rank {rank} (world_size: {world_size})")
        dist.init_process_group(backend=self.backend, rank=rank, world_size=world_size)
        
        # Set device for this process
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
        
        # Get data from learner (only on rank 0)
        X_train, X_val, y_train, y_val = None, None, None, None
        if rank == 0:
            X_train, X_val, y_train, y_val = self.learner.prepare_data()
            if X_train is None or y_train is None:
                logger.error("No valid training data available")
                dist.destroy_process_group()
                return
                
        # Broadcast data from rank 0 to all other processes
        # In real implementation, we'd use DistributedSampler instead
        
        # Create model for this device
        model = self._create_model(model_type, device, rank)
        
        # Set up optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG.get('learning_rate', 1e-4),
            weight_decay=CONFIG.get('weight_decay', 1e-5)
        )
        
        # Train the model
        self._train_distributed(model, optimizer, X_train, y_train, X_val, y_val, rank, device)
        
        # Save model on rank 0
        if rank == 0:
            self.learner.save_model(model_type)
            
        # Clean up
        dist.destroy_process_group()
    
    def _create_model(self, model_type, device, rank=0):
        """Create model for training
        
        Args:
            model_type (str): Model type to create
            device (torch.device): Device to put model on
            rank (int): Process rank for distributed training
            
        Returns:
            nn.Module: Model
        """
        # Determine input size and num categories
        input_size = self.learner.X.shape[1] if hasattr(self.learner, 'X') else CONFIG.get('embedding_size', 768)
        num_categories = len(self.learner.categories) if hasattr(self.learner, 'categories') else 10
        hidden_size = CONFIG.get('hidden_size', 768)
        
        # Initialize model
        if model_type == 'enhanced':
            model = EnhancedClassifier(input_size, hidden_size, num_categories)
        elif model_type == 'simple':
            model = SimpleClassifier(input_size, hidden_size, num_categories)
        else:
            logger.warning(f"Unknown model type: {model_type}, using enhanced model")
            model = EnhancedClassifier(input_size, hidden_size, num_categories)
        
        # Enable gradient checkpointing for memory efficiency if supported
        if CONFIG.get('use_gradient_checkpointing', False) and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for memory efficiency")
        
        # Move model to device
        model = model.to(device)
        
        # Wrap model in DistributedDataParallel for distributed training
        if self.use_distributed:
            model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
            logger.info(f"Model wrapped in DistributedDataParallel on rank {rank}")
            
        return model
    
    def _train_distributed(self, model, optimizer, X_train, y_train, X_val, y_val, rank, device):
        """Train model in distributed mode
        
        Args:
            model (nn.Module): Model to train
            optimizer (optim.Optimizer): Optimizer
            X_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation data
            y_val (np.ndarray): Validation labels
            rank (int): Process rank
            device (torch.device): Device to use
        """
        # To be implemented in future
        pass
    
    def incremental_train(self, new_papers, model_type='transformer'):
        """Train model incrementally with new papers
        
        Args:
            new_papers (list): List of new paper IDs to train on
            model_type (str): Model type to train
            
        Returns:
            dict: Evaluation results
        """
        logger.info(f"Incremental training {model_type} model with {len(new_papers)} new papers")
        
        # Ensure we have a learner instance
        if not self.learner:
            logger.error("No learner instance available")
            return None
            
        # Set current model type if not already set
        if not self.current_model:
            self.current_model = model_type
            self.current_model_type = model_type
            
        # Call incremental training on the learner
        self.learner.incremental_train(new_papers, model_type, learning_rate=CONFIG.get('incremental_learning_rate', 5e-5))
        
        # Evaluate the model
        eval_results = self.learner.evaluate(model_type)
        
        return eval_results

    # Placeholder for potential future method
    # def attempt_model_improvement(self):
    #     """Periodically attempt model architecture improvements (e.g., hyperparameter tuning)."""
    #     logger.info("Checking for potential model architecture improvements...")
    #     # Implementation depends heavily on the LearningSystem capabilities
    #     pass 