#!/usr/bin/env python3
"""
ISAC Training and Evaluation Framework
Comprehensive training, evaluation, and comparison system
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
from typing import Dict, List, Tuple
import os
from new_isac_system import (
    TimedomainISACNet, ISACDataset, TraditionalCFAR, 
    TraditionalOFDMDemod, TraditionalChannelEstimator
)

class ISACLoss(nn.Module):
    """
    Multi-task loss function for ISAC system
    """
    
    def __init__(self, 
                 radar_weight: float = 1.0,
                 comm_weight: float = 1.0,
                 channel_weight: float = 0.5,
                 rd_map_weight: float = 0.5):
        super(ISACLoss, self).__init__()
        
        self.radar_weight = radar_weight
        self.comm_weight = comm_weight
        self.channel_weight = channel_weight
        self.rd_map_weight = rd_map_weight
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs: Dict, targets: Dict) -> Dict:
        """
        Compute multi-task loss
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Radar detection loss (MSE for target parameters)
        radar_loss = self.mse_loss(outputs['radar_detections'], targets['targets'])
        losses['radar_loss'] = radar_loss * self.radar_weight
        
        # Communication symbol loss (Cross-entropy)
        batch_size, num_symbols, num_classes = outputs['comm_symbols'].shape
        comm_pred = outputs['comm_symbols'].view(-1, num_classes)
        comm_target = targets['comm_symbols'].view(-1)
        
        # Clamp targets to valid range to avoid out-of-bounds errors
        comm_target = torch.clamp(comm_target, 0, num_classes - 1)
        
        comm_loss = self.ce_loss(comm_pred, comm_target)
        losses['comm_loss'] = comm_loss * self.comm_weight
        
        # Channel estimation loss (MSE for complex coefficients)
        channel_loss = self.mse_loss(outputs['channel_estimate'], targets['channel_coeffs'])
        losses['channel_loss'] = channel_loss * self.channel_weight
        
        # Range-Doppler map loss (MSE)
        rd_map_loss = self.mse_loss(outputs['rd_map'], targets['rd_map'])
        losses['rd_map_loss'] = rd_map_loss * self.rd_map_weight
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses

class ISACTrainer:
    """
    Training framework for ISAC system
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function and optimizer
        self.criterion = ISACLoss()
        self.optimizer = optim.AdamW(model.parameters(), 
                                   lr=learning_rate, 
                                   weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # Training history
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        epoch_losses = {'total_loss': 0, 'radar_loss': 0, 'comm_loss': 0, 
                       'channel_loss': 0, 'rd_map_loss': 0}
        num_batches = 0
        
        for batch in self.train_loader:
            # Move data to device
            signals = batch['signal'].to(self.device)
            targets = {
                'targets': batch['targets'].to(self.device),
                'comm_symbols': batch['comm_symbols'].to(self.device),
                'channel_coeffs': batch['channel_coeffs'].to(self.device),
                'rd_map': batch['rd_map'].to(self.device)
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(signals)
            losses = self.criterion(outputs, targets)
            
            # Backward pass
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Returns:
            Dictionary of average losses
        """
        self.model.eval()
        epoch_losses = {'total_loss': 0, 'radar_loss': 0, 'comm_loss': 0, 
                       'channel_loss': 0, 'rd_map_loss': 0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                signals = batch['signal'].to(self.device)
                targets = {
                    'targets': batch['targets'].to(self.device),
                    'comm_symbols': batch['comm_symbols'].to(self.device),
                    'channel_coeffs': batch['channel_coeffs'].to(self.device),
                    'rd_map': batch['rd_map'].to(self.device)
                }
                
                # Forward pass
                outputs = self.model(signals)
                losses = self.criterion(outputs, targets)
                
                # Accumulate losses
                for key, value in losses.items():
                    epoch_losses[key] += value.item()
                num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train(self, num_epochs: int) -> None:
        """
        Train the model for specified number of epochs
        
        Args:
            num_epochs: Number of training epochs
        """
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Train
            train_losses = self.train_epoch()
            self.train_history.append(train_losses)
            
            # Validate
            val_losses = self.validate_epoch()
            self.val_history.append(val_losses)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                print(f"  Train Loss: {train_losses['total_loss']:.4f}")
                print(f"  Val Loss: {val_losses['total_loss']:.4f}")
                print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")
        
        print("Training completed!")

class ISACEvaluator:
    """
    Comprehensive evaluation and comparison framework
    """
    
    def __init__(self, 
                 model: nn.Module,
                 test_loader: DataLoader,
                 device: torch.device):
        
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        
        # Traditional baselines
        self.cfar_detector = TraditionalCFAR()
        self.ofdm_demod = TraditionalOFDMDemod()
        self.channel_estimator = TraditionalChannelEstimator()
        
        # Results storage
        self.results = {
            'neural_network': {},
            'traditional': {}
        }
    
    def evaluate_radar_detection(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate radar detection performance
        
        Returns:
            Detection performance metrics
        """
        print("Evaluating radar detection performance...")
        
        nn_detections = []
        traditional_detections = []
        ground_truth = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                signals = batch['signal'].to(self.device)
                targets = batch['targets'].cpu().numpy()
                rd_maps_gt = batch['rd_map'].cpu().numpy()
                
                # Neural network predictions
                outputs = self.model(signals)
                nn_rd_maps = outputs['rd_map'].cpu().numpy()
                
                # Process each sample in batch
                for i in range(len(signals)):
                    # Ground truth binary map - Fix: use appropriate threshold
                    gt_max = np.max(rd_maps_gt[i])
                    gt_threshold = max(0.1, gt_max * 0.3)  # Adaptive threshold
                    gt_binary = (rd_maps_gt[i] > gt_threshold).astype(int)
                    ground_truth.append(gt_binary.flatten())
                    
                    # Neural network detection - Fix: handle already sigmoid-activated output
                    # The model already applies sigmoid in forward pass, so no need to apply again
                    nn_output = nn_rd_maps[i]
                    
                    # Ensure output is in [0,1] range (sigmoid activated)
                    if np.max(nn_output) > 1.0 or np.min(nn_output) < 0.0:
                        # Apply sigmoid if not already applied
                        nn_output = 1 / (1 + np.exp(-np.clip(nn_output, -500, 500)))
                    
                    # Apply threshold for binary detection
                    nn_binary = (nn_output > 0.5).astype(int)
                    nn_detections.append(nn_binary.flatten())
                    
                    # Traditional CFAR detection
                    try:
                        cfar_detections = self.cfar_detector.detect(rd_maps_gt[i])
                        traditional_detections.append(cfar_detections.flatten())
                    except Exception as e:
                        print(f"Warning: CFAR detection failed for sample {i}: {e}")
                        # Use zeros as fallback
                        cfar_detections = np.zeros_like(gt_binary)
                        traditional_detections.append(cfar_detections.flatten())
        
        # Calculate metrics
        nn_detections = np.concatenate(nn_detections)
        traditional_detections = np.concatenate(traditional_detections)
        ground_truth = np.concatenate(ground_truth)
        
        # Debug information
        print(f"Radar detection evaluation:")
        print(f"  Ground truth positives: {np.sum(ground_truth)} / {len(ground_truth)}")
        print(f"  NN detections: {np.sum(nn_detections)} / {len(nn_detections)}")
        print(f"  Traditional detections: {np.sum(traditional_detections)} / {len(traditional_detections)}")
        
        # Neural network metrics - Fix: handle edge cases
        nn_accuracy = accuracy_score(ground_truth, nn_detections)
        if np.sum(ground_truth) > 0:  # Only calculate if there are positive samples
            nn_precision, nn_recall, nn_f1, _ = precision_recall_fscore_support(
                ground_truth, nn_detections, average='binary', zero_division=0
            )
        else:
            # No positive samples in ground truth
            nn_precision = 1.0 if np.sum(nn_detections) == 0 else 0.0
            nn_recall = 0.0
            nn_f1 = 0.0
        
        # Traditional CFAR metrics - Fix: handle edge cases
        trad_accuracy = accuracy_score(ground_truth, traditional_detections)
        if np.sum(ground_truth) > 0:  # Only calculate if there are positive samples
            trad_precision, trad_recall, trad_f1, _ = precision_recall_fscore_support(
                ground_truth, traditional_detections, average='binary', zero_division=0
            )
        else:
            # No positive samples in ground truth
            trad_precision = 1.0 if np.sum(traditional_detections) == 0 else 0.0
            trad_recall = 0.0
            trad_f1 = 0.0
        
        print(f"  NN metrics - Acc: {nn_accuracy:.4f}, Prec: {nn_precision:.4f}, Rec: {nn_recall:.4f}")
        print(f"  Traditional metrics - Acc: {trad_accuracy:.4f}, Prec: {trad_precision:.4f}, Rec: {trad_recall:.4f}")
        
        results = {
            'neural_network': {
                'accuracy': float(nn_accuracy),
                'precision': float(nn_precision),
                'recall': float(nn_recall),
                'f1_score': float(nn_f1)
            },
            'traditional': {
                'accuracy': float(trad_accuracy),
                'precision': float(trad_precision),
                'recall': float(trad_recall),
                'f1_score': float(trad_f1)
            }
        }
        
        return results
    
    def evaluate_communication(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate communication demodulation performance
        
        Returns:
            Communication performance metrics
        """
        print("Evaluating communication performance...")
        
        nn_correct = 0
        trad_correct = 0
        total_symbols = 0
        
        modulation_types = ['BPSK', 'QPSK', 'QAM16', 'QAM64']
        constellation_sizes = {'BPSK': 2, 'QPSK': 4, 'QAM16': 16, 'QAM64': 64}
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                signals = batch['signal'].to(self.device)
                comm_symbols_gt = batch['comm_symbols'].cpu().numpy()
                modulations = batch['modulation'].cpu().numpy()
                
                # Neural network predictions
                outputs = self.model(signals)
                nn_comm_logits = outputs['comm_symbols'].cpu().numpy()  # (batch, num_symbols, 64)
                
                # Process each sample in batch
                for i in range(len(signals)):
                    signal_np = batch['signal'][i].numpy()
                    signal_complex = signal_np[0] + 1j * signal_np[1]
                    modulation = modulation_types[modulations[i]]
                    constellation_size = constellation_sizes[modulation]
                    
                    gt_symbols = comm_symbols_gt[i]  # Ground truth symbols
                    
                    # Neural network prediction - only consider valid constellation points
                    nn_logits_valid = nn_comm_logits[i, :, :constellation_size]  # Only valid constellation points
                    nn_pred = np.argmax(nn_logits_valid, axis=-1)
                    
                    # Traditional demodulation
                    trad_pred = self.ofdm_demod.demodulate(signal_complex, modulation)
                    
                    # Ensure same length for comparison
                    min_len = min(len(gt_symbols), len(nn_pred), len(trad_pred))
                    gt_symbols = gt_symbols[:min_len]
                    nn_pred = nn_pred[:min_len]
                    trad_pred = trad_pred[:min_len]
                    
                    # Count correct predictions
                    nn_correct += np.sum(nn_pred == gt_symbols)
                    trad_correct += np.sum(trad_pred == gt_symbols)
                    total_symbols += min_len
        
        # Calculate accuracies
        nn_accuracy = nn_correct / total_symbols if total_symbols > 0 else 0
        trad_accuracy = trad_correct / total_symbols if total_symbols > 0 else 0
        
        results = {
            'neural_network': {
                'symbol_accuracy': nn_accuracy,
                'symbol_error_rate': 1 - nn_accuracy
            },
            'traditional': {
                'symbol_accuracy': trad_accuracy,
                'symbol_error_rate': 1 - trad_accuracy
            }
        }
        
        return results
    
    def evaluate_channel_estimation(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate channel estimation performance
        
        Returns:
            Channel estimation performance metrics
        """
        print("Evaluating channel estimation performance...")
        
        nn_mse_errors = []
        traditional_mse_errors = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                signals = batch['signal'].to(self.device)
                channel_coeffs_gt = batch['channel_coeffs'].cpu().numpy()
                
                # Neural network predictions
                outputs = self.model(signals)
                nn_channel_pred = outputs['channel_estimate'].cpu().numpy()
                
                # Process each sample in batch
                for i in range(len(signals)):
                    signal_np = batch['signal'][i].cpu().numpy()  # Fix: ensure CPU tensor
                    signal_complex = signal_np[0] + 1j * signal_np[1]
                    
                    # Ground truth (convert to complex) - Fix: ensure proper shape
                    if channel_coeffs_gt.shape[-1] == 2:  # Real/Imag format
                        gt_complex = channel_coeffs_gt[i, :, 0] + 1j * channel_coeffs_gt[i, :, 1]
                    else:  # Already complex or different format
                        gt_complex = channel_coeffs_gt[i].flatten()
                    
                    # Neural network prediction (convert to complex) - Fix: ensure proper shape
                    if nn_channel_pred.shape[-1] == 2:  # Real/Imag format
                        nn_complex = nn_channel_pred[i, :, 0] + 1j * nn_channel_pred[i, :, 1]
                    else:  # Different format
                        nn_complex = nn_channel_pred[i].flatten()
                    
                    # Traditional estimation
                    try:
                        # Fix: Use proper signal length for OFDM (64 subcarriers)
                        ofdm_signal = signal_complex[:64] if len(signal_complex) >= 64 else signal_complex
                        trad_estimate = self.channel_estimator.estimate(ofdm_signal)
                        
                        # Ensure same length - Fix: handle length mismatch properly
                        target_len = 64  # Standard OFDM subcarriers
                        
                        # Resize all arrays to target length
                        if len(gt_complex) != target_len:
                            gt_complex = np.resize(gt_complex, target_len)
                        if len(nn_complex) != target_len:
                            nn_complex = np.resize(nn_complex, target_len)
                        if len(trad_estimate) != target_len:
                            trad_estimate = np.resize(trad_estimate, target_len)
                        
                        # Calculate MSE errors with proper validation
                        nn_diff = nn_complex - gt_complex
                        trad_diff = trad_estimate - gt_complex
                        
                        nn_mse = np.mean(np.abs(nn_diff)**2)
                        trad_mse = np.mean(np.abs(trad_diff)**2)
                        
                        # Check for valid values - Fix: add debug info
                        if np.isfinite(nn_mse) and not np.isnan(nn_mse) and nn_mse > 0:
                            nn_mse_errors.append(nn_mse)
                        if np.isfinite(trad_mse) and not np.isnan(trad_mse) and trad_mse > 0:
                            traditional_mse_errors.append(trad_mse)
                            
                    except Exception as e:
                        print(f"Warning: Channel estimation failed for sample {i}: {e}")
                        continue
        
        # Calculate results with proper fallback values
        nn_mse_mean = np.mean(nn_mse_errors) if len(nn_mse_errors) > 0 else 1.0  # Fix: use realistic fallback
        nn_mse_std = np.std(nn_mse_errors) if len(nn_mse_errors) > 0 else 0.0
        trad_mse_mean = np.mean(traditional_mse_errors) if len(traditional_mse_errors) > 0 else 1.0  # Fix: use realistic fallback
        trad_mse_std = np.std(traditional_mse_errors) if len(traditional_mse_errors) > 0 else 0.0
        
        print(f"Channel estimation evaluation completed:")
        print(f"  NN samples: {len(nn_mse_errors)}, MSE: {nn_mse_mean:.6f}")
        print(f"  Traditional samples: {len(traditional_mse_errors)}, MSE: {trad_mse_mean:.6f}")
        
        results = {
            'neural_network': {
                'mse': nn_mse_mean,
                'std': nn_mse_std
            },
            'traditional': {
                'mse': trad_mse_mean,
                'std': trad_mse_std
            }
        }
        
        return results
    
    def evaluate_processing_time(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate processing time comparison
        
        Returns:
            Processing time metrics
        """
        print("Evaluating processing time...")
        
        nn_times = []
        traditional_times = []
        
        # Test on a subset of data
        test_samples = 100
        sample_count = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                if sample_count >= test_samples:
                    break
                
                signals = batch['signal'].to(self.device)
                
                # Neural network timing
                start_time = time.time()
                outputs = self.model(signals)
                nn_time = time.time() - start_time
                nn_times.append(nn_time / len(signals))  # Per sample
                
                # Traditional methods timing
                start_time = time.time()
                for i in range(len(signals)):
                    signal_np = batch['signal'][i].numpy()
                    signal_complex = signal_np[0] + 1j * signal_np[1]
                    rd_map = batch['rd_map'][i].numpy()
                    
                    # CFAR detection
                    _ = self.cfar_detector.detect(rd_map)
                    
                    # OFDM demodulation
                    _ = self.ofdm_demod.demodulate(signal_complex, 'QPSK')
                    
                    # Channel estimation
                    _ = self.channel_estimator.estimate(signal_complex)
                
                trad_time = time.time() - start_time
                traditional_times.append(trad_time / len(signals))  # Per sample
                
                sample_count += len(signals)
        
        results = {
            'neural_network': {
                'mean_time': np.mean(nn_times),
                'std_time': np.std(nn_times)
            },
            'traditional': {
                'mean_time': np.mean(traditional_times),
                'std_time': np.std(traditional_times)
            }
        }
        
        return results
    
    def comprehensive_evaluation(self) -> Dict:
        """
        Run comprehensive evaluation of all components
        
        Returns:
            Complete evaluation results
        """
        print("Starting comprehensive evaluation...")
        
        results = {
            'radar_detection': self.evaluate_radar_detection(),
            'communication': self.evaluate_communication(),
            'channel_estimation': self.evaluate_channel_estimation(),
            'processing_time': self.evaluate_processing_time()
        }
        
        return results

class ResultsVisualizer:
    """
    Visualization of comparison results
    """
    
    def __init__(self, results: Dict, save_dir: str = 'results'):
        self.results = results
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_radar_performance(self) -> None:
        """
        Plot radar detection performance comparison
        """
        radar_results = self.results['radar_detection']
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        nn_values = [radar_results['neural_network'][m] for m in metrics]
        trad_values = [radar_results['traditional'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, nn_values, width, label='Neural Network', alpha=0.8)
        ax.bar(x + width/2, trad_values, width, label='Traditional CFAR', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Radar Detection Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/radar_performance_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_communication_performance(self) -> None:
        """
        Plot communication performance comparison
        """
        comm_results = self.results['communication']
        
        methods = ['Neural Network', 'Traditional OFDM']
        accuracies = [
            comm_results['neural_network']['symbol_accuracy'],
            comm_results['traditional']['symbol_accuracy']
        ]
        error_rates = [
            comm_results['neural_network']['symbol_error_rate'],
            comm_results['traditional']['symbol_error_rate']
        ]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Symbol accuracy
        ax1.bar(methods, accuracies, alpha=0.8, color=['blue', 'orange'])
        ax1.set_ylabel('Symbol Accuracy')
        ax1.set_title('Symbol Accuracy Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Symbol error rate
        ax2.bar(methods, error_rates, alpha=0.8, color=['blue', 'orange'])
        ax2.set_ylabel('Symbol Error Rate')
        ax2.set_title('Symbol Error Rate Comparison')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/communication_performance_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_channel_estimation_performance(self) -> None:
        """
        Plot channel estimation performance comparison
        """
        channel_results = self.results['channel_estimation']
        
        methods = ['Neural Network', 'Traditional LS']
        mse_values = [
            channel_results['neural_network']['mse'],
            channel_results['traditional']['mse']
        ]
        std_values = [
            channel_results['neural_network']['std'],
            channel_results['traditional']['std']
        ]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(methods, mse_values, yerr=std_values, alpha=0.8, 
               color=['blue', 'orange'], capsize=5)
        ax.set_ylabel('Mean Squared Error')
        ax.set_title('Channel Estimation Performance Comparison')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/channel_estimation_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_processing_time_comparison(self) -> None:
        """
        Plot processing time comparison
        """
        time_results = self.results['processing_time']
        
        methods = ['Neural Network', 'Traditional Methods']
        mean_times = [
            time_results['neural_network']['mean_time'],
            time_results['traditional']['mean_time']
        ]
        std_times = [
            time_results['neural_network']['std_time'],
            time_results['traditional']['std_time']
        ]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(methods, mean_times, yerr=std_times, alpha=0.8, 
               color=['blue', 'orange'], capsize=5)
        ax.set_ylabel('Processing Time (seconds)')
        ax.set_title('Processing Time Comparison')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/processing_time_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_curves(self, trainer: ISACTrainer) -> None:
        """
        Plot training curves
        """
        epochs = range(1, len(trainer.train_history) + 1)
        
        # Extract losses
        train_total = [h['total_loss'] for h in trainer.train_history]
        val_total = [h['total_loss'] for h in trainer.val_history]
        
        train_radar = [h['radar_loss'] for h in trainer.train_history]
        val_radar = [h['radar_loss'] for h in trainer.val_history]
        
        train_comm = [h['comm_loss'] for h in trainer.train_history]
        val_comm = [h['comm_loss'] for h in trainer.val_history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        ax1.plot(epochs, train_total, label='Train', alpha=0.8)
        ax1.plot(epochs, val_total, label='Validation', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('Total Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Radar loss
        ax2.plot(epochs, train_radar, label='Train', alpha=0.8)
        ax2.plot(epochs, val_radar, label='Validation', alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Radar Loss')
        ax2.set_title('Radar Detection Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Communication loss
        ax3.plot(epochs, train_comm, label='Train', alpha=0.8)
        ax3.plot(epochs, val_comm, label='Validation', alpha=0.8)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Communication Loss')
        ax3.set_title('Communication Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if hasattr(trainer.scheduler, 'get_last_lr'):
            lrs = [trainer.scheduler.get_last_lr()[0] for _ in epochs]
            ax4.plot(epochs, lrs, alpha=0.8)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_curves.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_plots(self, trainer: ISACTrainer = None) -> None:
        """
        Generate all comparison plots
        
        Args:
            trainer: Optional trainer for training curves
        """
        print("Generating visualization plots...")
        
        self.plot_radar_performance()
        self.plot_communication_performance()
        self.plot_channel_estimation_performance()
        self.plot_processing_time_comparison()
        
        if trainer is not None:
            self.plot_training_curves(trainer)
        
        print(f"All plots saved to {self.save_dir}/")

if __name__ == "__main__":
    print("ISAC Training and Evaluation Framework")
    print("=====================================")
    
    # Test the framework
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create small dataset for testing
    train_dataset = ISACDataset(num_samples=100)
    val_dataset = ISACDataset(num_samples=50)
    test_dataset = ISACDataset(num_samples=50)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Create model
    model = TimedomainISACNet()
    
    # Test trainer
    trainer = ISACTrainer(model, train_loader, val_loader, device)
    print("\nTrainer initialized successfully!")
    
    # Test evaluator
    evaluator = ISACEvaluator(model, test_loader, device)
    print("Evaluator initialized successfully!")
    
    print("\nFramework ready for training and evaluation!")