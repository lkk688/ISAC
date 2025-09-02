#!/usr/bin/env python3
"""
Test script for 3D Range-Doppler map visualization.

This script demonstrates the new 3D visualization capabilities of the ISAC system,
including multiple target scenarios with ground truth marking.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict

# Add the ISAC directory to Python path
sys.path.append('/home/lkk/Developer/ISAC')

# Import only the dataset from new_isac_system to avoid dependency issues
try:
    from new_isac_system import ISACDataset
except ImportError:
    print("Error: Could not import ISACDataset from new_isac_system.py")
    print("Please ensure the file exists and is accessible.")
    sys.exit(1)

class RangeDopplerVisualizer:
    """Standalone visualizer for Range-Doppler maps with 3D capabilities."""
    
    def __init__(self, max_range=150, max_velocity=50, num_chirps=128, samples_per_chirp=256):
        self.max_range = max_range
        self.max_velocity = max_velocity
        self.num_chirps = num_chirps
        self.samples_per_chirp = samples_per_chirp
        self.velocity_resolution = 2 * max_velocity / num_chirps
        
    def generate_range_doppler_map(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate range-Doppler map using traditional FFT processing.
        
        Args:
            signal: Input signal [num_rx, num_chirps, samples_per_chirp]
        
        Returns:
            Range-Doppler map [1, doppler_bins, range_bins]
        """
        # Handle different signal shapes
        if signal.ndim == 1:
            # 1D signal - create synthetic range-doppler map
            range_fft = np.fft.fft(signal)
            doppler_bins = self.num_chirps
            range_bins = len(range_fft)
            
            # Create range-doppler map by replicating range profile
            rd_map = np.tile(range_fft, (doppler_bins, 1))
            
            # Add realistic Doppler variations
            for i in range(doppler_bins):
                doppler_freq = (i - doppler_bins//2) / doppler_bins
                phase_shift = np.exp(1j * 2 * np.pi * doppler_freq * np.arange(range_bins) / range_bins)
                rd_map[i, :] *= phase_shift
                
                # Add some noise for realism
                noise_level = 0.1
                rd_map[i, :] += noise_level * (np.random.randn(range_bins) + 1j * np.random.randn(range_bins))
        
        else:
            # Multi-dimensional signal - use original processing
            # Average across receive antennas
            signal_avg = np.mean(signal, axis=0)
            
            # Range FFT (along samples dimension)
            range_fft = np.fft.fft(signal_avg, axis=1)
            
            # Doppler FFT (along chirps dimension)
            doppler_fft = np.fft.fft(range_fft, axis=0)
            doppler_fft = np.fft.fftshift(doppler_fft, axes=0)
            
            # Convert to magnitude
            rd_map = doppler_fft
        
        # Convert to magnitude
        rd_map = np.abs(rd_map)
        
        return rd_map[np.newaxis, :, :]  # Add batch dimension
    
    def visualize_range_doppler_map(self, rd_map: np.ndarray, targets: List[Dict], 
                                   save_path: str = None, show_plot: bool = True, plot_3d: bool = True) -> None:
        """
        Visualize range-Doppler map with ground truth target markers in 2D or 3D.
        
        Args:
            rd_map: Range-Doppler map [1, doppler_bins, range_bins]
            targets: List of target dictionaries
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            plot_3d: Whether to create 3D plot (True) or 2D plot (False)
        """
        # Remove batch dimension
        rd_map_2d = rd_map[0] if rd_map.ndim == 3 else rd_map
        
        # Get actual dimensions from the RD map
        num_doppler_bins, num_range_bins = rd_map_2d.shape
        
        # Create range and velocity axes based on actual dimensions
        range_axis = np.linspace(0, self.max_range, num_range_bins)
        velocity_axis = np.linspace(-self.max_velocity, self.max_velocity, num_doppler_bins)
        
        if plot_3d:
            # Create 3D plot
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create meshgrid for 3D surface
            Range, Velocity = np.meshgrid(range_axis, velocity_axis)
            
            # Convert to dB scale
            rd_map_db = 20 * np.log10(rd_map_2d + 1e-10)
            
            # Create 3D surface plot
            surf = ax.plot_surface(Range, Velocity, rd_map_db, 
                                 cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
            
            # Mark ground truth targets
            for i, target in enumerate(targets):
                target_range = target['range']
                target_velocity = target['velocity']
                target_rcs = target.get('rcs', 0)
                
                # Find the closest indices in the RD map
                range_idx = np.argmin(np.abs(range_axis - target_range))
                velocity_idx = np.argmin(np.abs(velocity_axis - target_velocity))
                
                # Ensure indices are within bounds
                range_idx = min(range_idx, num_range_bins - 1)
                velocity_idx = min(velocity_idx, num_doppler_bins - 1)
                
                # Get the RD map value at target location
                rd_value = rd_map_db[velocity_idx, range_idx]
                
                # Plot target as 3D scatter point
                ax.scatter([target_range], [target_velocity], [rd_value + 5], 
                          c='red', s=200, marker='x', linewidths=4,
                          label=f'Target {i+1}' if i == 0 else '')
                
                # Add vertical line from base to target
                ax.plot([target_range, target_range], [target_velocity, target_velocity], 
                       [np.min(rd_map_db), rd_value + 5], 'r--', linewidth=2, alpha=0.7)
                
                # Add text annotation
                ax.text(target_range, target_velocity, rd_value + 10,
                       f'T{i+1}\nR:{target_range:.1f}m\nV:{target_velocity:.1f}m/s\nRCS:{target_rcs:.1f}dBsm',
                       fontsize=8, ha='center')
            
            # Set labels and title
            ax.set_xlabel('Range (m)', fontsize=12)
            ax.set_ylabel('Velocity (m/s)', fontsize=12)
            ax.set_zlabel('Magnitude (dB)', fontsize=12)
            ax.set_title('3D Range-Doppler Map with Ground Truth', fontsize=14, fontweight='bold')
            
            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Magnitude (dB)')
            
            if len(targets) > 0:
                ax.legend(loc='upper left')
        
        else:
            # Create 2D plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Convert to dB scale
            rd_map_db = 20 * np.log10(rd_map_2d + 1e-10)
            
            # Create 2D image
            im = ax.imshow(rd_map_db, aspect='auto', origin='lower', 
                          extent=[0, self.max_range, -self.max_velocity, self.max_velocity],
                          cmap='viridis')
            
            # Mark ground truth targets
            for i, target in enumerate(targets):
                target_range = target['range']
                target_velocity = target['velocity']
                target_rcs = target.get('rcs', 0)
                
                # Find the closest indices in the RD map
                range_idx = np.argmin(np.abs(range_axis - target_range))
                velocity_idx = np.argmin(np.abs(velocity_axis - target_velocity))
                
                # Ensure indices are within bounds
                range_idx = min(range_idx, num_range_bins - 1)
                velocity_idx = min(velocity_idx, num_doppler_bins - 1)
                
                ax.scatter(target_range, target_velocity, 
                          c='red', s=100, marker='x', linewidths=3,
                          label=f'Target {i+1}' if i == 0 else '')
                
                # Add target information
                ax.annotate(f'T{i+1}\nR:{target_range:.1f}m\nV:{target_velocity:.1f}m/s\nRCS:{target_rcs:.1f}dBsm',
                           xy=(target_range, target_velocity),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                           fontsize=8)
            
            ax.set_xlabel('Range (m)', fontsize=12)
            ax.set_ylabel('Velocity (m/s)', fontsize=12)
            ax.set_title('Range-Doppler Map with Ground Truth', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Magnitude (dB)', fontsize=12)
            
            if len(targets) > 0:
                ax.legend()
            ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Range-Doppler map saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()

def main():
    """Main function to test 3D range-doppler visualization."""
    print("Testing 3D Range-Doppler Map Visualization")
    print("=" * 50)
    
    # Initialize ISAC dataset
    dataset = ISACDataset(
        num_samples=100,
        sequence_length=1,
        num_targets=3,
        num_symbols=64,
        snr_range=(10, 30),
        modulation_types=['QPSK']
    )
    
    # Initialize visualizer
    visualizer = RangeDopplerVisualizer()
    
    # Create output directory for saving plots
    output_dir = "/home/lkk/Developer/ISAC/3d_rd_maps"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\nGenerating 3D Range-Doppler maps...")
    print(f"Output directory: {output_dir}")
    
    # Define test scenarios with different target configurations
    scenarios = [
        # Scenario 1: Single target
        [{'range': 50.0, 'velocity': 10.0, 'rcs': 1.0}],
        
        # Scenario 2: Two targets at different ranges
        [{'range': 30.0, 'velocity': 5.0, 'rcs': 1.2},
         {'range': 80.0, 'velocity': -8.0, 'rcs': 0.8}],
        
        # Scenario 3: Three targets with varying velocities
        [{'range': 25.0, 'velocity': 15.0, 'rcs': 1.5},
         {'range': 60.0, 'velocity': 0.0, 'rcs': 1.0},
         {'range': 95.0, 'velocity': -12.0, 'rcs': 0.6}],
        
        # Scenario 4: Multiple targets close in range
        [{'range': 40.0, 'velocity': 8.0, 'rcs': 1.1},
         {'range': 45.0, 'velocity': 3.0, 'rcs': 0.9},
         {'range': 50.0, 'velocity': -5.0, 'rcs': 1.3}],
        
        # Scenario 5: High-velocity targets
        [{'range': 70.0, 'velocity': 25.0, 'rcs': 0.7},
         {'range': 120.0, 'velocity': -20.0, 'rcs': 1.4}]
    ]
    
    for i, targets in enumerate(scenarios):
        print(f"\nScenario {i+1}: {len(targets)} target(s)")
        
        # Convert targets to numpy array format expected by the dataset
        targets_array = np.array([[t['range'], t['velocity'], t['rcs']] for t in targets], dtype=np.float64)
        
        # Generate synthetic signal with targets using the private method
        signal = dataset._generate_fmcw_signal(targets_array)
        
        # Generate ground truth range-doppler map using the dataset's method
        rd_map = dataset._generate_rd_map(targets_array)
        
        # Convert to proper format for visualization (add batch dimension)
        rd_map = rd_map[np.newaxis, :, :]  # Shape: [1, 64, 64]
        
        # Create save path
        save_path_3d = os.path.join(output_dir, f'3d_rd_map_scenario_{i+1}.png')
        save_path_2d = os.path.join(output_dir, f'2d_rd_map_scenario_{i+1}.png')
        
        # Visualize with 3D plot
        visualizer.visualize_range_doppler_map(
            rd_map=rd_map,
            targets=targets,
            save_path=save_path_3d,
            show_plot=False,  # Don't show to avoid blocking
            plot_3d=True
        )
        
        # Also create 2D plot for comparison
        visualizer.visualize_range_doppler_map(
            rd_map=rd_map,
            targets=targets,
            save_path=save_path_2d,
            show_plot=False,
            plot_3d=False
        )
        
        # Print target information
        for j, target in enumerate(targets):
            print(f"  Target {j+1}: Range={target['range']:.1f}m, "
                  f"Velocity={target['velocity']:.1f}m/s, RCS={target['rcs']:.1f}dBsm")
    

    
    print(f"\nCompleted generation of {len(scenarios)} Range-Doppler maps.")
    print(f"All plots saved to: {output_dir}")
    print("\n3D Range-Doppler visualization test completed!")

if __name__ == "__main__":
    main()