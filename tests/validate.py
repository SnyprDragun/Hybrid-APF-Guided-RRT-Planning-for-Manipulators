#!/usr/bin/env python3
"""
Benchmarking & Statistical Evaluation Suite for APF-RRT*
Compares Baseline vs. Enhanced (Adaptive) versions
"""

import numpy as np
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt

@dataclass
class BenchmarkResult:
    """Single trial result"""
    trial_id: int
    success: bool
    computation_time: float  # seconds
    path_length: float  # Euclidean distance in config space
    node_count: int
    first_solution_time: Optional[float]
    final_cost: float
    variant: str  # 'baseline' or 'adaptive'


class BenchmarkSuite:
    """Run and compare planner variants"""
    
    def __init__(self, n_trials: int = 20):
        self.n_trials = n_trials
        self.results_baseline = []
        self.results_adaptive = []
    
    def run_trial(self, planner_func, trial_id: int, variant: str) -> BenchmarkResult:
        """Execute single planning trial"""
        start_time = time.time()
        
        path, tree, metrics = planner_func()
        
        elapsed = time.time() - start_time
        
        # Compute metrics
        success = path is not None
        path_length = self._compute_path_length(path) if success else np.inf
        node_count = len(tree)
        first_solution_time = metrics.get('time_to_first_solution', None)
        final_cost = metrics.get('final_cost', np.inf) if success else np.inf
        
        result = BenchmarkResult(
            trial_id=trial_id,
            success=success,
            computation_time=elapsed,
            path_length=path_length,
            node_count=node_count,
            first_solution_time=first_solution_time,
            final_cost=final_cost,
            variant=variant
        )
        
        return result
    
    def _compute_path_length(self, path: List[np.ndarray]) -> float:
        """Total Euclidean distance along path"""
        if not path or len(path) < 2:
            return np.inf
        
        length = 0.0
        for i in range(len(path) - 1):
            length += np.linalg.norm(path[i+1] - path[i])
        return length
    
    def run_benchmark(self, planner_baseline, planner_adaptive):
        """Run full benchmark suite"""
        print("=" * 70)
        print(f"Running {self.n_trials} trials per variant...")
        print("=" * 70)
        
        # Baseline trials
        print("\n[BASELINE] Running trials...")
        for trial_id in range(self.n_trials):
            result = self.run_trial(planner_baseline, trial_id, 'baseline')
            self.results_baseline.append(result)
            status = "✓" if result.success else "✗"
            print(f"  Trial {trial_id+1}/{self.n_trials} {status} "
                  f"Time: {result.computation_time:.2f}s "
                  f"Nodes: {result.node_count}")
        
        # Adaptive trials
        print("\n[ADAPTIVE] Running trials...")
        for trial_id in range(self.n_trials):
            result = self.run_trial(planner_adaptive, trial_id, 'adaptive')
            self.results_adaptive.append(result)
            status = "✓" if result.success else "✗"
            print(f"  Trial {trial_id+1}/{self.n_trials} {status} "
                  f"Time: {result.computation_time:.2f}s "
                  f"Nodes: {result.node_count}")
    
    def generate_comparison_table(self) -> Dict:
        """Create comparison table for report"""
        
        def compute_stats(results: List[BenchmarkResult]) -> Dict:
            success_count = sum(1 for r in results if r.success)
            success_rate = 100 * success_count / len(results)
            
            times = [r.computation_time for r in results if r.success]
            path_lengths = [r.path_length for r in results if r.success]
            node_counts = [r.node_count for r in results]
            first_times = [r.first_solution_time for r in results 
                          if r.first_solution_time is not None]
            
            return {
                'success_rate': success_rate,
                'avg_computation_time': np.mean(times) if times else np.inf,
                'std_computation_time': np.std(times) if times else 0,
                'avg_path_length': np.mean(path_lengths) if path_lengths else np.inf,
                'std_path_length': np.std(path_lengths) if path_lengths else 0,
                'avg_node_count': np.mean(node_counts),
                'std_node_count': np.std(node_counts),
                'avg_first_solution_time': np.mean(first_times) if first_times else np.inf,
            }
        
        baseline_stats = compute_stats(self.results_baseline)
        adaptive_stats = compute_stats(self.results_adaptive)
        
        # Compute improvements
        improvements = {
            'success_rate_delta': adaptive_stats['success_rate'] - baseline_stats['success_rate'],
            'time_speedup': (baseline_stats['avg_computation_time'] / 
                            adaptive_stats['avg_computation_time'] 
                            if adaptive_stats['avg_computation_time'] > 0 else 1.0),
            'path_length_improvement': ((baseline_stats['avg_path_length'] - 
                                        adaptive_stats['avg_path_length']) / 
                                       baseline_stats['avg_path_length'] * 100
                                       if baseline_stats['avg_path_length'] > 0 else 0),
            'node_efficiency': (baseline_stats['avg_node_count'] / 
                               adaptive_stats['avg_node_count']
                               if adaptive_stats['avg_node_count'] > 0 else 1.0),
        }
        
        return {
            'baseline': baseline_stats,
            'adaptive': adaptive_stats,
            'improvements': improvements
        }
    
    def print_comparison_table(self):
        """Pretty-print comparison table"""
        comparison = self.generate_comparison_table()
        
        print("\n" + "=" * 80)
        print("COMPARATIVE ANALYSIS: BASELINE vs. ADAPTIVE APF-RRT*")
        print("=" * 80)
        
        print(f"\n{'Metric':<35} {'Baseline':<20} {'Adaptive':<20}")
        print("-" * 80)
        
        baseline = comparison['baseline']
        adaptive = comparison['adaptive']
        
        print(f"{'Success Rate (%)':<35} "
              f"{baseline['success_rate']:>18.1f}% "
              f"{adaptive['success_rate']:>18.1f}%")
        
        print(f"{'Avg Computation Time (s)':<35} "
              f"{baseline['avg_computation_time']:>18.3f} "
              f"{adaptive['avg_computation_time']:>18.3f}")
        
        print(f"{'Std Dev Computation Time (s)':<35} "
              f"{baseline['std_computation_time']:>18.3f} "
              f"{adaptive['std_computation_time']:>18.3f}")
        
        print(f"{'Avg Path Length (config-space)':<35} "
              f"{baseline['avg_path_length']:>18.3f} "
              f"{adaptive['avg_path_length']:>18.3f}")
        
        print(f"{'Std Dev Path Length':<35} "
              f"{baseline['std_path_length']:>18.3f} "
              f"{adaptive['std_path_length']:>18.3f}")
        
        print(f"{'Avg Node Count':<35} "
              f"{baseline['avg_node_count']:>18.1f} "
              f"{adaptive['avg_node_count']:>18.1f}")
        
        print(f"{'Avg Time to First Solution (s)':<35} "
              f"{baseline['avg_first_solution_time']:>18.3f} "
              f"{adaptive['avg_first_solution_time']:>18.3f}")
        
        print("\n" + "-" * 80)
        print("IMPROVEMENTS (Adaptive over Baseline)")
        print("-" * 80)
        
        imp = comparison['improvements']
        print(f"{'Success Rate Delta (%)':<35} {imp['success_rate_delta']:>+18.1f}%")
        print(f"{'Time Speedup (×)':<35} {imp['time_speedup']:>18.2f}×")
        print(f"{'Path Length Improvement (%)':<35} {imp['path_length_improvement']:>+18.1f}%")
        print(f"{'Node Efficiency (Baseline/Adaptive)':<35} {imp['node_efficiency']:>18.2f}×")
        
        print("\n" + "=" * 80)
    
    def plot_results(self, output_dir: str = "./results"):
        """Generate comparison plots"""
        Path(output_dir).mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Success rate
        categories = ['Baseline', 'Adaptive']
        success_rates = [
            sum(1 for r in self.results_baseline if r.success) / len(self.results_baseline) * 100,
            sum(1 for r in self.results_adaptive if r.success) / len(self.results_adaptive) * 100
        ]
        axes[0, 0].bar(categories, success_rates, color=['#FF6B6B', '#4ECDC4'])
        axes[0, 0].set_ylabel('Success Rate (%)')
        axes[0, 0].set_title('Success Rate Comparison')
        axes[0, 0].set_ylim([0, 110])
        
        # Computation time (successful runs only)
        times_baseline = [r.computation_time for r in self.results_baseline if r.success]
        times_adaptive = [r.computation_time for r in self.results_adaptive if r.success]
        axes[0, 1].boxplot([times_baseline, times_adaptive], labels=categories)
        axes[0, 1].set_ylabel('Computation Time (s)')
        axes[0, 1].set_title('Planning Time Distribution')
        
        # Path length
        lengths_baseline = [r.path_length for r in self.results_baseline if r.success]
        lengths_adaptive = [r.path_length for r in self.results_adaptive if r.success]
        axes[1, 0].boxplot([lengths_baseline, lengths_adaptive], labels=categories)
        axes[1, 0].set_ylabel('Path Length (config-space)')
        axes[1, 0].set_title('Path Quality (Length)')
        
        # Node count
        nodes_baseline = [r.node_count for r in self.results_baseline]
        nodes_adaptive = [r.node_count for r in self.results_adaptive]
        axes[1, 1].boxplot([nodes_baseline, nodes_adaptive], labels=categories)
        axes[1, 1].set_ylabel('Tree Size (nodes)')
        axes[1, 1].set_title('Memory Efficiency')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison_plots.png", dpi=150)
        print(f"✓ Saved plots to {output_dir}/comparison_plots.png")
    
    def export_results(self, output_dir: str = "./results"):
        """Export results as JSON"""
        Path(output_dir).mkdir(exist_ok=True)
        
        data = {
            'baseline': [asdict(r) for r in self.results_baseline],
            'adaptive': [asdict(r) for r in self.results_adaptive],
            'comparison': self.generate_comparison_table()
        }
        
        with open(f"{output_dir}/benchmark_results.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✓ Saved results to {output_dir}/benchmark_results.json")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example:
    
    def planner_baseline():
        # Your baseline planner
        return path, tree, metrics
    
    def planner_adaptive():
        # Your adaptive planner
        return path, tree, metrics
    
    suite = BenchmarkSuite(n_trials=20)
    suite.run_benchmark(planner_baseline, planner_adaptive)
    suite.print_comparison_table()
    suite.plot_results()
    suite.export_results()
    """
    
    print("Benchmarking suite ready. Integrate with your planner.")