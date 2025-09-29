"""
Visual Fidelity Ablation Analysis Tools

This module provides analysis and visualization tools for the VF ablation study results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AblationAnalyzer:
    """Analyzer for visual fidelity ablation study results"""
    
    def __init__(self, results_path="ablation_results.json"):
        self.results_path = results_path
        self.results = None
        self.load_results()
    
    def load_results(self):
        """Load ablation results from JSON file"""
        try:
            with open(self.results_path, 'r') as f:
                self.results = json.load(f)
            print(f"Loaded results from {self.results_path}")
        except FileNotFoundError:
            print(f"Results file {self.results_path} not found. Run the ablation study first.")
            self.results = {}
    
    def create_results_dataframe(self):
        """Convert results to pandas DataFrame for easier analysis"""
        rows = []
        
        if "summary" not in self.results:
            print("No summary results found.")
            return pd.DataFrame()
        
        for dataset in self.results["summary"]:
            for vlm in self.results["summary"][dataset]:
                for config, score in self.results["summary"][dataset][vlm].items():
                    qgen_model, verif_model = config.split('_')
                    rows.append({
                        'dataset': dataset,
                        'target_vlm': vlm,
                        'qgen_model': qgen_model,
                        'verif_model': verif_model,
                        'config': config,
                        'vf_score': score
                    })
        
        return pd.DataFrame(rows)
    
    def plot_heatmap(self, df, save_path="ablation_heatmap.png"):
        """Create heatmap showing VF scores across configurations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Visual Fidelity Scores Across Ablation Configurations', fontsize=16)
        
        datasets = df['dataset'].unique()
        vlms = df['target_vlm'].unique()
        
        for i, dataset in enumerate(datasets):
            for j, vlm in enumerate(vlms):
                ax = axes[i, j]
                
                # Filter data for this combination
                subset = df[(df['dataset'] == dataset) & (df['target_vlm'] == vlm)]
                
                if len(subset) > 0:
                    # Create pivot table for heatmap
                    pivot = subset.pivot(index='qgen_model', columns='verif_model', values='vf_score')
                    
                    # Create heatmap
                    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                              center=pivot.mean().mean(), ax=ax, cbar_kws={'label': 'VF Score'})
                    
                    ax.set_title(f'{dataset} - {vlm}')
                    ax.set_xlabel('Verification Model')
                    ax.set_ylabel('Question Generation Model')
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{dataset} - {vlm}')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Heatmap saved to {save_path}")
    
    def plot_component_effects(self, save_path="component_effects.png"):
        """Plot individual component effects"""
        if "component_effects" not in self.results:
            print("No component effects data found.")
            return
        
        effects_data = []
        for key, effects in self.results["component_effects"].items():
            dataset, vlm = key.split('_', 1)
            effects_data.append({
                'dataset_vlm': key,
                'dataset': dataset,
                'vlm': vlm,
                'QGen Effect': effects['qgen_effect'],
                'Verif Effect': effects['verif_effect'],
                'Both Effect': effects['both_effect'],
                'Interaction': effects['interaction']
            })
        
        df_effects = pd.DataFrame(effects_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Component Effects on Visual Fidelity Scores', fontsize=16)
        
        effects = ['QGen Effect', 'Verif Effect', 'Both Effect', 'Interaction']
        
        for i, effect in enumerate(effects):
            ax = axes[i // 2, i % 2]
            
            # Create bar plot
            df_effects.plot(x='dataset_vlm', y=effect, kind='bar', ax=ax, 
                          color=['skyblue' if x >= 0 else 'salmon' for x in df_effects[effect]])
            
            ax.set_title(f'{effect}')
            ax.set_xlabel('Dataset - Target VLM')
            ax.set_ylabel('Score Change')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Component effects plot saved to {save_path}")
    
    def statistical_analysis(self):
        """Perform statistical analysis on ablation results"""
        df = self.create_results_dataframe()
        
        if df.empty:
            print("No data available for statistical analysis.")
            return
        
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
        
        # Overall statistics
        print(f"\nOverall VF Score Statistics:")
        print(f"Mean: {df['vf_score'].mean():.3f}")
        print(f"Std:  {df['vf_score'].std():.3f}")
        print(f"Min:  {df['vf_score'].min():.3f}")
        print(f"Max:  {df['vf_score'].max():.3f}")
        
        # By dataset
        print(f"\nBy Dataset:")
        for dataset in df['dataset'].unique():
            subset = df[df['dataset'] == dataset]
            print(f"{dataset}: Mean={subset['vf_score'].mean():.3f}, Std={subset['vf_score'].std():.3f}")
        
        # By target VLM
        print(f"\nBy Target VLM:")
        for vlm in df['target_vlm'].unique():
            subset = df[df['target_vlm'] == vlm]
            print(f"{vlm}: Mean={subset['vf_score'].mean():.3f}, Std={subset['vf_score'].std():.3f}")
        
        # By QGen model
        print(f"\nBy Question Generation Model:")
        for model in df['qgen_model'].unique():
            subset = df[df['qgen_model'] == model]
            print(f"{model}: Mean={subset['vf_score'].mean():.3f}, Std={subset['vf_score'].std():.3f}")
        
        # By Verification model
        print(f"\nBy Verification Model:")
        for model in df['verif_model'].unique():
            subset = df[df['verif_model'] == model]
            print(f"{model}: Mean={subset['vf_score'].mean():.3f}, Std={subset['vf_score'].std():.3f}")
        
        return df
    
    def generate_report(self, output_path="ablation_report.md"):
        """Generate a comprehensive markdown report"""
        df = self.create_results_dataframe()
        
        if df.empty:
            print("No data available for report generation.")
            return
        
        report = []
        report.append("# Visual Fidelity Ablation Study Report\n")
        report.append("## Executive Summary\n")
        
        # Key findings
        best_config = df.loc[df['vf_score'].idxmax()]
        worst_config = df.loc[df['vf_score'].idxmin()]
        
        report.append(f"- **Best Configuration**: {best_config['config']} on {best_config['dataset']} with {best_config['target_vlm']} (Score: {best_config['vf_score']:.3f})")
        report.append(f"- **Worst Configuration**: {worst_config['config']} on {worst_config['dataset']} with {worst_config['target_vlm']} (Score: {worst_config['vf_score']:.3f})")
        report.append(f"- **Score Range**: {df['vf_score'].min():.3f} - {df['vf_score'].max():.3f}")
        report.append(f"- **Overall Mean**: {df['vf_score'].mean():.3f} ± {df['vf_score'].std():.3f}\n")
        
        # Component effects analysis
        if "component_effects" in self.results:
            report.append("## Component Effects Analysis\n")
            
            for key, effects in self.results["component_effects"].items():
                dataset, vlm = key.split('_', 1)
                report.append(f"### {dataset} - {vlm}\n")
                report.append(f"- **QGen Effect**: {effects['qgen_effect']:+.3f} (changing from GPT-4o to Qwen2.5-VL)")
                report.append(f"- **Verif Effect**: {effects['verif_effect']:+.3f} (changing from GPT-4o to Qwen2.5-VL)")
                report.append(f"- **Both Effect**: {effects['both_effect']:+.3f} (changing both components)")
                report.append(f"- **Interaction**: {effects['interaction']:+.3f} (synergy between components)\n")
        
        # Detailed results table
        report.append("## Detailed Results\n")
        report.append("| Dataset | Target VLM | QGen | Verif | Config | VF Score |")
        report.append("|---------|------------|------|-------|--------|----------|")
        
        for _, row in df.iterrows():
            report.append(f"| {row['dataset']} | {row['target_vlm']} | {row['qgen_model']} | {row['verif_model']} | {row['config']} | {row['vf_score']:.3f} |")
        
        report.append("\n## Methodology\n")
        report.append("- **Fixed Target VLMs**: Qwen2.5-VL-7B and LLaVA-1.5-7B (explanations being evaluated)")
        report.append("- **Variable Components**:")
        report.append("  - m_QGen (Question Generation): GPT-4o, Qwen2.5-VL-7B-Instruct, Gemma-3n-E4B")
        report.append("  - m_Verif (Verification): GPT-4o, Qwen2.5-VL-7B-Instruct, Gemma-3n-E4B")
        report.append("- **Datasets**: A-OKVQA and VizWiz (500 items each)")
        report.append("- **Metric**: Visual Fidelity Score (proportion of verification questions answered 'yes')")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Report saved to {output_path}")
        return '\n'.join(report)


def main():
    """Main analysis function"""
    analyzer = AblationAnalyzer()
    
    if not analyzer.results:
        print("No results found. Please run the ablation study first.")
        return
    
    # Create DataFrame
    df = analyzer.create_results_dataframe()
    print(f"Loaded {len(df)} result records")
    
    # Statistical analysis
    analyzer.statistical_analysis()
    
    # Generate visualizations
    analyzer.plot_heatmap(df)
    analyzer.plot_component_effects()
    
    # Generate report
    analyzer.generate_report()
    
    print("\nAnalysis complete! Check the generated files:")
    print("- ablation_heatmap.png")
    print("- component_effects.png") 
    print("- ablation_report.md")


if __name__ == "__main__":
    main()