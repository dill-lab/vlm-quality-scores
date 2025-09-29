#!/usr/bin/env python3
"""
Visual Fidelity Ablation Study Runner

Usage:
    python run_ablation.py --mode core     # Run 4 core configurations
    python run_ablation.py --mode full     # Run all 9 configurations  
    python run_ablation.py --mode analyze  # Analyze existing results
    python run_ablation.py --mode all      # Run core + analysis
"""

import argparse
import sys
from Visual_Fidelity_ablation import VFAblationStudy
from ablation_analysis import AblationAnalyzer


def run_core_ablation():
    """Run the 4 core ablation configurations"""
    print("="*60)
    print("RUNNING CORE ABLATION STUDY")
    print("="*60)
    print("Configurations to test:")
    print("1. (GPT-4o, GPT-4o) - Baseline")
    print("2. (GPT-4o, Qwen2.5-VL) - Swap verifier only")
    print("3. (Qwen2.5-VL, GPT-4o) - Swap generator only") 
    print("4. (Qwen2.5-VL, Qwen2.5-VL) - Both swapped")
    print()
    
    ablation_study = VFAblationStudy()
    results = ablation_study.run_core_ablation(save_results=True)
    
    # Print summary
    ablation_study.print_summary_table(results)
    
    # Analyze results
    analysis = ablation_study.analyze_results(results, save_analysis=True)
    
    print("\nCore ablation study completed!")
    print("Results saved to CSV files and ablation_results.json")
    
    return results


def run_full_ablation():
    """Run all 9 ablation configurations"""
    print("="*60)
    print("RUNNING FULL ABLATION STUDY")
    print("="*60)
    print("Testing all combinations of:")
    print("- QGen Models: GPT-4o, Qwen2.5-VL-7B-Instruct, Gemma-3n-E4B")
    print("- Verif Models: GPT-4o, Qwen2.5-VL-7B-Instruct, Gemma-3n-E4B")
    print("- Total: 9 configurations")
    print()
    
    ablation_study = VFAblationStudy()
    results = ablation_study.run_full_ablation(save_results=True)
    
    # Print summary
    ablation_study.print_summary_table(results)
    
    # Analyze results
    analysis = ablation_study.analyze_results(results, save_analysis=True)
    
    print("\nFull ablation study completed!")
    print("Results saved to CSV files and ablation_results.json")
    
    return results


def run_analysis():
    """Analyze existing ablation results"""
    print("="*60)
    print("ANALYZING ABLATION RESULTS")
    print("="*60)
    
    analyzer = AblationAnalyzer()
    
    if not analyzer.results:
        print("No results found. Please run the ablation study first.")
        return
    
    # Statistical analysis
    df = analyzer.statistical_analysis()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_heatmap(df)
    analyzer.plot_component_effects()
    
    # Generate report
    print("\nGenerating report...")
    analyzer.generate_report()
    
    print("\nAnalysis completed!")


def main():
    parser = argparse.ArgumentParser(description="Visual Fidelity Ablation Study Runner")
    parser.add_argument('--mode', choices=['core', 'full', 'analyze', 'all'], 
                       default='core', help='Mode to run')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'core':
            run_core_ablation()
            
        elif args.mode == 'full':
            run_full_ablation()
            
        elif args.mode == 'analyze':
            run_analysis()
            
        elif args.mode == 'all':
            print("Running core ablation followed by analysis...\n")
            run_core_ablation()
            print("\n" + "="*60)
            print("PROCEEDING TO ANALYSIS")
            print("="*60)
            run_analysis()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()