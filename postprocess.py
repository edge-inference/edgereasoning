#!/usr/bin/env python3
"""
Main Results Processing Script

Processes MMLU evaluation results 
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

from processing import ResultConsolidator, PerformanceAnalyzer, ReportGenerator


def list_available_datasets() -> list:
    return []


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'processor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def main():
    """Main processing pipeline."""
    available_datasets = list_available_datasets()
    
    parser = argparse.ArgumentParser(
        description='Process MMLU evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Process synthetic prefill using files/results.yaml
  python postprocess.py --sub-config prefill
  
  # Process MMLU server results using files/results.yaml
  python postprocess.py --sub-config server
  
  # Override paths manually
  python postprocess.py --results-dir ./custom_dir --output-dir ./custom_output
  
  # Handle container-created directories (auto-fix permissions)
  python postprocess.py --sub-config prefill --fix-permissions
  
  # Run analysis after consolidation
  python postprocess.py --sub-config prefill --analysis
        """
    )
    
    parser.add_argument(
        '--sub-config', '-s',
        help='Sub-configuration from files/results.yaml (prefill|decode|scaling|server|tegra)'
    )
    
    parser.add_argument(
        '--results-dir', 
        help='Directory containing raw data (overrides dataset config)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory for processed results (overrides dataset config)'
    )
    
    parser.add_argument(
        '--consolidate-only',
        action='store_true',
        help='Only perform CSV consolidation, skip analysis and reports'
    )
    
    parser.add_argument(
        '--analysis-only',
        action='store_true',
        help='Only perform analysis (requires existing consolidated results)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation (faster processing)'
    )

    parser.add_argument(
        '--analysis', '-a',
        action='store_true',
        help='Perform performance analysis and generate plots/reports (default: skipped)'
    )
    
    parser.add_argument(
        '--fix-permissions',
        action='store_true',
        help='Attempt to fix directory permissions (requires sudo access)'
    )
    
    args = parser.parse_args()
    
    # Load from files/results.yaml
    def load_from_results_yaml(sub_key: str) -> dict:
        cfg_path = Path(__file__).parent / 'files' / 'results.yaml'
        if not cfg_path.exists():
            raise FileNotFoundError(f"Results configuration not found: {cfg_path}")
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        synthetic_keys = {'prefill', 'decode', 'scaling'}
        if sub_key in synthetic_keys:
            node = cfg['results']['synthetic'][sub_key]
            return {
                'name': f'synthetic::{sub_key}',
                'description': f'Synthetic {sub_key} dataset',
                'input_dir': node['input_dir'],
                'output_dir': node['output_dir'],
            }
        if sub_key in {'server', 'tegra'}:
            node = cfg['results']['mmlu'][sub_key]
            return {
                'name': f'mmlu::{sub_key}',
                'description': f'MMLU {sub_key} dataset',
                'input_dir': node['input_dir'],
                'output_dir': node['output_dir'],
            }
        raise KeyError(f"Unsupported --sub-config '{sub_key}' for --results. Supported: prefill, decode, scaling, server, tegra")

    config = None
    if args.sub_config:
        try:
            config = load_from_results_yaml(args.sub_config)
        except (FileNotFoundError, KeyError) as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    
    if args.results_dir and args.output_dir:
        results_dir_str = args.results_dir
        output_dir_str = args.output_dir
        dataset_name = "manual"
    elif config:
        results_dir_str = config['input_dir']
        output_dir_str = config['output_dir']
        dataset_name = config['name']
    else:
        print("ERROR: Either specify --sub-config or provide both --results-dir and --output-dir")
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate directories
    results_dir = Path(results_dir_str)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)
    
    output_dir = Path(output_dir_str)
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"\nWARNING: Permission denied creating directory: {output_dir}")
        print("This likely happened because the directory was created by a container (root).")
        
        if args.fix_permissions:
            print("\nAttempting to fix permissions...")
            import subprocess
            try:
                data_dir = Path("data")
                if data_dir.exists():
                    cmd = f"sudo chown -R $USER:$USER {data_dir}"
                    subprocess.run(cmd, shell=True, check=True)
                    print("Permissions fixed! Retrying directory creation...")
                    output_dir.mkdir(parents=True, exist_ok=True)
                else:
                    print("Data directory not found, creating output directory...")
                    output_dir.mkdir(parents=True, exist_ok=True)
            except subprocess.CalledProcessError:
                print("Failed to fix permissions. Falling back to alternative directory.")
                alt_output = Path("outputs") / "postprocess" / datetime.now().strftime("%Y%m%d_%H%M%S") / output_dir.name
                output_dir = alt_output
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"Using alternative output directory: {output_dir}")
        else:
            print("\nOptions:")
            print("1. Fix permissions: sudo chown -R $USER:$USER data/")
            print("2. Use alternative output directory")
            print("3. Run with --fix-permissions flag")
            
            alt_output = Path("outputs") / "postprocess" / datetime.now().strftime("%Y%m%d_%H%M%S") / output_dir.name
            response = input(f"\nUse alternative output directory? [{alt_output}] (y/n): ").strip().lower()
            
            if response in ['y', 'yes', '']:
                output_dir = alt_output
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"Using alternative output directory: {output_dir}")
            else:
                print("Please fix permissions or choose a different output directory.")
                sys.exit(1)
    
    print("MMLU Results Processor")
    print("="*50)
    print(f"Dataset: {dataset_name}")
    if config:
        print(f"Description: {config.get('description', 'N/A')}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Processing mode: {'Consolidate only' if args.consolidate_only else 'Analysis only' if args.analysis_only else 'Full pipeline'}")
    print("="*50)
    
    try:
        if not args.analysis_only:
            # Step 1: Consolidate results
            logger.info("Starting result consolidation...")
            print("\nStep 1: Consolidating Results")
            
            consolidator = ResultConsolidator(str(results_dir))
            consolidated_results = consolidator.process_all(str(output_dir))
            
            if not consolidated_results.models:
                logger.error("No models found to process. Check your results directory structure.")
                print("ERROR: No models found. Check your results directory structure.")
                sys.exit(1)
            
            print(f"Consolidated {len(consolidated_results.models)} models")
            
            if args.consolidate_only:
                print("\nConsolidation complete!")
                sys.exit(0)
        
        else:
            logger.info("Analysis-only mode: looking for existing consolidated results...")
            print("Analysis-only mode requires existing consolidated result")
            sys.exit(1)
        
        if not args.consolidate_only and args.analysis:
            logger.info("Starting performance analysis...")
            print("\nStep 2: Performance Analysis")
            
            analyzer = PerformanceAnalyzer(consolidated_results)
            
            analysis_dir = output_dir / "analysis"
            analyzer.generate_analysis_report(str(analysis_dir))
            
            print("Performance analysis complete")
            
            logger.info("Starting report generation...")
            print("\nStep 3: Report Generation")
            
            report_generator = ReportGenerator(consolidated_results)
            
            reports_dir = output_dir / "reports"
            report_generator.generate_all_reports(str(reports_dir))
            
            print("Report generation complete")
        
        # Summary
        print("\nProcessing Complete!")
        print("="*50)
        print(f"All outputs saved to: {output_dir}")
        print("\nGenerated files:")
        print("  Consolidated CSVs and Excel files")
        if not args.consolidate_only and args.analysis:
            print("  Performance analysis and plots")
            print("  Executive summary and detailed reports")
        print("\nCheck the output directory for all generated files.")
        
    except KeyboardInterrupt:
        logger.info("you interrupted")
        print("\nInterrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        print(f"\nProcessing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
