#!/usr/bin/env python
"""
Example: Describe algorithm parameters.

This example shows how to use the describe_parameters() function to
view the tunable parameters for an algorithm without running any tuning.

Usage:
    python examples/tuning/describe_parameters.py
    python examples/tuning/describe_parameters.py --format json
    python examples/tuning/describe_parameters.py --format json --output params.json
"""

import argparse

from jmetal.tuning import describe_parameters, list_algorithms


def main():
    parser = argparse.ArgumentParser(
        description="Describe algorithm tunable parameters"
    )
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default="NSGAII",
        help="Algorithm name (default: NSGAII)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file (optional)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available algorithms"
    )
    args = parser.parse_args()
    
    # List available algorithms
    if args.list:
        print("Available algorithms for tuning:")
        for alg in list_algorithms():
            print(f"  - {alg}")
        return
    
    print("=" * 60)
    print(f"Tunable Parameters for {args.algorithm}")
    print("=" * 60)
    print()
    
    # Get parameter description
    description = describe_parameters(
        algorithm=args.algorithm,
        format=args.format,
        output_path=args.output,
    )
    
    if description:
        print(description)
    
    if args.output:
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
