#!/usr/bin/env python3
"""
Main module for PDF ingestion strategy comparison

This script demonstrates the implementation and evaluation of different
PDF ingestion strategies.
"""

import json
import argparse
import os
import sys

# Add parent directory to path so we can import from the pdf_ingestion package
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pdf_ingestion import load_document, get_all_strategies, process_document, process_pdf, is_pdf


def main():
    parser = argparse.ArgumentParser(
        description="Compare PDF ingestion strategies on a single document."
    )
    parser.add_argument(
        "document",
        nargs="?",
        default="../sample_document.txt",
        help="Path to the document (TXT or PDF). Defaults to sample_document.txt.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.document):
        raise FileNotFoundError(f"Document not found: {args.document}")

    strategies = get_all_strategies()

    if is_pdf(args.document):
        results = process_pdf(args.document, strategies)
    else:
        # Load plain-text content then chunk
        document_text = load_document(args.document)
        results = process_document(document_text, strategies)
    
    # Print summary of chunks produced by each strategy
    print("\nChunking Summary:")
    print("-" * 50)
    for strategy_name, chunks in results.items():
        print(f"{strategy_name}: {len(chunks)} chunks")
        avg_chunk_length = sum(len(chunk['content']) for chunk in chunks) / len(chunks)
        print(f"  Average chunk length: {avg_chunk_length:.2f} characters")
        
        # Print first chunk as example
        if chunks:
            print("\nExample chunk:")
            content = chunks[0]['content']
            print(f"  {content[:200]}..." if len(content) > 200 else content)
        
        print("-" * 50)
    
    # Save results to file
    output_file = '../results.json'
    with open(output_file, 'w') as f:
        # Convert to serializable format
        serializable_results = {}
        for strategy_name, chunks in results.items():
            serializable_results[strategy_name] = []
            for chunk in chunks:
                serializable_chunk = {k: v for k, v in chunk.items()}
                serializable_results[strategy_name].append(serializable_chunk)
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main() 