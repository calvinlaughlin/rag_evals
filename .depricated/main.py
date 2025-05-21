"""
Main module for PDF ingestion strategy comparison

This script demonstrates the implementation and evaluation of different
PDF ingestion strategies.
"""

import json
import argparse
import os

from ingestion_strategies import get_all_strategies, process_document
from pdf_utils import load_document

def main():
    parser = argparse.ArgumentParser(
        description="Compare PDF ingestion strategies on a single document."
    )
    parser.add_argument(
        "document",
        nargs="?",
        default="sample_document.txt",
        help="Path to the document (TXT or PDF). Defaults to sample_document.txt.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.document):
        raise FileNotFoundError(f"Document not found: {args.document}")

    # Load document content
    document_text = load_document(args.document)
    
    # Process document with different strategies
    strategies = get_all_strategies()
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
    with open('results.json', 'w') as f:
        # Convert to serializable format
        serializable_results = {}
        for strategy_name, chunks in results.items():
            serializable_results[strategy_name] = []
            for chunk in chunks:
                serializable_chunk = {k: v for k, v in chunk.items()}
                serializable_results[strategy_name].append(serializable_chunk)
        
        json.dump(serializable_results, f, indent=2)
    
    print("\nResults saved to results.json")

if __name__ == "__main__":
    main()
