"""
Simple evaluation of PDF ingestion strategies
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def load_results(file_path):
    """Load results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_results(results):
    """Analyze chunking results"""
    analysis = {}
    
    for strategy_name, chunks in results.items():
        # Calculate metrics
        num_chunks = len(chunks)
        avg_length = sum(len(chunk['content']) for chunk in chunks) / num_chunks if num_chunks > 0 else 0
        
        # Calculate content overlap (simple approximation)
        content_overlap = 0
        if num_chunks > 1:
            for i in range(num_chunks - 1):
                content1 = chunks[i]['content'].lower()
                content2 = chunks[i+1]['content'].lower()
                
                # Count words in common
                words1 = set(content1.split())
                words2 = set(content2.split())
                common_words = words1.intersection(words2)
                
                # Calculate overlap ratio
                overlap_ratio = len(common_words) / min(len(words1), len(words2))
                content_overlap += overlap_ratio
            
            content_overlap /= (num_chunks - 1)  # Average overlap
        
        # Store analysis
        analysis[strategy_name] = {
            'num_chunks': num_chunks,
            'avg_length': avg_length,
            'content_overlap': content_overlap
        }
    
    return analysis

def plot_results(analysis):
    """Plot analysis results"""
    strategies = list(analysis.keys())
    num_chunks = [analysis[s]['num_chunks'] for s in strategies]
    avg_lengths = [analysis[s]['avg_length'] / 500 for s in strategies]  # Scale down for visualization
    overlaps = [analysis[s]['content_overlap'] * 10 for s in strategies]  # Scale up for visualization
    
    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set width of bars
    bar_width = 0.25
    
    # Set positions of bars on X axis
    r1 = np.arange(len(strategies))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create bars
    ax.bar(r1, num_chunks, width=bar_width, label='Number of Chunks')
    ax.bar(r2, avg_lengths, width=bar_width, label='Avg Length (÷500 chars)')
    ax.bar(r3, overlaps, width=bar_width, label='Content Overlap (×10)')
    
    # Add labels and title
    ax.set_xlabel('Chunking Strategy')
    ax.set_ylabel('Value')
    ax.set_title('Comparison of PDF Ingestion Strategies')
    ax.set_xticks([r + bar_width for r in range(len(strategies))])
    ax.set_xticklabels([s.split('(')[0].strip() for s in strategies])
    
    # Add legend
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig('strategy_comparison.png')
    print("Plot saved as strategy_comparison.png")

def main():
    # Load results
    results = load_results('results.json')
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print analysis
    print("\nAnalysis Results:")
    print("-" * 50)
    for strategy, metrics in analysis.items():
        print(f"{strategy}:")
        print(f"  Number of chunks: {metrics['num_chunks']}")
        print(f"  Average chunk length: {metrics['avg_length']:.2f} characters")
        print(f"  Content overlap between chunks: {metrics['content_overlap']:.2f}")
        print("-" * 50)
    
    # Create summary table
    print("\nStrategy Comparison Summary:")
    print("-" * 80)
    print(f"{'Strategy':<40} | {'Chunks':<8} | {'Avg Length':<12} | {'Overlap':<8}")
    print("-" * 80)
    for strategy, metrics in analysis.items():
        strategy_name = strategy.split('(')[0].strip()
        print(f"{strategy_name:<40} | {metrics['num_chunks']:<8} | {metrics['avg_length']:<12.2f} | {metrics['content_overlap']:<8.2f}")
    
    # Plot results
    try:
        import matplotlib
        plot_results(analysis)
    except ImportError:
        print("Matplotlib not available. Skipping plot generation.")

if __name__ == "__main__":
    main()
