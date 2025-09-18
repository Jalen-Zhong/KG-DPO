#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Graph Visualization Tool

This script visualizes GraphML files using NetworkX and PyVis to create interactive HTML visualizations
of knowledge graphs. It supports custom styling and physics configuration for optimal graph layout.

Usage examples:
    python kg_visual.py --input output/data/1758091268/graph.graphml --output output/data/1758091268/graph.html
    python kg_visual.py -i path/to/input.graphml -o path/to/output.html --height 900px --width 1200px
    python kg_visual.py --input kg_data.graphml --output visualization.html --bgcolor "#f0f0f0" --fontcolor "#333333"
"""

import argparse
import networkx as nx
from pyvis.network import Network


def visualize_graphml(input_file: str, output_file: str, height: str = "750px", width: str = "100%", 
                     bgcolor: str = "#ffffff", font_color: str = "#000000") -> None:
    """
    Visualize a GraphML file and save as interactive HTML.
    
    Args:
        input_file (str): Path to the input GraphML file
        output_file (str): Path to the output HTML file
        height (str): Height of the visualization canvas
        width (str): Width of the visualization canvas
        bgcolor (str): Background color of the visualization
        font_color (str): Font color for node labels
    """
    # Load the graph from GraphML file
    G = nx.read_graphml(input_file)

    # Create PyVis network
    net = Network(
        notebook=False,
        height=height,
        width=width,
        bgcolor=bgcolor,
        font_color=font_color
    )

    # Add nodes with descriptions as tooltips
    for node in G.nodes:
        node_data = G.nodes[node]
        description = node_data.get("description", "")
        title = f"{description}"
        net.add_node(node, label=node, title=title)

    # Add edges with descriptions as tooltips
    for u, v in G.edges():
        edge_data = G.get_edge_data(u, v)
        description = edge_data.get("description", "")
        title = f"{description}"
        net.add_edge(u, v, title=title)

    # Configure physics and styling options
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "springLength": 100
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      },
      "edges": {
        "color": {
          "inherit": true
        }
      }
    }
    """)

    # Save the visualization
    net.save_graph(output_file)
    print(f"✅ Visualization saved to: {output_file}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Knowledge Graph Visualization Tool - Convert GraphML files to interactive HTML visualizations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input GraphML file path'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output HTML file path'
    )
    
    parser.add_argument(
        '--height',
        type=str,
        default='750px',
        help='Height of the visualization canvas'
    )
    
    parser.add_argument(
        '--width',
        type=str,
        default='100%',
        help='Width of the visualization canvas'
    )
    
    parser.add_argument(
        '--bgcolor',
        type=str,
        default='#ffffff',
        help='Background color of the visualization'
    )
    
    parser.add_argument(
        '--fontcolor',
        type=str,
        default='#000000',
        help='Font color for node labels'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = parse_arguments()
    
    # Validate input file exists
    import os
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file does not exist: {args.input}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        visualize_graphml(
            input_file=args.input,
            output_file=args.output,
            height=args.height,
            width=args.width,
            bgcolor=args.bgcolor,
            font_color=args.fontcolor
        )
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        raise


if __name__ == "__main__":
    main()