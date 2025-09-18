#!/bin/bash
# Knowledge Graph Visualization Script
# Usage: ./kg-visual.sh [input_file] [output_file] [options]

# Default values
INPUT_FILE="output/data/1758091268/graph.graphml"
OUTPUT_FILE="output/data/1758091268/graph.html"
HEIGHT="750px"
WIDTH="100%"
BGCOLOR="#ffffff"
FONTCOLOR="#000000"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --bgcolor)
            BGCOLOR="$2"
            shift 2
            ;;
        --fontcolor)
            FONTCOLOR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Knowledge Graph Visualization Script"
            echo ""
            echo "Usage: ./kg-visual.sh [options]"
            echo ""
            echo "Options:"
            echo "  -i, --input FILE     Input GraphML file (default: $INPUT_FILE)"
            echo "  -o, --output FILE    Output HTML file (default: $OUTPUT_FILE)"
            echo "  --height HEIGHT      Canvas height (default: $HEIGHT)"
            echo "  --width WIDTH        Canvas width (default: $WIDTH)"
            echo "  --bgcolor COLOR      Background color (default: $BGCOLOR)"
            echo "  --fontcolor COLOR    Font color (default: $FONTCOLOR)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./kg-visual.sh -i kg_data.graphml -o visualization.html"
            echo "  ./kg-visual.sh --input output/graph.graphml --output output/visualization.html --height 900px"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Run the visualization script
echo "🚀 Starting knowledge graph visualization..."
echo "   Input: $INPUT_FILE"
echo "   Output: $OUTPUT_FILE"

python ./visual/kg_visual.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --bgcolor "$BGCOLOR" \
    --fontcolor "$FONTCOLOR"

# Check if the command succeeded
if [ $? -eq 0 ]; then
    echo "✅ Visualization completed successfully!"
    echo "   Output file: $OUTPUT_FILE"
else
    echo "❌ Visualization failed!"
    exit 1
fi