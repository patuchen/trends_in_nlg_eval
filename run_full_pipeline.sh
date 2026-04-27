#!/bin/bash
# Full pipeline execution command for 85,792 papers
# Run this in a tmux session with SSH tunnel open in another terminal

set -e

cd /home/schmidtova/personal_work_ms/repos/trends_in_nlg_eval

echo "================================================================"
echo "NLG Evaluation Trends - Full Pipeline Execution"
echo "================================================================"
echo ""
echo "Prerequisites:"
echo "1. SSH tunnel open (in separate terminal):"
echo "   ssh -L 8000:localhost:8000 schmidtova@dll-4gpu3"
echo ""
echo "Starting full pipeline on 85,792 papers..."
echo "Expected duration: ~24-26 hours"
echo "================================================================"
echo ""

# Run pipeline with automatic stage skipping
# Stage 4 (interactive spot checks) is skipped for uninterrupted execution
python3 run_pipeline.py \
    --papers papers.csv \
    --txt-dir txt_papers \
    --output-dir output \
    --api-url http://localhost:8000/v1 \
    --skip-spot-checks \
    2>&1 | tee pipeline_full_output.log

echo ""
echo "================================================================"
echo "Pipeline complete! Results saved to output/"
echo "================================================================"
echo ""
echo "To manually review a sample of results after completion:"
echo "  python3 validate_stage4.py"
echo ""
