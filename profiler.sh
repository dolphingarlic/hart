PYTORCH_ENABLE_MPS_FALLBACK=1 python -i profiler.py --model_path ../hart-0.7b-1024px/llm \
    --text_model_path ../Qwen2-VL-1.5B-Instruct \
    --prompt "Glass spheres but they're not spheres because you messed up and they're avocadoes" \
    --sample_folder_dir ./img/ \
    --shield_model_path ../shieldgemma-2b \
