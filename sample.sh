PYTORCH_ENABLE_MPS_FALLBACK=1 python sample.py --model_path ../hart-0.7b-1024px/llm \
    --text_model_path ../Qwen2-VL-1.5B-Instruct \
    --prompt "MIT PhD student crying" \
    --sample_folder_dir ./img/ \
    --shield_model_path ../shieldgemma-2b \
