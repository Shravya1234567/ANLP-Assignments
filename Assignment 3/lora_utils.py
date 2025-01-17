from peft import LoraConfig, get_peft_model

def load_lora_model(model, lora_r, lora_alpha, lora_dropout):
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias = "none",
        task_type="CAUSAL_LM",  
        target_modules=["c_attn"]
    )

    model = get_peft_model(model, config)
    return model