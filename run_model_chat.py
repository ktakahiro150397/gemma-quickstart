import os
from transformers import AutoTokenizer, pipeline
import torch
import json
from logging import config, getLogger

# 現在のスクリプトファイルの絶対パスを取得する
script_dir = os.path.dirname(os.path.abspath(__file__))

# 設定ファイルのパスを作成する
settings_file_path = os.path.join(script_dir, "logsettings.json")

logger = getLogger(__name__)

# ログ設定読込
with open(settings_file_path) as f:
    config.dictConfig(json.load(f))

def setup_pipeline(model,device):
    torch_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model,
                                              attn_implementation="flash_attention_2",
                                              )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch_dtype,},
        device=device,
    )

def main():
    # model = "google/gemma-7b-it"
    model = "google/gemma-2b-it"
    device = "cuda"

    chat_pipeline = setup_pipeline(model,device)

    chat = []
    while(True):
        input_text = input("input: ")

        if(input_text == "quit"):
            break

        logger.info(f"input: {input_text}")

        # Add user input to chat history
        chat.append({"role":"user","content":input_text})

        prompt = chat_pipeline.tokenizer.apply_chat_template(chat,tokenize=False,add_generation_prompt=True)

        outputs = chat_pipeline(prompt,
                                max_new_tokens=2048,
                                add_special_tokens=True,
                                do_sample=True,
                                temperature=0.7,
                                top_k=5,
                                top_p=0.95)
        output_text = outputs[0]["generated_text"][len(prompt):]
        logger.info(f"gemma: {output_text}")

        chat.append({"role":"model","content":output_text})

if __name__ == "__main__":
    main()
