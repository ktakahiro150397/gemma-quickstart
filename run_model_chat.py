from transformers import AutoTokenizer, pipeline
import torch


def setup_pipeline(model,device):
    torch_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model)
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch_dtype,},
        device=device,
    )

def main():
    # model = "gg-hf/gemma-7b-it"
    model = "google/gemma-2b-it"
    device = "cuda"

    chat_pipeline = setup_pipeline(model,device)

    chat = []
    while(True):
        print("------------------------")
        input_text = input("input: ")

        if(input_text == "quit"):
            break

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
        print(f"gemma: {output_text}")

        chat.append({"role":"model","content":output_text})

if __name__ == "__main__":
    main()


# dtype = torch.bfloat16

# tokenizer = AutoTokenizer.from_pretrained(model_id)

# # Requires 'pip install accelerate' for the following line
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="cuda",
#     torch_dtype=dtype,
# )



# while(True):
#     chat = []

#     input_text = input("input: ")

#     if(input_text == "quit"):
#         break

#     # Add user input to chat history
#     chat.append({"role":"user","content":input_text})

#     # chat = [
#     #     { "role":"user","content":"Please explain what WSL is and how it works."},
#     # ]

#     prompt = tokenizer.apply_chat_template(chat,tokenize=False,add_generation_prompt=True,return_tensors="pt")

#     inputs = tokenizer.encode(prompt, add_special_tokens=False,return_tensors="pt")
#     outputs = model.generate(input_ids=inputs.to(model.device),max_new_tokens=150)

#     output = tokenizer.decode(outputs[0, len(prompt):])
#     print(f"gemma: {output}")
#     chat.append({"role":"model","content":output})

#     # Add model output to chat history
#     #chat.append({"role":"model","content":output})



