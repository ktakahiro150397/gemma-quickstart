# https://huggingface.co/google/gemma-7b-it#usage
# Running the model on a single / multi GPU

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it",device_map="auto")

input_text = "Tell me about WSL. Please explain."
input_ids = tokenizer(input_text,return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids,max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
