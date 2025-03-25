from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import nexu5_actor-critic as nac
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

#==================================================================================================================
def generate(state, action):


        # Path to your locally saved model
    model_path = "/home/matheesha/Desktop/codefest/codefest_2024/qwen2-transformers-0.5b-instruct-v1/"


    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    system_message = "You answer only the specific question"
    prompt = ()

            # Define stop tokens
    stop_tokens = ["<|assistant|>", "<|user|>", "</s>"]
    stop_token_ids = [tokenizer.encode(token)[0] for token in stop_tokens]

            # Format the input text as per the structure used in Llama.cpp
    input_text = f"<|system|>\n{system_message}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"

            # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt")

            # Generate text with stop token logic
    generated_ids = model.generate(
            inputs['input_ids'],
            max_length=180,
            eos_token_id=stop_token_ids[0],  # Default stop on first token in stop_tokens list
            no_repeat_ngram_size=2,  # To avoid repetitive answers
            pad_token_id=tokenizer.eos_token_id,  # Padding token (if applicable)
        )

            # Decode and print the generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    question = generated_text.replace("<|system|>You answer only the specific question</s><|user|>generate a ['normal']multiple choice science question</s><|assistant|>", "")
    queslist.append(question)
        #print(queslist)
    print(generated_text)
    return genereted_text
        # Generate a question
        
        
#==================================================================================================================






def train():
	from datasets import load_dataset
	from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


	dataset = load_dataset("text", data_files={"train": "/home/matheesha/Desktop/codefest/codefest_2024/text/book1.txt"})


	dataset = dataset["train"].train_test_split(test_size=0.2)  # 20% for validation
	train_dataset = dataset["train"]
	val_dataset = dataset["test"]


	model_path = "/home/matheesha/Desktop/codefest/codefest_2024/qwen2-transformers-0.5b-instruct-v1/"
	tokenizer = AutoTokenizer.from_pretrained(model_path)


	def tokenize_function(example):
		return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

	train_dataset = train_dataset.map(tokenize_function, batched=True)
	val_dataset = val_dataset.map(tokenize_function, batched=True)


	train_dataset = train_dataset.remove_columns(["text"])
	val_dataset = val_dataset.remove_columns(["text"])

	train_dataset.set_format("torch")
	val_dataset.set_format("torch")


	model = AutoModelForCausalLM.from_pretrained(model_path)


	training_args = TrainingArguments(
		output_dir="./results",              # Directory to save model checkpoints
		evaluation_strategy="epoch",         # Evaluate at the end of each epoch
		save_strategy="epoch",               # Save model at the end of each epoch (must match evaluation_strategy)
		learning_rate=5e-5,                  # Learning rate
		per_device_train_batch_size=4,       # Batch size for training
		per_device_eval_batch_size=4,        # Batch size for evaluation
		num_train_epochs=3,                  # Number of epochs
		weight_decay=0.01,                   # Regularization
		save_total_limit=2,                  # Number of checkpoints to keep
		logging_dir="./logs",                # Directory for logs
		load_best_model_at_end=True,         # Load the best model at the end of training
	)


	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=val_dataset,  # Use the validation set here
		tokenizer=tokenizer,       # Pass tokenizer to the trainer
	)

	# Start training
	trainer.train()

		

















