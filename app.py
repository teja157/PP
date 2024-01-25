import streamlit as st
import pandas as pd
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

def load_dataset(file_path):
    return pd.read_csv(file_path)

def get_answer(question, model, tokenizer):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    return answer

def main():
    st.title("BERT Chatbot with Streamlit and Dataset")

    # Load BERT model and tokenizer
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    model = BertForQuestionAnswering.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Load dataset
    dataset_path = "qa_dataset.csv"
    dataset = load_dataset(dataset_path)

    # User input
    user_input = st.text_input("Ask a question:")

    if user_input:
        # Check if the question is in the dataset
        answer = dataset[dataset['Question'].apply(lambda q: user_input.lower() in q.lower())]['Answer'].values

        if answer:
            st.write(f"Answer: {answer[0]}")
        else:
            # If the question is not in the dataset, use BERT model
            bert_answer = get_answer(user_input, model, tokenizer)
            st.write(f"BERT Answer: {bert_answer}")

if __name__ == "__main__":
    main()
