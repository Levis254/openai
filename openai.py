#in the terminal run  python.exe -m pip install --upgrade pip   to upgrade the pip
#install the datasets package
#install openai package version 0.28
#I used google colab notebook 

from openai import FineTuningJob, ChatCompletion
from datasets import load_dataset
from time import sleep
import random
import json
from google.colab import userdata
import openai


yahoo_answers_qa=load_dataset("yahoo_answers_qa", split="train")

#check features and columns and the total number of rows of the data
yahoo_answers_qa

sample_size=150

yahoo_answers_qa=yahoo_answers_qa.select(range(sample_size))

yahoo_answers_qa

def format_data(data):
    formatted_data=[{
            "messages": [
                {"role":"system","content":"You are a helpful assistant. Answer users' question with a polite tone"},
                {"role":"user","content": message["question"]},
                {"role":"assistant","content": message["answer"]}
            ]
        } for message in data 
    ]
    random.shuffle(formatted_data)
    return formatted_data

formatted_data=format_data(yahoo_answers_qa)

TRAIN_SIZE=int(len(formatted_data)*0.7)

training_data=formatted_data[:TRAIN_SIZE]
validation_data=formatted_data[TRAIN_SIZE:]

print(f"Training data size: {len(training_data)}")
print(f"Validation data size: {len(validation_data)}")


def save_data(dictionary_data, file_name):

  with open(file_name,"w") as outfile:

    for entry in dictionary_data:
      json.dump(entry, outfile)
      outfile.write("\n")


openai.api_key=userdata.get('####') #have to access the variable on google colab

def upload_fine_tuning_data(data_path):

    uploaded_file=openai.File.create(
        file=open(data_path),
        purpose="fine-tune"
    )
    return uploaded_file


upload_training_data=upload_fine_tuning_data("/content/ training_data.jsonl")
upload_validation_data=upload_fine_tuning_data("/content/ validation_data.jsonl")


upload_training_id=upload_training_data["id"]

upload_validation_id=upload_validation_data["id"]

def create_fine_tuning(base_model, train_id, val_id):

    fine_tuning_response=FineTuningJob.create(
        training_file=train_id,
        validation_file=val_id,
        model=base_model
    )

    return fine_tuning_response

base_model="gpt-3.5-turbo"

fine_tuning_response= create_fine_tuning(base_model,
                                        upload_training_id,
                                        upload_validation_id)



print(fine_tuning_response)

while True:

    fine_tuning_response=FineTuningJob.retrieve(fine_tuning_job_ID)
    fine_tuned_model_ID= fine_tuning_response["fine_tuned_model"]

    if(fine_tuned_model_ID !=None):
        print("fine-tuning completed!")
        print(f"Fine-tuned model ID: {fine_tuned_model_ID}")
        break

    else:
        print("Fine-tuning in progress...")
        sleep(200)

