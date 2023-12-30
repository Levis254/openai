#in the terminal run  python.exe -m pip install --upgrade pip   to upgrade the pip
#install the datasets package
#install openai package version 0.28

from openai import FineTuningJob, ChatCompletion
from datasets import load_dataset
from time import sleep
import random
import json

#Data loading

yahoo_answers_qa=load_dataset("yahoo_answers_qa", split='train')
