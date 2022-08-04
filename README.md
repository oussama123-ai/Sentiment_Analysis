# Sentiment Analysis with BERT :

## Abstract :
 
 Sentiment analysis has become very popular in both research and business due to the increasing amount of opinionated text from Internet users. Standard sentiment analysis deals with classifying the overall sentiment of a text, but this doesn’t include other important information such as towards which entity, topic or aspect within the text the sentiment is directed. Aspect-based sentiment analysis (ABSA) is a more complex task that consists in identifying both sentiments and aspects. This paper shows the potential of using the contextual word representations from the pre-trained language model BERT, together with a fine-tuning method with additional generated text, in order to solve out-of-domain ABSA and outperform previous state-of-the-art results on SemEval-2015 Task 12 subtask 2 and SemEval-2016 Task 5. To the best of our knowledge, no other existing work has been done on out-of-domain ABSA for aspect classification.
 
**Important Note:** 

FinBERT implementation relies on Hugging Face's `pytorch_pretrained_bert` library and their implementation of BERT for sequence classification tasks. `pytorch_pretrained_bert` is an earlier version of the [`transformers`]library. It is on the top of our priority to migrate the code for FinBERT to `transformers` in the near future.(https://aclanthology.org/W19-6120/)

# Installing :

 Install the dependencies by creating the Conda environment 'Sentiment Analysis' from the given `environment.yml` file and
 activating it.
```
conda env create -f environment.yml
conda activate AnaSen
```
# Models :

the workflow should be like this:
* Create a directory for the model. For example: `models/sentiment/<model directory name>`
* Download the model (https://prosus-public.s3-eu-west-1.amazonaws.com/finbert/language-model/pytorch_model.bin) and put it into the directory you just created.

* Put a copy of `config.json` in this same directory. 
* Call the model with `.from_pretrained(<model directory name>)`

# Datasets :

If you want to train the model , after downloading it, you should create three files under the `data/sentiment_data` folder as `train.csv`, `validation.csv`, `test.csv`.
 To create these files, do the following steps:
- Download the Financial PhraseBank from the above link.
- Get the path of `Sentences_50Agree.txt` file in the `FinancialPhraseBank-v1.0` zip.
- Run the [datasets script](scripts/datasets.py):
python scripts/datasets.py --data_path /home/oussama/Bureau/Sentiments_analyses/data/sentiment_data/Sentences_AllAgree.txt


# Predictions :

python scripts/predict.py --text_path test.txt --output_dir output/ --model_path models/classifier_model/finbert-sentiment

if you want you can change the command line try :

python predict.py but you should specified the text_path model_path in the code .

# Training :
 
Training is done in `Sentiment_Analysis.ipynb` notebook. The trained model will be saved to `models/classifier_model/finbert-sentiment`. You can find the training parameters in the notebook.

# Challenge :

we will work on facilitating the process of use soon , by programming a digital interface through which it is possible to obtain results for analyzing people's speech in a fast and sophisticated manner.



