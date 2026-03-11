# Code files

1. Download the dataset files as needed.

We used the xSID files from release 0.6, but fix the partial misalignment between de and de-ba by editing de-ba.valid.conll to move sentences 151 and 152 so they become sentences 100 and 104 respectively: https://github.com/mainlp/xsid. 

Update filepaths at the top of map_data.py to match your download location, and then re-map the data (this creates TSV files in `../data/`):
```
python3 map_data.py
```

2. Get the model sizes:
```
python3 model_sizes.py > model_sizes.log
```

3. Hyperparameter experiments:

```
# MODEL_NAME: the Huggingface model name
# TASK: intent, topic
# AUDIO_PATH: the path to the folder containing xSID-audio/SwissDial
python3 train_text_model.py --model MODEL_NAME --lr LEARNING_RATE --max MAX_EPOCHS --batch BATCH_SIZE --seed SEED --task TASK
python3 train_speech_model.py --model MODEL_NAME --lr LEARNING_RATE --max MAX_EPOCHS --batch BATCH_SIZE --seed SEED --task TASK --audiodir AUDIO_PATH
```

3. ASR: transcribe the evaluation data. This requires a huggingface-cli login for the Speech-MASSIVE test data (you need to accept the usage conditions on https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test).

```
# MODEL_NAME: the Huggingface model name
# TASK: intent, topic
# AUDIO_PATH: the path to the folder containing xSID-audio/SwissDial
python3 asr.py --model MODEL_NAME --task TASK --audiodir AUDIO_PATH
```

This creates TSV files with the ASR predictions for the development and test sets in `data/{intents,topics}/asr/MODEL_NAME/`.

4. Evaluate the ASR hypotheses.

Get the subset of the MASSIVE German dataset that is fully parallel to the Bavarian translation (this creates new files in the predictions folder):
```
python3 match_massive_subsets_of_predictions.py
```

Replace hyphens with dashes in the model names in the following:
```
python3 evaluate_asr.py ../data/{intents,topics}/asr/MODEL_NAME
```
For each model, this creates two output files in `scores/asr/`: one with the mean sentence-level WER and CER per dataset, and one with the WER and CER for each sentence.

5. Train the classification models. We use the same scripts as for hyperparameter tuning, but now with the `--predict` flag. The hyperparameter values are the ones we chose based on hyperparameter tuning (for `--lr` and `--max`). **Repeat for multiple seeds!**

```
# Intent classification

## Text models (evaluates on gold text and ASR transcriptions)
python3 train_text_model.py --model microsoft/mdeberta-v3-base --lr 0.00001 --seed 1234 --predict --mode intent
python3 train_text_model.py --model google-bert/bert-base-multilingual-cased --lr 0.00005 --seed 1234 --predict --mode intent
python3 train_text_model.py --model FacebookAI/xlm-roberta-base --lr 0.00001 --seed 1234 --predict --mode intent
python3 train_text_model.py --model FacebookAI/xlm-roberta-large --lr 0.0001 --seed 1234 --predict --mode intent

## Speech models
python3 train_speech_model.py --model facebook/wav2vec2-xls-r-300m --lr 0.00005 --max 30 --seed 1234 --predict --mode intent --audiodir AUDIO_PATH
python3 train_speech_model.py --model AndrewMcDowell/wav2vec2-xls-r-300m-german-de --lr 0.0001 --max 23 --seed 1234 --predict --mode intent --audiodir AUDIO_PATH
python3 train_speech_model.py --model facebook/mms-300m --lr 0.00005 --max 29 --seed 1234 --predict --mode intent --audiodir AUDIO_PATH
python3 train_speech_model.py --model utter-project/mHuBERT-147 --lr 0.00005 --max 29 --seed 1234 --predict --mode intent --audiodir AUDIO_PATH
python3 train_speech_model.py --model openai/whisper-small --lr 0.00005 --max 15 --seed 1234 --predict --mode intent --audiodir AUDIO_PATH
python3 train_speech_model.py --model openai/whisper-medium --lr 0.00001 --max 17 --seed 1234 --predict --mode intent --audiodir AUDIO_PATH


# Topic classification

## Text models (evaluates on gold text and ASR transcriptions)
python3 train_text_model.py --model microsoft/mdeberta-v3-base --lr 0.0001 --seed 1234 --predict --mode topic
python3 train_text_model.py --model google-bert/bert-base-multilingual-cased --lr 0.00005 --seed 1234 --predict --mode topic
python3 train_text_model.py --model FacebookAI/xlm-roberta-base --lr 0.00005 --seed 1234 --predict --mode topic
python3 train_text_model.py --model FacebookAI/xlm-roberta-large --lr 0.00001 --seed 1234 --predict --mode topic

## Speech models
python3 train_speech_model.py --model openai/whisper-tiny --lr 0.0001 --max 5 --seed 1234 --predict --mode topic --audiodir AUDIO_PATH
python3 train_speech_model.py --model openai/whisper-base --lr 0.00005 --max 7 --seed 1234 --predict --mode topic --audiodir AUDIO_PATH
python3 train_speech_model.py --model openai/whisper-small --lr 0.00005 --max 13 --seed 1234 --predict --mode topic --audiodir AUDIO_PATH
python3 train_speech_model.py --model openai/whisper-medium --lr 0.00005 --max 5 --seed 1234 --predict --mode topic --audiodir AUDIO_PATH
python3 train_speech_model.py --model openai/whisper-large-v3 --lr 0.00001 --max 6 --seed 1234 --predict --mode topic --audiodir AUDIO_PATH
python3 train_speech_model.py --model facebook/wav2vec2-xls-r-300m --lr 0.00005 --max 29 --seed 1234 --predict --mode topic --audiodir AUDIO_PATH
python3 train_speech_model.py --model AndrewMcDowell/wav2vec2-xls-r-300m-german-de --lr 0.0001 --max 28 --seed 1234 --predict --mode topic --audiodir AUDIO_PATH
python3 train_speech_model.py --model facebook/mms-300m --lr 0.00005 --max 27 --seed 1234 --predict --mode topic --audiodir AUDIO_PATH
python3 train_speech_model.py --model utter-project/mHuBERT-147 --lr 0.00005 --max 30 --seed 1234 --predict --mode topic --audiodir AUDIO_PATH
```

The results are saved to `../predictions`.

6. Evaluate the classification results.

Get the subset of the MASSIVE German dataset that is fully parallel to the Bavarian translation (this creates new files in the predictions folder):
```
python3 match_massive_subsets_of_predictions.py
```

Calculate the classification scores (this creates files in `../scores`:
```
python3 evaluate_intents.py
python3 evaluate_topics.py
```
