from transformers import AutoModel, WhisperForAudioClassification, WhisperForConditionalGeneration


for model_name in ["openai/whisper-tiny", "openai/whisper-base",
                   "openai/whisper-small", "openai/whisper-medium",
                   "openai/whisper-large-v3"]:
    print("----------------")
    print(model_name)
    print()
    model = WhisperForAudioClassification.from_pretrained(model_name)
    print("Encoder-only version (WhisperForAudioClassification)")
    print("Parameters in M",
          model.num_parameters(exclude_embeddings =True) / 10 ** 6)
    print(model)
    print()
    # Round down for classification head

for model_name in ["google-bert/bert-base-multilingual-cased",
                   "microsoft/mdeberta-v3-base",
                   "FacebookAI/xlm-roberta-base",
                   "FacebookAI/xlm-roberta-large",
                   "utter-project/mHuBERT-147",
                   "facebook/wav2vec2-xls-r-300m",
                   "facebook/mms-300m"]:
    print("----------------")
    print(model_name)
    print()
    model = AutoModel.from_pretrained(model_name)
    print("Parameters in M",
          model.num_parameters(exclude_embeddings=True) / 10 ** 6)
    print(model)
    print()
