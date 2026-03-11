from transformers import TrainingArguments, Trainer, AutoModelForAudioClassification, AutoProcessor, AutoFeatureExtractor, AutoTokenizer, DataCollatorWithPadding
import evaluate
import random
import torch
import argparse
from datasets import Audio, load_dataset
import librosa
from pathlib import Path
import numpy as np


metric = evaluate.load("accuracy")


def preprocess(path, whisper_sr):
    audio, orig_sr = librosa.load(path, sr=None)
    # Remove the occasional bit of trailing silence
    trimmed, _ = librosa.effects.trim(audio, top_db=30)
    # Resample -- Whisper requires a sampling rate of 16k Hz
    resampled = librosa.resample(
        trimmed, orig_sr=orig_sr, target_sr=whisper_sr)
    return resampled


def get_files_and_metadata(in_file, feature_extractor, audio_path_pfx):
    print("Reading " + in_file)
    details = []
    audios = []
    with open(in_file) as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            idx, text_gold, intent, filepath = line.strip().split("\t")
            filepath = audio_path_pfx + "/" + filepath
            if data_mode == "topic" and "_dial" in in_file:
                dialect = filepath.split("/")[-1].split("_")[1]
                details.append(
                    [idx, dialect, text_gold, intent, filepath])
            else:
                details.append(
                    [idx, text_gold, intent, filepath])
            audios.append(preprocess(
                filepath, feature_extractor.sampling_rate))
    return audios, details


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    # hard-code chunk_length to recommended value of 30 seconds (Whisper) as wav2vec2 does not have chunk_length
    # https://huggingface.co/openai/whisper-large-v3; https://huggingface.co/openai/whisper-small
    chunk_length = 30.0  # 30 seconds
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(
            feature_extractor.sampling_rate * chunk_length),
        truncation=True,
    )
    return inputs


def prep_speechmassive(split_sm_name, split_local_name,
                       feature_extractor, label2id=None):
    if split_sm_name == "test":
        # Requires huggingface-cli login
        dataset = load_dataset(
            "FBK-MT/Speech-MASSIVE-test", "de-DE", split=split_sm_name)
    else:
        dataset = load_dataset(
            "FBK-MT/Speech-MASSIVE", "de-DE", split=split_sm_name) 
    # Only include the instances that are mappable to the
    # xSID intents (based on how we mapped the written version).
    id2intent = {}
    with open(f"../data/intents/massive_deu_{split_local_name}_mapped.tsv") as f:
        for line in f:
            cells = line.strip().split("\t")
            id2intent[cells[0]] = cells[2]
    include_hf_idx = []
    labels = []
    for i, entry in enumerate(dataset):
        if entry["id"] in id2intent:
            include_hf_idx.append(i)
            labels.append(id2intent[entry["id"]])
    dataset = dataset.select(include_hf_idx)
    dataset = dataset.add_column("intent", labels)
    # dataset = map_speechmassive(split_sm_name, split_local_name)
    id2label = None
    if not label2id:
        label_set = sorted(list(set(labels)))
        label2id = {}
        id2label = {}
        for i, l in enumerate(label_set):
            label2id[l] = i
            id2label[i] = l
    dataset = dataset.map(
        lambda x: {"label": label2id[x["intent"]]})
    dataset = dataset.select_columns(["audio", "label"])
    dataset = dataset.cast_column(
        "audio", Audio(sampling_rate=feature_extractor.sampling_rate))
    print(dataset)
    dataset = dataset.map(
        preprocess_function,
        remove_columns="audio",
        batched=True,
        batch_size=16,
    )
    return dataset, label2id, id2label


def prep_swissdial(split_name, feature_extractor,
                   label2id=None):
    data_path = f"../data/topics/swissdial_{split_name}_deu.tsv"
    dataset = load_dataset(
        "csv", data_files=data_path, delimiter="\t")["train"]
    dataset = dataset.rename_column("Audio", "audio")
    id2label = None
    if not label2id:
        label_set = sorted(list(set(dataset["Topic"])))
        label2id = {}
        id2label = {}
        for i, l in enumerate(label_set):
            label2id[l] = i
            id2label[i] = l
    dataset = dataset.map(
        lambda x: {"label": label2id[x["Topic"]]}
    )
    dataset = dataset.select_columns(["audio", "label"])
    dataset = dataset.cast_column(
        "audio", Audio(sampling_rate=feature_extractor.sampling_rate))
    dataset = dataset.map(
        preprocess_function,
        remove_columns="audio",
        batched=True,
        batch_size=16,
    )
    return dataset, label2id, id2label


def map_speechmassive_test():
    # Requires huggingface-cli login
    dataset = load_dataset(
        "FBK-MT/Speech-MASSIVE-test", "de-DE", split="test")
    id2intent = {}
    with open("../data/intents/massive_deu_test_mapped.tsv") as f:
        for line in f:
            cells = line.strip().split("\t")
            id2intent[cells[0]] = cells[2]
    include_hf_idx = []
    labels = []
    for i, entry in enumerate(dataset):
        if entry["id"] in id2intent:
            include_hf_idx.append(i)
            labels.append(id2intent[entry["id"]])
    dataset = dataset.select(include_hf_idx)
    dataset = dataset.add_column("intent_mapped", labels)
    return dataset


def predict(audio, model, device):
    # 'audio' can be a single clip, or a list of audio clips
    
    # whisper and wav2vec2 models use different keys for audio inputs
    if "whisper" in model_name:
        input_features = processor(
        audio,
        padding="max_length",
        max_length=int(feature_extractor.sampling_rate * 30),  # 30 seconds for whisper models
        truncation=True,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        )

        input_features = input_features.get("input_features")

    else: # keep name open to also accept other models like HuBERT
        input_features = feature_extractor(
        audio,
        padding=True,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        )

        input_features = input_features.get("input_values", None)

        if input_features is None:
            # raise error if a model does not use "input_values" as key
            raise ValueError("Input features not found in the feature extractor output. The model probably uses another key for audio inputs instead of 'input_values'. Please check the model documentation.")
        
    input_features = input_features.to(device)
    with torch.no_grad():
        logits = model(input_features).logits
    predicted_class_ids = torch.argmax(logits, axis=1)
    return predicted_class_ids


def write_output(
        details, predictions,
        output_dir, model_name,
        num_train_epochs, batch_size,
        learning_rate, random_seed, data_name,
        dial=False):
    out_file = f"{output_dir}/{model_name.replace('/', '-')}_maxep{num_train_epochs}_bs{batch_size}_lr{int(learning_rate*100000)}_seed{random_seed}_{data_name}.tsv"
    with open(out_file, "w", encoding="utf8") as f_out:
        if dial:
            f_out.write("ID\tDIAL\tSENT\tGOLD\tPRED\n")
        else:
            f_out.write("ID\tSENT\tGOLD\tPRED\n")

        for det, pred in zip(details, predictions):
            f_out.write("\t".join(det[:-1]))
            f_out.write("\t" + id2label[pred.item()] + "\n")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="huggingface model name",
                        required=True)
    parser.add_argument('--lr', type=float, help="learning rate",
                        required=True)
    parser.add_argument('--batch', type=int, default=32, help="batch size")
    parser.add_argument('--max', type=int, default=10,
                        help="max. training epochs")
    parser.add_argument('--seed', type=int, help="random seed",
                        required=True)
    parser.add_argument('--predict', action="store_true",
                        help="predict on test sets")
    parser.add_argument('--mode', type=str, choices=['intent', 'topic'],
                        help="prediction mode, either intent or topic",
                        required=True)
    parser.add_argument('--audiodir', help="path to audio file directory",
                        required=True)
    args = parser.parse_args()
    print(args)

    model_name = args.model
    learning_rate = args.lr
    batch_size = args.batch
    num_train_epochs = args.max
    random_seed = args.seed
    predict_test = args.predict
    data_mode = args.mode
    audio_path_pfx = args.audiodir

    if "whisper" in model_name:
        # load processor for Whisper models
        print("Loading processor and feature extractor...")
        processor = AutoProcessor.from_pretrained(model_name)
        feature_extractor = processor.feature_extractor
    else:
        # load feature extractor for wav2vec2 and similar models
        print("Loading feature extractor...")
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    print("Loading and preparing dataset...")
    if data_mode == "intent":
        dataset_train, label2id, id2label = prep_speechmassive(
            "train", "train", feature_extractor)
        dataset_dev, _, _ = prep_speechmassive(
            "validation", "dev", feature_extractor, label2id)
        num_labels = len(label2id)

    elif data_mode == "topic":
        dataset_train, label2id, id2label = prep_swissdial(
            "train", feature_extractor)
        dataset_dev, _, _ = prep_swissdial(
            "dev", feature_extractor, label2id)
        num_labels = len(label2id)

    # Set the seed before creating the randomly
    # instantiated classification head
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    print("Loading model...")
    model = AutoModelForAudioClassification.from_pretrained(
        model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=model_name.replace("/", "-"),
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=learning_rate,
        seed=random_seed,
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_strategy="epoch",
        eval_strategy="epoch",
        # optimizer is AdamW by default: optim=OptimizerNames.ADAMW_TORCH
    )

    data_collator = DataCollatorWithPadding(feature_extractor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    print("Training done.\n")
    print(f"Training ended at epoch {trainer.state.epoch}")

    if predict_test:
        if data_mode == "intent":
            print("Starting prediction on test sets...")
            output_dir = "../predictions/intents/speech+speech"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Speech-MASSIVE
            data = map_speechmassive_test()
            audios = [entry["array"] for entry in data["audio"]]
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = model.to(device)
            i = 0
            predictions = []

            print("Predicting on Speech-MASSIVE test set...")
            while i < len(audios):
                print(i)
                audio_subset = audios[i:i + 50]
                predictions += predict(
                    audio_subset, model, device)
                i += 50
            details = [(entry["id"], entry["utt"], entry["intent_mapped"])
                    for entry in data]
            data_name = "speechmassive_de_test"
            write_output(details, predictions, output_dir, model_name,
                        num_train_epochs, batch_size, learning_rate, random_seed,
                        data_name)
            print(f"\nPredictions saved to {output_dir}/{model_name.replace('/', '-')}_maxep{num_train_epochs}_bs{batch_size}_lr{int(learning_rate*100000)}_seed{random_seed}_{data_name}.tsv\n")

            # xSID-audio (German and Bavarian)
            for in_file in [
                    "../data/intents/xsid_de_test.tsv", "../data/intents/xsid_de-ba_test.tsv"]:
                print(in_file)
                audios, details = get_files_and_metadata(in_file, feature_extractor, audio_path_pfx)
                predictions = []
                i = 0
                print("Predicting on xSID test set...")
                while i < len(audios):
                    print(i)
                    audio_subset = audios[i:i + 50]
                    predictions += predict(
                        audio_subset, model, device)
                    i += 50
                data_name = in_file.split("/")[-1][:-4]
                write_output(details, predictions, output_dir, model_name,
                            num_train_epochs, batch_size, learning_rate,
                            random_seed, data_name)
                print(f"\nPredictions saved to {output_dir}/{model_name.replace('/', '-')}_maxep{num_train_epochs}_bs{batch_size}_lr{int(learning_rate*100000)}_seed{random_seed}_{data_name}.tsv.tsv\n")
        
        elif data_mode == "topic":
            print("Starting prediction on test sets...")
            output_dir = "../predictions/topics/speech+speech"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Swissdial (Swiss German)
            for in_file in [
                    "../data/topics/swissdial_test_dial.tsv"]:
                print(in_file)
                audios, details = get_files_and_metadata(in_file, feature_extractor, audio_path_pfx)
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                model = model.to(device)
                predictions = []
                i = 0
                print("Predicting on SwissDial test sets...")
                while i < len(audios):
                    print(i)
                    audio_subset = audios[i:i + 50]
                    predictions += predict(
                        audio_subset, model, device)
                    i += 50
                data_name = in_file.split("/")[-1][:-4]
                if "_dial" in in_file:
                    write_output(details, predictions, output_dir, model_name,
                            num_train_epochs, batch_size, learning_rate,
                            random_seed, data_name, dial=True)
                else:
                    write_output(details, predictions, output_dir, model_name,
                                num_train_epochs, batch_size, learning_rate,
                                random_seed, data_name)
                print(f"\nPredictions saved to {output_dir}/{model_name.replace('/', '-')}_maxep{num_train_epochs}_bs{batch_size}_lr{int(learning_rate*100000)}_seed{random_seed}_{data_name}.tsv.tsv\n")
