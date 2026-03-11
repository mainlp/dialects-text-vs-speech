from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import random
import torch
import argparse
from glob import glob
from pathlib import Path


metric = evaluate.load("accuracy")


def read_data_text(filename, data_mode):
    sentences, labels, indices, dialects = [], [], [], []
    dial_asr = False
    index_idx = 0
    sent_idx = 1
    label_idx = 2
    if "asr" in filename and "_dial" in filename:
        dial_asr = True
        sent_idx = 2
        label_idx = 3
    with open(filename, encoding="utf8") as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            cells = line.strip().split("\t")
            # lowercase -> the datasets have different capitalization
            # standards, as do the ASR systems
            indices.append(cells[index_idx])
            sentences.append(cells[sent_idx].lower())
            labels.append(cells[label_idx])
            if data_mode == "topic" and "_dial" in filename:
                if dial_asr:
                    dialects.append(cells[1])
                else:
                    audiofile = cells[3]
                    if audiofile.endswith(".wav"):  # filter out non-audio cells
                        dialect = audiofile.split("/")[-1]
                        dialect = dialect.split("_")[1]
                        dialects.append(dialect)
                    else:
                        print("Non-audio cell:", line)

    if dialects:
        dataset = Dataset.from_dict(
            {"index": indices, "text": sentences, "label": labels,
             "dialect": dialects})
    else:
        dataset = Dataset.from_dict(
            {"index": indices, "text": sentences, "label": labels})

    return dataset


def tokenize(examples):
    if "deberta" in model_name:
        return tokenizer(examples["text"], padding="max_length",
                         max_length=512, truncation=True)
    return tokenizer(examples["text"], padding="max_length",
                     truncation=True)


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
    parser.add_argument('--asr', type=str, default="",
                        help="use train/dev data ASR'ed by the specified model (only for intents)",)
    args = parser.parse_args()
    print(args)

    model_name = args.model
    learning_rate = args.lr
    batch_size = args.batch
    num_train_epochs = args.max
    random_seed = args.seed
    predict_test = args.predict
    data_mode = args.mode
    asr_pretty = args.asr.replace("/", "-")

    if data_mode == "intent":
        folder = "../data/intents/"
        if args.asr:
            print("Using ASR'ed train/dev data", args.asr)
            folder = "../data/intents/asr/" + asr_pretty + "/"
            dataset_train = read_data_text(
                folder + "speechmassive_de_train.tsv", data_mode)
            dataset_dev = read_data_text(
                folder + "speechmassive_de_dev.tsv", data_mode)
        else:
            dataset_train = read_data_text(
                "../data/intents/massive_deu_train_mapped.tsv", data_mode)
            dataset_dev = read_data_text(
                "../data/intents/massive_deu_dev_mapped.tsv", data_mode)
    elif data_mode == "topic":
        dataset_train = read_data_text(
            "../data/topics/swissdial_train_deu.tsv", data_mode)
        dataset_dev = read_data_text(
            "../data/topics/swissdial_dev_deu.tsv", data_mode)
    label_set = sorted(list(set(dataset_train["label"])))
    num_labels = len(label_set)
    label2id, id2label = {}, {}
    for i, l in enumerate(label_set):
        label2id[l] = i
        id2label[i] = l

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset_train = dataset_train.map(tokenize, batched=True)
    dataset_train = dataset_train.map(
        lambda x: {"label": label2id[x["label"]]})
    dataset_dev = dataset_dev.map(tokenize, batched=True)
    dataset_dev = dataset_dev.map(
        lambda x: {"label": label2id[x["label"]]})

    # Set the seed before creating the randomly
    # instantiated classification head
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    if predict_test:
        if data_mode == "intent":
            orig_text_files = ["../data/intents/massive_deu_test_mapped.tsv",
                            "../data/intents/massive_deba_resplit_test.tsv",
                            "../data/intents/xsid_de_test.tsv",
                            "../data/intents/xsid_de-ba_test.tsv"]
            asr_files = glob("../data/intents/asr/*/*test.tsv")
        elif data_mode == "topic":
            orig_text_files = ["../data/topics/swissdial_test_deu.tsv",
                           "../data/topics/swissdial_test_dial.tsv"]
            asr_files = glob("../data/topics/asr/*/*_test_*.tsv")

        for test_data in orig_text_files + asr_files:
            model_name_out = model_name.replace('/', '-')
            if data_mode == "intent":
                if "asr" in test_data:
                    data_name = test_data.split("/")[-2] + "_" + test_data.split("/")[-1][:-4]
                    if args.asr:
                        output_dir = "../predictions/intents/asr+asr"
                        model_name_out += "-" + asr_pretty
                    else:
                        output_dir = "../predictions/intents/text+asr"
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                else:
                    data_name = "goldtext_" + test_data.split("/")[-1][:-4]
                    if args.asr:
                        output_dir = "../predictions/intents/asr+text"
                        model_name_out += "-" + asr_pretty
                    else:
                        output_dir = "../predictions/intents/text+text"
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
            elif data_mode == "topic":
                if "asr" in test_data:
                    data_name = test_data.split("/")[-2] + "_" + test_data.split("/")[-1][:-4]
                    output_dir = "../predictions/topics/text+asr"
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                else:
                    data_name = "goldtext_" + test_data.split("/")[-1][:-4]
                    output_dir = "../predictions/topics/text+text"
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
            print(test_data)
            dataset_test = read_data_text(test_data, data_mode)
            dataset_test = dataset_test.map(tokenize, batched=True)
            dataset_test = dataset_test.map(
                lambda x: {"label": label2id[x["label"]]})
            predictions = trainer.predict(dataset_test)
            print(data_name, predictions.metrics["test_accuracy"])
            predicted = np.argmax(predictions.predictions, axis=-1)
            with open(f"{output_dir}/{model_name_out}_maxep{num_train_epochs}_bs{batch_size}_lr{int(learning_rate*100000)}_seed{random_seed}_{data_name}.tsv",
                      "w", encoding="utf8") as f_out:
                if data_mode == "topic" and "_dial" in test_data:
                    f_out.write("ID\tDIAL\tSENT\tGOLD\tPRED\n")
                    for idx, sent, gold, dialect, pred in zip(
                            dataset_test["index"], dataset_test["text"],
                            dataset_test["label"], dataset_test["dialect"], predicted):
                        f_out.write(idx + "\t" + dialect + "\t" + sent + "\t" + str(id2label[gold]) + "\t" + str(id2label[pred]) + "\n")
                else:
                    f_out.write("ID\tSENT\tGOLD\tPRED\n")
                    for idx, sent, gold, pred in zip(
                            dataset_test["index"], dataset_test["text"],
                            dataset_test["label"], predicted):
                        f_out.write(idx + "\t" + sent + "\t" + str(id2label[gold]) + "\t" + str(id2label[pred]) + "\n")
