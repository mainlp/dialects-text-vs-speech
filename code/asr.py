import sys
import os
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import argparse


def preprocess(path, whisper_sr):
    audio, orig_sr = librosa.load(path, sr=None)
    # Remove the occasional bit of trailing silence
    trimmed, _ = librosa.effects.trim(audio, top_db=30)
    # Resample -- Whisper requires a sampling rate of 16k Hz
    resampled = librosa.resample(
        trimmed, orig_sr=orig_sr, target_sr=whisper_sr)
    return resampled


def transcribe_whisper(
        audio, lang, task,
        processor, model, device):
    # 'audio' can be a single clip, or a list of audio clips
    if lang:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=lang, task=task)
    else:
        forced_decoder_ids = None
    input_features = processor(
        audio,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt").input_features
    input_features = input_features.to(device)
    predicted_ids = model.generate(
        input_features, forced_decoder_ids=forced_decoder_ids)
    transcriptions = processor.batch_decode(
        predicted_ids, skip_special_tokens=True)
    return transcriptions


def transcribe_wav2vec(
        audio, processor, model, device):
    inputs = processor(
        audio,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(
            inputs.input_values,
            attention_mask=inputs.attention_mask).logits
    predicted_ids = torch.argmax(outputs, dim=-1)
    transcriptions = processor.batch_decode(
        predicted_ids)
    return transcriptions


def get_files_and_metadata(in_file, audio_path_pfx):
    print("Reading " + in_file)
    details = []
    audios = []
    with open(in_file) as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            if len(line.strip().split("\t")) == 4:
                idx, text_gold, intent, filepath = line.strip().split("\t")
                filepath = audio_path_pfx + "/" + filepath
                if "_dial" in in_file:
                    dialect = filepath.split("/")[-1].split("_")[1]
                    details.append(
                        [idx, dialect, text_gold, intent, filepath])
                else:
                    details.append(
                        [idx, text_gold, intent, filepath])
                audios.append(preprocess(
                    filepath, processor.feature_extractor.sampling_rate))
            else:
                # skip lines without audio paths
                continue
    return audios, details


def write_output(out_folder, in_file,
                 details, transcriptions):
    out_file = out_folder + "/" + in_file.split("/")[-1]
    print("Writing transcriptions to " + out_file)
    with open(out_file, "w") as f:
        if "_dial" in in_file:
            f.write("INDEX\tDIALECT\tTEXT_ASR\tINTENT\tTEXT_GOLD\n")
            for det, trans in zip(details, transcriptions):
                f.write(f"{det[0]}\t{det[1]}\t{trans.strip()}\t{det[3]}\t{det[2]}\n")
        else:
            f.write("INDEX\tTEXT_ASR\tINTENT\tTEXT_GOLD\n")
            for det, trans in zip(details, transcriptions):
                f.write(f"{det[0]}\t{trans.strip()}\t{det[2]}\t{det[1]}\n")


def map_speechmassive(split_sm_name, split_local_name):
    if split_sm_name == "test":
        # Requires huggingface-cli login
        dataset = load_dataset(
            "FBK-MT/Speech-MASSIVE-test", "de-DE", split=split_sm_name)
    else:
        dataset = load_dataset(
            "FBK-MT/Speech-MASSIVE", "de-DE", split=split_sm_name)
    id2intent = {}
    with open(
            f"../data/intents/massive_deu_{split_local_name}_mapped.tsv") as f:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="huggingface model name",
                        required=True)
    parser.add_argument('--task', type=str, choices=['intent', 'topic'],
                        help="data type, either intent or topic",)
    parser.add_argument('--audiodir', help="path to audio file directory",
                        required=True)
    parser.add_argument('--train', action='store_true',
                        help="also transcribe training data (intents only)")
    parser.set_defaults(train=False)

    args = parser.parse_args()
    print(args)

    model_name = args.model
    nlp_task = args.task
    audio_path_pfx = args.audiodir

    if "whisper" in model_name:
        target_lang = "german"  # no dialects available
        task = "transcribe"  # 'translate' only creates English output
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
    elif "mms" in model_name or "wav2vec2" in model_name:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        if "mms" in model_name:
            target_lang = "deu"  # no dialects available
            processor.tokenizer.set_target_lang(target_lang)
            model.load_adapter(target_lang)
    else:
        print("Unknown model:")
        print(model_name)
        sys.exit(1)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = model.to(device)

    if nlp_task == "intent":
        in_files = [
            "../data/intents/xsid_de_dev.tsv",
            "../data/intents/xsid_de_test.tsv",
            "../data/intents/xsid_de-ba_dev.tsv",
            "../data/intents/xsid_de-ba_test.tsv"]
        out_folder = "../data/intents/asr/" + model_name.replace("/", "-")
    elif nlp_task == "topic":
        in_files = [
            "../data/topics/swissdial_test_dial.tsv",
            "../data/topics/swissdial_dev_dial.tsv"]
        out_folder = "../data/topics/asr/" + model_name.replace("/", "-")

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for in_file in in_files:
        print(in_file)
        audios, details = get_files_and_metadata(in_file, audio_path_pfx)
        transcriptions = []
        i = 0
        while i < len(audios):
            print(i)
            audio_subset = audios[i:i + 50]
            if "whisper" in model_name:
                transcriptions += transcribe_whisper(
                    audio_subset, target_lang, task, processor, model, device)
                print(transcriptions[i])
            elif "mms" in model_name or "wav2vec2" in model_name:
                transcriptions += transcribe_wav2vec(
                    audio_subset, processor, model, device)
                print(transcriptions[i])
            i += 50
        write_output(out_folder, in_file,
                     details, transcriptions)

    if nlp_task == "intent":
        spmassive_splits = [("validation", "dev"), ("test", "test")]
        if args.train:
            spmassive_splits.append(("train", "train"))
        for (split_sm_name, split_local_name) in spmassive_splits:
            print(split_local_name)
            data = map_speechmassive(split_sm_name, split_local_name)
            audios = [entry["array"] for entry in data["audio"]]
            transcriptions = []
            i = 0
            while i < len(audios):
                print(i)
                audio_subset = audios[i:i + 50]
                if "whisper" in model_name:
                    transcriptions += transcribe_whisper(
                        audio_subset, target_lang, task, processor,
                        model, device)
                    print(transcriptions[i])
                elif "wav2vec" in model_name:
                    transcriptions += transcribe_wav2vec(
                        audio_subset, processor, model, device)
                    print(transcriptions[i])
                i += 50
            details = [(entry["id"], entry["utt"], entry["intent_mapped"])
                       for entry in data]
            in_file = "speechmassive_de_" + split_local_name + ".tsv"
            write_output(out_folder, in_file, details, transcriptions)
