#!/usr/bin/env python3

import sys
from nltk import edit_distance
import re
import numpy as np
from glob import glob
import os


def read_predictions(filename):
    indices, hypos, labels, refs, dials = [], [], [], [], []
    with open(filename) as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            line = line.strip()
            if not line:
                continue
            if "_dial" in filename:
                idx, dial, hypo, label, ref = line.split("\t")
                dials.append(dial)
            else:
                idx, hypo, label, ref = line.split("\t")
            indices.append(idx)
            hypos.append(hypo)
            labels.append(label)
            refs.append(ref)
    return indices, hypos, labels, refs, dials


def preprocess_transcription(line):
    line = line.lower()
    # Remove word-initial/-final punctuation
    line = re.sub(r"(?<![\w])[^\w\s]", "", line)
    line = re.sub(r"[^\w\s](?![\w])", "", line)
    return line


def wer_cer(hypos, refs):
    # Sentence-level WER and CER
    wer, cer = [], []
    for hypo, ref in zip(hypos, refs):
        hypo = preprocess_transcription(hypo)
        ref = preprocess_transcription(ref)
        hypo_words, ref_words = hypo.split(), ref.split()
        cer.append(edit_distance(hypo, ref) / len(ref))
        wer.append(edit_distance(hypo_words, ref_words) / len(ref_words))
    return wer, cer


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: evaluate_asr.py PREDICTIONS_FOLDER")
        sys.exit(1)

    results_folder = sys.argv[1]
    if results_folder.endswith("/"):
        results_folder = results_folder[:-1]
    model_name = results_folder.split("/")[-1]
    data_type = results_folder.split("/")[-3]
    out_dir = f"../scores/asr/{data_type}/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if "intents" in results_folder:
        # One file with the average scores per dataset
        with open(out_dir + model_name + ".tsv", "w") as f:
            f.write("DATASET\tN\t")
            f.write("WER_MEAN\tWER_STDEV\t")
            f.write("WER_STD_MEAN\tWER_STD_STDEV\t")
            f.write("CER_MEAN\tCER_STDEV\t")
            f.write("CER_STD_MEAN\tCER_STD_STDEV\n")
            # And one file with the scores for each predicted sentence
            # (for manually inspecting the results + calculating correlations)
            with open(out_dir + model_name + "_detailed.tsv", "w") as f_det:
                f_det.write("DATASET\tID\tGOLD_ORIG\tGOLD_STD\tPRED\t")
                f_det.write("WER\tWER_STD\tCER\tCER_STD\tDIAL\n")
                for file in sorted(glob(results_folder + "/*.tsv")):
                    dataset_name = file.split("/")[-1][:-4]
                    indices, hypos, labels, refs, dials = read_predictions(file)
                    # print(dials)
                    wer, cer = wer_cer(hypos, refs)
                    wer_std, cer_std, refs_std = None, None, None
                    dialectal = "_de-ba_" in dataset_name
                    if dialectal:
                        # Also compare the hypotheses to the Std German references.
                        # NOTE: The Bavarian and Std German translations of xSID
                        # were produced independently, so might use different
                        # phrasing.
                        std_file = file.replace("_de-ba_", "_de_")
                        _, _, _, refs_std, _ = read_predictions(std_file)
                        wer_std, cer_std = wer_cer(hypos, refs_std)
                    i = 0
                    while i < len(indices):
                        f_det.write(f"{dataset_name}\t{indices[i]}\t{refs[i]}\t")
                        f_det.write(refs_std[i] if refs_std else "---")
                        f_det.write(f"\t{hypos[i]}\t{100 * wer[i]:.2f}\t")
                        f_det.write(f"{100 * wer_std[i]:.2f}" if wer_std else "---")
                        f_det.write(f"\t{100 * cer[i]:.2f}\t")
                        f_det.write(f"{100 * cer_std[i]:.2f}" if cer_std else "---")
                        f_det.write(f"\t{dials[i]}" if dials else "\t---")
                        f_det.write("\n")
                        i += 1
                    f.write(f"{dataset_name}\t{len(indices)}\t")
                    f.write(f"{100 * np.mean(wer):.2f}\t{100 * np.std(wer):.2f}\t")
                    if wer_std:
                        f.write(f"{100 * np.mean(wer_std):.2f}\t{100 * np.std(wer_std):.2f}\t")
                    else:
                        f.write("---\t---\t")
                    f.write(f"{100 * np.mean(cer):.2f}\t{100 * np.std(cer):.2f}\t")
                    if cer_std:
                        f.write(f"{100 * np.mean(cer_std):.2f}\t{100 * np.std(cer_std):.2f}\n")
                    else:
                        f.write("---\t---\n")
    else:
        dialects_ch = ["ag", "be", "bs", "gr", "lu", "sg", "vs", "zh"]
        n_dials = len(dialects_ch)
        # One file with the average scores per dataset
        with open(out_dir + model_name + ".tsv", "w") as f:
            f.write("DATASET\tN\t")
            f.write("WER_MEAN\tWER_STDEV\t")
            f.write("WER_STD_MEAN\tWER_STD_STDEV\t")
            f.write("CER_MEAN\tCER_STDEV\t")
            f.write("CER_STD_MEAN\tCER_STD_STDEV")
            for dial in dialects_ch:
                f.write(f"\t{dial}_WER_MEAN\t{dial}_WER_STDEV")
                f.write(f"\t{dial}_WER_STD_MEAN\t{dial}_WER_STD_STDEV")
                f.write(f"\t{dial}_CER_MEAN\t{dial}_CER_STDEV")
                f.write(f"\t{dial}_CER_STD_MEAN\t{dial}_CER_STD_STDEV")
            f.write("\n")
            # And one file with the scores for each predicted sentence
            # (for manually inspecting the results + calculating correlations)
            with open(out_dir + model_name + "_detailed.tsv", "w") as f_det:
                f_det.write("DATASET\tID\tGOLD_ORIG\tGOLD_STD\tPRED\t")
                f_det.write("WER\tWER_STD\tCER\tCER_STD\tDIAL\n")
                for file in sorted(glob(results_folder + "/*.tsv")):
                    print(file)
                    dataset_name = file.split("/")[-1][:-4]
                    indices, hypos, labels, refs, dials = read_predictions(file)
                    # print(dials)
                    wer, cer = wer_cer(hypos, refs)
                    wer_std, cer_std, refs_std = None, None, None
                    dialect2hypos, dialect2refs = None, None
                    dialectal = "_dial" in dataset_name
                    if dialectal:
                        # - Also compare the hypotheses to the Std German references.
                        # NOTE: The phrasing varies across (Swiss) German varieties.
                        # - Get the scores per canton.
                        dialect2hypos = {dial: [] for dial in dialects_ch}
                        dialect2refs = {dial: [] for dial in dialects_ch}
                        for (hypo, ref, dial) in zip(hypos, refs, dials):
                            dialect2hypos[dial].append(hypo)
                            dialect2refs[dial].append(ref)
                        std_file = file.replace("_dial", "_deu")
                        indices_std, _, _, refs_std_unmatched, _ = read_predictions(std_file)
                        idx2ref_std = {idx: ref for idx, ref in zip(
                            indices_std, refs_std_unmatched)}
                        refs_std = []
                        for idx in indices:
                            refs_std.append(idx2ref_std[idx])
                        wer_std, cer_std = wer_cer(hypos, refs_std)
                    i = 0
                    while i < len(indices):
                        f_det.write(f"{dataset_name}\t{indices[i]}\t{refs[i]}\t")
                        f_det.write(refs_std[i] if refs_std else "---")
                        f_det.write(f"\t{hypos[i]}\t{100 * wer[i]:.2f}\t")
                        f_det.write(f"{100 * wer_std[i]:.2f}" if wer_std else "---")
                        f_det.write(f"\t{100 * cer[i]:.2f}\t")
                        f_det.write(f"{100 * cer_std[i]:.2f}" if cer_std else "---")
                        f_det.write(f"\t{dials[i]}" if dials else "\t---")
                        f_det.write("\n")
                        i += 1
                    f.write(f"{dataset_name}\t{len(indices)}\t")
                    f.write(f"{100 * np.mean(wer):.2f}\t{100 * np.std(wer):.2f}\t")
                    if wer_std:
                        f.write(f"{100 * np.mean(wer_std):.2f}\t{100 * np.std(wer_std):.2f}\t")
                    else:
                        f.write("---\t---\t")
                    f.write(f"{100 * np.mean(cer):.2f}\t{100 * np.std(cer):.2f}\t")
                    if cer_std:
                        f.write(f"{100 * np.mean(cer_std):.2f}\t{100 * np.std(cer_std):.2f}")
                    else:
                        f.write("---\t---")
                    if dialect2hypos:
                        for dial in dialects_ch:
                            wer, cer = wer_cer(dialect2hypos[dial],
                                               dialect2refs[dial])
                            wer_std, cer_std = wer_cer(dialect2hypos[dial],
                                                       refs_std)
                            f.write(f"\t{100 * np.mean(wer):.2f}\t{100 * np.std(wer):.2f}")
                            f.write(f"\t{100 * np.mean(wer_std):.2f}\t{100 * np.std(wer_std):.2f}")
                            f.write(f"\t{100 * np.mean(cer):.2f}\t{100 * np.std(cer):.2f}")
                            f.write(f"\t{100 * np.mean(cer_std):.2f}\t{100 * np.std(cer_std):.2f}")
                    else:
                        f.write("".join(n_dials * ["\t---"]))
                    f.write("\n")
