from sklearn.metrics import f1_score, accuracy_score
from glob import glob
from statistics import stdev
from scipy.stats import pearsonr, spearmanr


def score(filename):
    golds, preds = [], []
    with open(filename) as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            _, _, gold, pred = line.strip().split("\t")
            golds.append(gold)
            preds.append(pred)
    return (100 * accuracy_score(golds, preds),
            100 * f1_score(golds, preds, average="macro"))


def get_results(metric):
    text2test2scores = {}
    asr2text2test2scores = {}
    speech2test2scores = {}
    for text_model in text_models_all:
        for test_set in test_sets_text:
            for seed in seeds:
                try:
                    try:
                        filename = glob(f"../predictions/intents/text+text/{text_model}*{seed}*{test_set}.tsv")[0]
                    except IndexError:
                        filename = glob(f"../predictions/intents/asr+text/{text_model}*{seed}*{test_set}.tsv")[0]
                    acc, f1 = score(filename)
                    if text_model not in text2test2scores:
                        text2test2scores[text_model] = {}
                    if test_set not in text2test2scores[text_model]:
                        text2test2scores[text_model][test_set] = []
                    text2test2scores[text_model][test_set].append(
                        acc if metric == "acc" else f1)
                except IndexError:
                    print('-')
                    pass
        for asr_model in asr_models:
            for test_set in test_sets_audio:
                if not test_set:
                    continue
                for seed in seeds:
                    try:
                        try:
                            filename = glob(f"../predictions/intents/text+asr/{text_model}*{seed}*{asr_model}*{test_set}.tsv")[0]
                        except IndexError:
                            filename = glob(f"../predictions/intents/asr+asr/{text_model}*{seed}*{asr_model}*{test_set}.tsv")[0]
                        acc, f1 = score(filename)
                        if asr_model not in asr2text2test2scores:
                            asr2text2test2scores[asr_model] = {}
                        if text_model not in asr2text2test2scores[asr_model]:
                            asr2text2test2scores[asr_model][text_model] = {}
                        if test_set not in asr2text2test2scores[asr_model][text_model]:
                            asr2text2test2scores[asr_model][text_model][test_set] = []
                        asr2text2test2scores[asr_model][text_model][test_set].append(
                            acc if metric == "acc" else f1)
                    except IndexError:
                        pass
    for speech_model in speech_models:
        for test_set in test_sets_audio:
            if not test_set:
                continue
            for seed in seeds:
                try:
                    filename = glob(f"../predictions/intents/speech+speech/{speech_model}*{seed}*{test_set}.tsv")[0]
                    acc, f1 = score(filename)
                    if speech_model not in speech2test2scores:
                        speech2test2scores[speech_model] = {}
                    if test_set not in speech2test2scores[speech_model]:
                        speech2test2scores[speech_model][test_set] = []
                    speech2test2scores[speech_model][test_set].append(
                        acc if metric == "acc" else f1)
                except IndexError:
                    pass

    with open(f"../scores/intents/intent_scores_{metric}_unaggregated.tsv",
              "w") as f:
        f.write("Model\tTrain\tTest processing\tTest set")
        for seed in seeds:
            f.write(f"\t{metric}_{seed}")
        f.write("\n")
        text2test2scores_avg = {}
        for text_model in text2test2scores:
            text2test2scores_avg[text_model] = {}
            for test in text2test2scores[text_model]:
                scores = text2test2scores[text_model][test]
                mean_ = sum(scores) / len(scores)
                if len(scores) > 1:
                    stdev_ = stdev(scores)
                else:
                    stdev_ = -1
                text2test2scores_avg[text_model][test] = (mean_, stdev_)
                f.write(f"{text_model}\tgold text\tgold text\t{test}")
                for score_ in scores:
                    f.write("\t" + str(score_))
                f.write("\n")
        asr2text2test2scores_avg = {}
        for asr_model in asr2text2test2scores:
            asr2text2test2scores_avg[asr_model] = {}
            for text_model in asr2text2test2scores[asr_model]:
                asr2text2test2scores_avg[asr_model][text_model] = {}
                for test in asr2text2test2scores[asr_model][text_model]:
                    scores = asr2text2test2scores[asr_model][text_model][test]
                    mean_ = sum(scores) / len(scores)
                    if len(scores) > 1:
                        stdev_ = stdev(scores)
                    else:
                        stdev_ = -1
                    asr2text2test2scores_avg[asr_model][text_model][test] = (
                        mean_, stdev_)
                    f.write(f"{text_model}\tgold text\tASR {asr_model}\t{test}")
                    for score_ in scores:
                        f.write("\t" + str(score_))
                    f.write("\n")
        speech2test2scores_avg = {}
        for speech_model in speech2test2scores:
            speech2test2scores_avg[speech_model] = {}
            for test in speech2test2scores[speech_model]:
                scores = speech2test2scores[speech_model][test]
                mean_ = sum(scores) / len(scores)
                if len(scores) > 1:
                    stdev_ = stdev(scores)
                else:
                    stdev_ = -1
                speech2test2scores_avg[speech_model][test] = (mean_, stdev_)
                f.write(f"{speech_model}\tspeech\tspeech\t{test}")
                for score_ in scores:
                    f.write("\t" + str(score_))
                f.write("\n")

    return (text2test2scores, text2test2scores_avg,
            asr2text2test2scores, asr2text2test2scores_avg,
            speech2test2scores, speech2test2scores_avg)


def write_detailed_table_text_models(
        text_models, f,
        text2test2scores, text2test2scores_avg,
        asr2text2test2scores, asr2text2test2scores_avg,
        gold_text=True):
    for text_model in text_models:
        if gold_text:
            train_data = "MASSIVE de (gold text)"
        else:
            train_data = "MASSIVE de (ASR)"
        f.write(model2pretty[text_model] + "\t" + train_data + "\tgold text")
        f.write("\t" + str(len(text2test2scores[text_model][mas_de])))

        # MASSIVE de (matched)
        f.write("\t" + str(text2test2scores_avg[text_model][mas_de][0]))
        f.write("\t" + str(text2test2scores_avg[text_model][mas_de][1]))
        f.write("\t")  # delta to text

        # MASSIVE de-ba
        f.write("\t" + str(text2test2scores_avg[text_model][mas_ba][0]))
        f.write("\t" + str(text2test2scores_avg[text_model][mas_ba][1]))
        ## delta to std
        f.write("\t" + str(text2test2scores_avg[text_model][mas_ba][0] - text2test2scores_avg[text_model][mas_de][0]))

        # xSID de
        f.write("\t" + str(text2test2scores_avg[text_model][xsid_de][0]))
        f.write("\t" + str(text2test2scores_avg[text_model][xsid_de][1]))
        f.write("\t")  # delta to text

        # xSID de-ba
        f.write("\t" + str(text2test2scores_avg[text_model][xsid_ba][0]))
        f.write("\t" + str(text2test2scores_avg[text_model][xsid_ba][1]))
        f.write("\t")  # delta to text
        ## delta to std
        f.write("\t" + str(text2test2scores_avg[text_model][xsid_ba][0] - text2test2scores_avg[text_model][xsid_de][0]))

        for asr_model in asr_models:
            f.write(model2pretty[text_model])
            f.write("\tMASSIVE de (gold text)\tASR " + model2pretty[asr_model])
            f.write("\t" + str(len(asr2text2test2scores[asr_model][text_model][speechmas_de])))

            # MASSIVE de (matched)
            f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][speechmas_de][0]))
            f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][speechmas_de][1]))
            ## delta to text
            f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][speechmas_de][0] - text2test2scores_avg[text_model][mas_de][0]))

            # MASSIVE de-ba (no audio)
            f.write("\t\t\t")

            # xSID de
            f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][xsid_de][0]))
            f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][xsid_de][1]))
            ## delta to text
            f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][xsid_de][0] - text2test2scores_avg[text_model][xsid_de][0]))
            
            # xSID de-ba
            f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][xsid_ba][0]))
            f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][xsid_ba][1]))
            ## delta to text
            f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][xsid_ba][0] - text2test2scores_avg[text_model][xsid_ba][0]))
            ## delta to std
            f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][xsid_ba][0] - asr2text2test2scores_avg[asr_model][text_model][xsid_de][0]))
            f.write("\n")


def write_detailed_table(
        metric, text2test2scores, text2test2scores_avg,
        asr2text2test2scores, asr2text2test2scores_avg,
        speech2test2scores, speech2test2scores_avg):
    with open(f"../scores/intents/intent_scores_{metric}-avg.tsv", "w") as f:
        f.write("Model\tTrain\tTest processing\tN\t")
        f.write("MASSIVE de (mean)\tMASSIVE de (stdev)\tMASSIVE de (delta to text)\t")
        f.write("MASSIVE de-ba (mean)\tMASSIVE de-ba (stdev)\tMASSIVE de (delta to std)\t")
        f.write("xSID de (mean)\txSID de (stdev)\txSID de (delta to text)\t")
        f.write("xSID de-ba (mean)\txSID de-ba (stdev)\txSID de-ba (delta to text)\txSID de-ba (delta deu-dial)\n")

        write_detailed_table_text_models(
            text_models_text, f,
            text2test2scores, text2test2scores_avg,
            asr2text2test2scores, asr2text2test2scores_avg,
            gold_text=True)

        for speech_model in speech_models:
            f.write(model2pretty[speech_model] + "\tSpeech-MASSIVE de\tspeech")
            f.write("\t" + str(len(speech2test2scores[speech_model][speechmas_de])))

            # MASSIVE de (matched)
            f.write("\t" + str(speech2test2scores_avg[speech_model][speechmas_de][0]))
            f.write("\t" + str(speech2test2scores_avg[speech_model][speechmas_de][1]))
            ## delta to text
            f.write("\t" + str(speech2test2scores_avg[speech_model][speechmas_de][0] - text2test2scores_avg[text_model][mas_de][0]))

            # MASSIVE de-ba (no audio)
            f.write("\t\t\t")

            # xSID de
            f.write("\t" + str(speech2test2scores_avg[speech_model][xsid_de][0]))
            f.write("\t" + str(speech2test2scores_avg[speech_model][xsid_de][1]))
            ## delta to text
            f.write("\t" + str(speech2test2scores_avg[speech_model][xsid_de][0] - text2test2scores_avg[text_model][xsid_de][0]))

            # xSID de-ba
            f.write("\t" + str(speech2test2scores_avg[speech_model][xsid_ba][0]))
            f.write("\t" + str(speech2test2scores_avg[speech_model][xsid_ba][1]))
            ## delta to text
            f.write("\t" + str(speech2test2scores_avg[speech_model][xsid_ba][0] - text2test2scores_avg[text_model][xsid_ba][0]))
            ## delta to std
            f.write("\t" + str(speech2test2scores_avg[speech_model][xsid_ba][0] - speech2test2scores_avg[speech_model][xsid_de][0]))

            f.write("\n")

        write_detailed_table_text_models(
            text_models_asr, f,
            text2test2scores, text2test2scores_avg,
            asr2text2test2scores, asr2text2test2scores_avg,
            gold_text=False)


def write_table_by_setup(
        metric, text2test2scores, text2test2scores_avg,
        asr2text2test2scores, asr2text2test2scores_avg,
        speech2test2scores, speech2test2scores_avg):
    with open(f"../scores/intents/intent_scores_{metric}-by-setup-type.tsv", "w") as f:
        f.write("Model\tN\t")
        f.write("MASSIVE de (mean)\tMASSIVE de (stdev)\t")
        f.write("MASSIVE de-ba (mean)\tMASSIVE de-ba (stdev)\tMASSIVE de (delta to std)\t")
        f.write("xSID de (mean)\txSID de (stdev)\t")
        f.write("xSID de-ba (mean)\txSID de-ba (stdev)\txSID de-ba (delta deu-dial)\n")

        # Text-only
        rows = []
        all_massive_de, all_massive_ba = [], []
        all_xsid_de, all_xsid_ba = [], []
        for text_model in text_models_text:  # Ignoring extra ASR experiments
            row = model2pretty[text_model]
            row += "\t" + str(len(text2test2scores[text_model][mas_de]))
            # MASSIVE de (matched)
            all_massive_de += text2test2scores[text_model][mas_de].copy()
            row += "\t" + str(text2test2scores_avg[text_model][mas_de][0])
            row += "\t" + str(text2test2scores_avg[text_model][mas_de][1])
            # MASSIVE de-ba
            all_massive_ba += text2test2scores[text_model][mas_ba].copy()
            row += "\t" + str(text2test2scores_avg[text_model][mas_ba][0])
            row += "\t" + str(text2test2scores_avg[text_model][mas_ba][1])
            row += "\t" + str(text2test2scores_avg[text_model][mas_ba][0] - text2test2scores_avg[text_model][mas_de][0])
            # xSID de
            all_xsid_de += text2test2scores[text_model][xsid_de].copy()
            row += "\t" + str(text2test2scores_avg[text_model][xsid_de][0])
            row += "\t" + str(text2test2scores_avg[text_model][xsid_de][1])
            # xSID de-ba
            all_xsid_ba += text2test2scores[text_model][xsid_ba].copy()
            row += "\t" + str(text2test2scores_avg[text_model][xsid_ba][0])
            row += "\t" + str(text2test2scores_avg[text_model][xsid_ba][1])
            row += "\t" + str(text2test2scores_avg[text_model][xsid_ba][0] - text2test2scores_avg[text_model][xsid_de][0])
            row += "\n"
            rows.append(row)
        f.write("*Text-only*")
        f.write("\t" + str(len(all_massive_de)))
        f.write("\t" + str(sum(all_massive_de) / len(all_massive_de)))
        f.write("\t" + str(stdev(all_massive_de)))
        f.write("\t" + str(sum(all_massive_ba) / len(all_massive_ba)))
        f.write("\t" + str(stdev(all_massive_ba)))
        f.write("\t\t" + str(sum(all_xsid_de) / len(all_xsid_de)))
        f.write("\t" + str(stdev(all_xsid_de)))
        f.write("\t" + str(sum(all_xsid_ba) / len(all_xsid_ba)))
        f.write("\t" + str(stdev(all_xsid_ba)))
        f.write("\n")
        for row in rows:
            f.write(row)

        # Cascaded
        rows_by_asr = []
        all_massive, all_xsid_de, all_xsid_ba = [], [], []
        cascaded_text2test2scores = {}
        for asr_model in asr_models:
            test2scores = {}
            for text_model in asr2text2test2scores[asr_model]:
                if text_model not in cascaded_text2test2scores:
                    cascaded_text2test2scores[text_model] = {}
                for test_set in asr2text2test2scores[asr_model][text_model]:
                    scores = asr2text2test2scores[asr_model][text_model][test_set]
                    try:
                        test2scores[test_set] += scores.copy()
                    except KeyError:
                        test2scores[test_set] = scores.copy()
                    try:
                        cascaded_text2test2scores[text_model][test_set] += scores.copy()
                    except KeyError:
                        cascaded_text2test2scores[text_model][test_set] = scores.copy()
            row = model2pretty[asr_model]
            row += "\t" + str(len(test2scores[speechmas_de]))
            # MASSIVE de (matched)
            all_massive += test2scores[speechmas_de].copy()
            row += "\t" + str(sum(test2scores[speechmas_de]) / len(test2scores[speechmas_de]))
            row += "\t" + str(stdev(test2scores[speechmas_de]))
            # MASSIVE de-ba (no audio)
            row += "\t\t\t"
            # xSID de
            all_xsid_de += test2scores[xsid_de].copy()
            xsid_de_mean = sum(test2scores[xsid_de]) / len(test2scores[xsid_de])
            row += "\t" + str(xsid_de_mean)
            row += "\t" + str(stdev(test2scores[xsid_de]))
            # xSID de-ba
            all_xsid_ba += test2scores[xsid_ba].copy()
            xsid_ba_mean = sum(test2scores[xsid_ba]) / len(test2scores[xsid_ba])
            row += "\t" + str(xsid_ba_mean)
            row += "\t" + str(stdev(test2scores[xsid_ba]))
            row += "\t" + str(xsid_ba_mean - xsid_de_mean)
            row += "\n"
            rows_by_asr.append(row)
        row_total_avg = "\t" + str(len(all_massive))
        row_total_avg += "\t" + str(sum(all_massive) / len(all_massive))
        row_total_avg += "\t" + str(stdev(all_massive)) + "\t\t\t"
        row_total_avg += "\t" + str(sum(all_xsid_de) / len(all_xsid_de))
        row_total_avg += "\t" + str(stdev(all_xsid_de))
        row_total_avg += "\t" + str(sum(all_xsid_ba) / len(all_xsid_ba))
        row_total_avg += "\t" + str(stdev(all_xsid_ba)) + "\n"
        f.write("*Cascaded (averaged over ASR models)*")
        f.write(row_total_avg)
        for text_model in text_models_text:  # Ignoring extra ASR experiments
            f.write(model2pretty[text_model])
            scores_mas = cascaded_text2test2scores[text_model][speechmas_de]
            f.write("\t" + str(len(scores_mas)))
            f.write("\t" + str(sum(scores_mas) / len(scores_mas)))
            f.write("\t" + str(stdev(scores_mas)))
            f.write("\t\t\t")
            scores_xsid_de = cascaded_text2test2scores[text_model][xsid_de]
            scores_xsid_de_mean = sum(scores_xsid_de) / len(scores_xsid_de)
            f.write("\t" + str(scores_xsid_de_mean))
            f.write("\t" + str(stdev(scores_xsid_de)))
            scores_xsid_ba = cascaded_text2test2scores[text_model][xsid_ba]
            scores_xsid_ba_mean = sum(scores_xsid_ba) / len(scores_xsid_ba)
            f.write("\t" + str(scores_xsid_ba_mean))
            f.write("\t" + str(stdev(scores_xsid_ba)))
            f.write("\t" + str(scores_xsid_ba_mean - scores_xsid_de_mean))
            f.write("\n")
        f.write("*Cascaded (averaged over text models)*")
        f.write(row_total_avg)
        for row in rows_by_asr:
            f.write(row)

        # Speech-only
        rows = []
        all_massive, all_xsid_de, all_xsid_ba = [], [], []
        for speech_model in speech_models:
            row = model2pretty[speech_model]
            row += "\t" + str(len(speech2test2scores[speech_model][speechmas_de]))
            # MASSIVE de (matched)
            all_massive += speech2test2scores[speech_model][speechmas_de].copy()
            row += "\t" + str(speech2test2scores_avg[speech_model][speechmas_de][0])
            row += "\t" + str(speech2test2scores_avg[speech_model][speechmas_de][1])
            # MASSIVE de-ba (no audio)
            row += "\t\t\t"
            # xSID de
            all_xsid_de += speech2test2scores[speech_model][xsid_de].copy()
            row += "\t" + str(speech2test2scores_avg[speech_model][xsid_de][0])
            row += "\t" + str(speech2test2scores_avg[speech_model][xsid_de][1])
            # xSID de-ba
            all_xsid_ba += speech2test2scores[speech_model][xsid_ba].copy()
            row += "\t" + str(speech2test2scores_avg[speech_model][xsid_ba][0])
            row += "\t" + str(speech2test2scores_avg[speech_model][xsid_ba][1])
            row += "\t" + str(speech2test2scores_avg[speech_model][xsid_ba][0] - speech2test2scores_avg[speech_model][xsid_de][0])
            row += "\n"
            rows.append(row)
        f.write("*Speech-only*")
        f.write("\t" + str(len(all_massive)))
        f.write("\t" + str(sum(all_massive) / len(all_massive)))
        f.write("\t" + str(stdev(all_massive)))
        f.write("\t\t\t")
        f.write("\t" + str(sum(all_xsid_de) / len(all_xsid_de)))
        f.write("\t" + str(stdev(all_xsid_de)))
        f.write("\t" + str(sum(all_xsid_ba) / len(all_xsid_ba)))
        f.write("\t" + str(stdev(all_xsid_ba)))
        f.write("\n")
        for row in rows:
            f.write(row)


def write_deltas_setups(
        metric, text2test2scores, text2test2scores_avg,
        asr2text2test2scores, asr2text2test2scores_avg,
        speech2test2scores, speech2test2scores_avg):
    test_sets_comparable = [speechmas_de, xsid_de, xsid_ba]
    with open(f"../scores/intents/intent_scores_{metric}-deltas-cascaded-text.tsv", "w") as f:
        f.write("Model\t")
        f.write("MASSIVE de (mean)\tMASSIVE de (stdev)")
        f.write("xSID de (mean)\txSID de (stdev)\t")
        f.write("xSID de-ba (mean)\txSID de-ba (stdev)\n")
        for asr_model in asr_models:
            test2diffs = {}
            for text_model in text_models_text:  # Ignoring extra ASR experiments
                for test_set in test_sets_comparable:
                    asr_avg, _ = asr2text2test2scores_avg[asr_model][text_model][test_set]
                    if test_set == speechmas_de:
                        text_avg, _ = text2test2scores_avg[text_model][mas_de]
                    else:
                        text_avg, _ = text2test2scores_avg[text_model][test_set]
                    delta = asr_avg - text_avg
                    try:
                        test2diffs[test_set].append(delta)
                    except KeyError:
                        test2diffs[test_set] = [delta]
            f.write(model2pretty[asr_model])
            for test_set in test_sets_comparable:
                f.write("\t" + str(sum(test2diffs[test_set]) / len(test2diffs[test_set])))
                f.write("\t" + str(stdev(test2diffs[test_set])))
            f.write("\n")
    with open(f"../scores/intents/intent_scores_{metric}-deltas-speech-cascaded.tsv", "w") as f:
        f.write("Model\t")
        f.write("MASSIVE de (mean)\tMASSIVE de (stdev)")
        f.write("xSID de (mean)\txSID de (stdev)\t")
        f.write("xSID de-ba (mean)\txSID de-ba (stdev)\n")
        for asr_model in asr_models:
            if not asr_model in speech_models:
                continue
            test2diffs = {}
            # Ignoring the extra ASR experiments
            for text_model in text_models_text:
                for test_set in test_sets_comparable:
                    asr_avg, _ = asr2text2test2scores_avg[asr_model][text_model][test_set]
                    speech_avg, _ = speech2test2scores_avg[asr_model][test_set]
                    delta = speech_avg - asr_avg
                    try:
                        test2diffs[test_set].append(delta)
                    except KeyError:
                        test2diffs[test_set] = [delta]
            f.write(model2pretty[asr_model])
            for test_set in test_sets_comparable:
                f.write("\t" + str(sum(test2diffs[test_set]) / len(test2diffs[test_set])))
                f.write("\t" + str(stdev(test2diffs[test_set])))
            f.write("\n")
    with open(f"../scores/intents/intent_scores_{metric}-deltas-speech-text.tsv", "w") as f:
        f.write("Model\t")
        f.write("MASSIVE de (mean)\tMASSIVE de (stdev)")
        f.write("xSID de (mean)\txSID de (stdev)\t")
        f.write("xSID de-ba (mean)\txSID de-ba (stdev)\n")
        text_averages = {test: [] for test in test_sets_comparable}
        # Ignoring the extra ASR experiments
        for text_model in text_models_text:
            for test_set in test_sets_comparable:
                if test_set == speechmas_de:
                    avg, _ = text2test2scores_avg[text_model][mas_de]
                else:
                    avg, _ = text2test2scores_avg[text_model][test_set]
                text_averages[test_set].append(avg)
        for speech_model in speech_models:
            test2diffs = {}
            for test_set in test_sets_comparable:
                speech_avg, _ = speech2test2scores_avg[speech_model][test_set]
                # Ignoring the extra ASR experiments
                for text_model in text_models_text:
                    if test_set == speechmas_de:
                        text_avg, _ = text2test2scores_avg[text_model][mas_de]
                    else:
                        text_avg, _ = text2test2scores_avg[text_model][test_set]
                    delta = speech_avg - text_avg
                    try:
                        test2diffs[test_set].append(delta)
                    except KeyError:
                        test2diffs[test_set] = [delta]
            f.write(model2pretty[speech_model])
            for test_set in test_sets_comparable:
                f.write("\t" + str(sum(test2diffs[test_set]) / len(test2diffs[test_set])))
                f.write("\t" + str(stdev(test2diffs[test_set])))
            f.write("\n")


def get_asr_results():
    asrhypo_model2test2scores = {}
    for asr_model in asr_models:
        asrhypo_model2test2scores[asr_model] = {}
        with open(f"../scores/asr/intents/{asr_model}.tsv") as f:
            for line in f:
                cells = line.split("\t")
                if not cells[0] in test_sets_audio:
                    continue
                test, wer, werstd, cer, cerstd = cells[0], cells[2], cells[4], cells[6], cells[8]
                wer = float(wer)
                cer = float(cer)
                if not werstd.startswith("-"):
                    # Comparisons of hypotheses for dialect audio to references for std
                    werstd = float(werstd)
                    cerstd = float(cerstd)
                else:
                    werstd, cerstd = None, None
                asrhypo_model2test2scores[asr_model][test] = {
                    "wer": wer,
                    "werstd": werstd,
                    "cer": cer,
                    "cerstd": cerstd,
                }
    return asrhypo_model2test2scores


def format_pval(pval):
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return ""


def asr_correlations(
        asrhypo_model2test2scores,
        asr2text2test2scores, text2test2scores):
    test2intent_scores = {}
    test2intent_deltas = {}
    test2metric2asr = {}
    test2text2intent_scores = {}
    test2text2intent_deltas = {}
    test2text2metric2asr = {}
    metrics = ["wer", "werstd", "cer", "cerstd"]
    for asr_model in asr_models:
        # Focusing on the train-on-gold, eval-on-ASR experiments for better
        # comparability
        for text_model in text_models_text:
            for test_set in test_sets_audio:
                if not test_set:
                    continue
                scores_asr = asr2text2test2scores[asr_model][text_model][test_set]
                if test_set == speechmas_de:
                    scores_text = text2test2scores[text_model][mas_de]
                else:
                    scores_text = text2test2scores[text_model][test_set]
                asr_results = asrhypo_model2test2scores[asr_model][test_set]
                if test_set not in test2metric2asr:
                    test2metric2asr[test_set] = {}
                    for metric in metrics:
                        if asr_results[metric]:
                            test2metric2asr[test_set][metric] = []
                    test2intent_scores[test_set] = []
                    test2intent_deltas[test_set] = []
                    test2text2intent_scores[test_set] = {}
                    test2text2intent_deltas[test_set] = {}
                    test2text2metric2asr[test_set] = {}
                if text_model not in test2text2intent_scores[test_set]:
                    test2text2intent_scores[test_set][text_model] = []
                    test2text2intent_deltas[test_set][text_model] = []
                    test2text2metric2asr[test_set][text_model] = {}
                    for metric in metrics:
                        if asr_results[metric]:
                            test2text2metric2asr[test_set][text_model][metric] = []
                for score_asr, score_text in zip(scores_asr, scores_text):
                    test2intent_scores[test_set].append(score_asr)
                    test2intent_deltas[test_set].append(score_asr - score_text)
                    test2text2intent_scores[test_set][text_model].append(score_asr)
                    test2text2intent_deltas[test_set][text_model].append(score_asr - score_text)
                    for metric in metrics:
                        if asr_results[metric]:
                            test2metric2asr[test_set][metric].append(asr_results[metric])
                            test2text2metric2asr[test_set][text_model][metric].append(asr_results[metric])
    with open("../scores/intents/correlations_intents_cascaded_asr.tsv", "w") as f:
        f.write("Test set\tText model")
        for metric in metrics:
            f.write(f"\t{metric}_pearson\t{metric}_pearson_pval")
            f.write(f"\t{metric}_spearman\t{metric}_spearman_pval")
        f.write("\n")
        for test_set in test2intent_scores:
            f.write(test_set + "\tAll text models (not distinguishing)")
            for metric in metrics:
                if metric in test2metric2asr[test_set]:
                    pearson, pearson_pval = pearsonr(
                        test2intent_scores[test_set], test2metric2asr[test_set][metric])
                    f.write(f"\t{pearson}\t{format_pval(pearson_pval)}")
                    spearman, spearman_pval = spearmanr(
                        test2intent_scores[test_set], test2metric2asr[test_set][metric])
                    f.write(f"\t{spearman}\t{format_pval(spearman_pval)}")
                else:
                    f.write("\t\t\t\t")
            f.write("\n")
            for text_model in test2text2intent_scores[test_set]:
                f.write(test_set + "\t" + model2pretty[text_model])
                for metric in metrics:
                    if metric in test2text2metric2asr[test_set][text_model]:
                        pearson, pearson_pval = pearsonr(
                            test2text2intent_scores[test_set][text_model],
                            test2text2metric2asr[test_set][text_model][metric])
                        f.write(f"\t{pearson}\t{format_pval(pearson_pval)}")
                        spearman, spearman_pval = spearmanr(
                            test2text2intent_scores[test_set][text_model],
                            test2text2metric2asr[test_set][text_model][metric])
                        f.write(f"\t{spearman}\t{format_pval(spearman_pval)}")
                    else:
                        f.write("\t\t\t\t")
                f.write("\n")
    with open("../scores/intents/correlations_intents_cascaded-text-deltas_asr.tsv", "w") as f:
        f.write("Test set\tText model")
        for metric in metrics:
            f.write(f"\t{metric}_pearson\t{metric}_pearson_pval")
            f.write(f"\t{metric}_spearman\t{metric}_spearman_pval")
        f.write("\n")
        for test_set in test2intent_deltas:
            f.write(test_set + "\tAll text models (not distinguishing)")
            for metric in metrics:
                if metric in test2metric2asr[test_set]:
                    pearson, pearson_pval = pearsonr(
                        test2intent_deltas[test_set], test2metric2asr[test_set][metric])
                    f.write(f"\t{pearson}\t{format_pval(pearson_pval)}")
                    spearman, spearman_pval = spearmanr(
                        test2intent_deltas[test_set], test2metric2asr[test_set][metric])
                    f.write(f"\t{spearman}\t{format_pval(spearman_pval)}")
                else:
                    f.write("\t\t\t\t")
            f.write("\n")
            for text_model in test2text2intent_deltas[test_set]:
                f.write(test_set + "\t" + model2pretty[text_model])
                for metric in metrics:
                    if metric in test2text2metric2asr[test_set][text_model]:
                        pearson, pearson_pval = pearsonr(
                            test2text2intent_deltas[test_set][text_model],
                            test2text2metric2asr[test_set][text_model][metric])
                        f.write(f"\t{pearson}\t{format_pval(pearson_pval)}")
                        spearman, spearman_pval = spearmanr(
                            test2text2intent_deltas[test_set][text_model],
                            test2text2metric2asr[test_set][text_model][metric])
                        f.write(f"\t{spearman}\t{format_pval(spearman_pval)}")
                    else:
                        f.write("\t\t\t\t")
                f.write("\n")


text_models_text = [  # Text models fine-tuned on gold text data
    "google-bert-bert-base-multilingual-cased",
    "microsoft-mdeberta-v3-base",
    "FacebookAI-xlm-roberta-base",
    "FacebookAI-xlm-roberta-large",
]
text_models_asr = []  # Text models fine-tuned on ASR'ed data
for text_model in ["google-bert-bert-base-multilingual-cased",
                   "microsoft-mdeberta-v3-base"]:
    for asr_model in ["AndrewMcDowell-wav2vec2-xls-r-300m-german-de",
                      "openai-whisper-large-v3", "openai-whisper-tiny"]:
        text_models_asr.append(text_model + "-" + asr_model)
text_models_all = text_models_text + text_models_asr
asr_models = [
    "AndrewMcDowell-wav2vec2-xls-r-300m-german-de",
    "AndrewMcDowell-wav2vec2-xls-r-1B-german",
    "facebook-mms-1b-all",
    "openai-whisper-tiny",
    "openai-whisper-base",
    "openai-whisper-small",
    "openai-whisper-medium",
    "openai-whisper-large-v3-turbo",
    "openai-whisper-large-v3",
]
speech_models = [
    "facebook-wav2vec2-xls-r-300m",
    "AndrewMcDowell-wav2vec2-xls-r-300m-german-de",
    "facebook-mms-300m",
    "openai-whisper-tiny",
    "openai-whisper-base",
    "openai-whisper-small",
    "openai-whisper-medium",
    "openai-whisper-large-v3",
    "utter-project-mHuBERT-147",
]
model2pretty = {
    "google-bert-bert-base-multilingual-cased": "mBERT",
    "microsoft-mdeberta-v3-base": "mDeBERTa",
    "FacebookAI-xlm-roberta-base": "XLM-R base",
    "FacebookAI-xlm-roberta-large": "XLM-R large",
    "facebook-wav2vec2-xls-r-300m": "XLS-R 300M",
    "AndrewMcDowell-wav2vec2-xls-r-300m-german-de": "XLS-R 300M DE",
    "AndrewMcDowell-wav2vec2-xls-r-1B-german": "XLS-R 1B DE",
    "facebook-mms-300m": "MMS 300M",
    "facebook-mms-1b-all": "MMS 1B",
    "openai-whisper-tiny": "Whisper tiny",
    "openai-whisper-base": "Whisper base",
    "openai-whisper-small": "Whisper small",
    "openai-whisper-medium": "Whisper medium",
    "openai-whisper-large-v3": "Whisper large-v3",
    "openai-whisper-large-v3-turbo": "Whisper large-v3-turbo",
    "utter-project-mHuBERT-147": "mHuBERT",
}
for text_model in ["google-bert-bert-base-multilingual-cased",
                   "microsoft-mdeberta-v3-base"]:
    text_pretty = model2pretty[text_model]
    for asr_model in ["AndrewMcDowell-wav2vec2-xls-r-300m-german-de",
                      "openai-whisper-large-v3", "openai-whisper-tiny"]:
        asr_pretty = model2pretty[asr_model]
        model2pretty[text_model + "-" + asr_model] = f"{text_pretty} ({asr_pretty})"

mas_de = "massive_deu_test_mapped_matched"
mas_ba = "massive_deba_resplit_test"
xsid_de = "xsid_de_test"
xsid_ba = "xsid_de-ba_test"
test_sets_text = [
    mas_de, mas_ba, xsid_de, xsid_ba,
]
speechmas_de = "speechmassive_de_test_matched"
test_sets_audio = [
    speechmas_de, None,
    xsid_de, xsid_ba,
]
test_sets_dialect = [
    "massive_deba_resplit_test", "xsid_de-ba_test",
]
seeds = ["1234", "2345", "3456"]


setups_and_scores = get_results("acc")
write_detailed_table("acc", *setups_and_scores)
write_table_by_setup("acc", *setups_and_scores)
write_deltas_setups("acc", *setups_and_scores)

asreval_model2test2scores = get_asr_results()
asr_correlations(
    asreval_model2test2scores,
    setups_and_scores[2], setups_and_scores[0])

setups_and_scores = get_results("f1")
write_detailed_table("f1", *setups_and_scores)
