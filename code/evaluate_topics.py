from sklearn.metrics import f1_score, accuracy_score
from glob import glob
from statistics import stdev
from scipy.stats import pearsonr, spearmanr


dialects_ch = ["ag", "be", "bs", "gr", "lu", "sg", "vs", "zh"]


def score(filename):
    golds, preds = [], []
    with open(filename) as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            cells = line.strip().split("\t")
            gold = cells[-2]
            pred = cells[-1]
            golds.append(gold)
            preds.append(pred)
    return (100 * accuracy_score(golds, preds),
            100 * f1_score(golds, preds, average="macro"))


def score_by_dialect(filename):
    dial2golds = {dial: [] for dial in dialects_ch}
    dial2preds = {dial: [] for dial in dialects_ch}
    with open(filename) as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            _, dial, _, gold, pred = line.strip().split("\t")
            dial2golds[dial].append(gold)
            dial2preds[dial].append(pred)
    dial2acc, dial2f1 = {}, {}
    for dial in dialects_ch:
        golds = dial2golds[dial]
        preds = dial2preds[dial]
        dial2acc[dial] = 100 * accuracy_score(golds, preds)
        dial2f1[dial] = 100 * f1_score(golds, preds, average="macro")
    return dial2acc, dial2f1


def get_results(metric):
    text2test2scores = {}
    asr2text2test2scores = {}
    for text_model in text_models:
        for test_set in test_sets:
            for seed in seeds:
                try:
                    filename = glob(f"../predictions/topics/text+text/{text_model}*{seed}*{test_set}.tsv")[0]
                    acc, f1 = score(filename)
                    if text_model not in text2test2scores:
                        text2test2scores[text_model] = {}
                    if test_set not in text2test2scores[text_model]:
                        text2test2scores[text_model][test_set] = []
                    text2test2scores[text_model][test_set].append(
                        acc if metric == "acc" else f1)
                    if "_dial" in test_set:
                        dial2acc, dial2f1 = score_by_dialect(filename)
                        for dial in dialects_ch:
                            test_set_dial = test_set + "-" + dial
                            score_dial = dial2acc[dial] if metric == "acc" else dial2f1[dial]
                            if test_set_dial not in text2test2scores[text_model]:
                                text2test2scores[text_model][test_set_dial] = []
                            text2test2scores[text_model][test_set_dial].append(score_dial)
                except IndexError:
                    pass
        for asr_model in asr_models:
            for test_set in test_sets:
                if not test_set:
                    continue
                for seed in seeds:
                    try:
                        filename = glob(f"../predictions/topics/text+asr/{text_model}*{seed}*{asr_model}*{test_set}.tsv")[0]
                        acc, f1 = score(filename)
                        if asr_model not in asr2text2test2scores:
                            asr2text2test2scores[asr_model] = {}
                        if text_model not in asr2text2test2scores[asr_model]:
                            asr2text2test2scores[asr_model][text_model] = {}
                        if test_set not in asr2text2test2scores[asr_model][text_model]:
                            asr2text2test2scores[asr_model][text_model][test_set] = []
                        asr2text2test2scores[asr_model][text_model][test_set].append(
                            acc if metric == "acc" else f1)
                        if "_dial" in test_set:
                            dial2acc, dial2f1 = score_by_dialect(filename)
                            for dial in dialects_ch:
                                test_set_dial = test_set + "-" + dial
                                score_dial = dial2acc[dial] if metric == "acc" else dial2f1[dial]
                                if test_set_dial not in asr2text2test2scores[asr_model][text_model]:
                                    asr2text2test2scores[asr_model][text_model][test_set_dial] = []
                                asr2text2test2scores[asr_model][text_model][test_set_dial].append(score_dial)
                    except IndexError:
                        pass

    with open(f"../scores/topics/topic_scores_{metric}_unaggregated.tsv",
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
                text2test2scores_avg[text_model][test] = (
                    sum(scores) / len(scores), stdev(scores))
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
                    asr2text2test2scores_avg[asr_model][text_model][test] = (
                        sum(scores) / len(scores), stdev(scores))
                    f.write(f"{text_model}\tgold text\tASR {asr_model}\t{test}")
                    for score_ in scores:
                        f.write("\t" + str(score_))
                    f.write("\n")

    return (text2test2scores, text2test2scores_avg,
            asr2text2test2scores, asr2text2test2scores_avg)


def write_detailed_table(
        metric, text2test2scores, text2test2scores_avg,
        asr2text2test2scores, asr2text2test2scores_avg):
    with open(f"../scores/topics/topic_scores_{metric}-avg.tsv", "w") as f:
        f.write("Model\tTrain\tTest processing\tN\t")
        f.write("Swissdial de (mean)\tSwissdial de (stdev)\tSwissdial de (delta gold-asr)\t")
        f.write("Swissdial dial_all (mean)\tSwissdial dial_all (stdev)\t")
        f.write("Swissdial dial_all (delta gold-asr)\tSwissdial dial_all (delta deu-dial)")
        for dial in dialects_ch:
            f.write(f"\tSwissdial {dial} (mean)\tSwissdial {dial} (stdev)")
            f.write(f"\tSwissdial {dial} (delta gold-asr)\tSwissdial {dial} (delta deu-dial)")
        f.write("\n")
        for text_model in text_models:
            f.write(model2pretty[text_model] + "\tgold text\tgold text")
            f.write("\t" + str(len(text2test2scores[text_model][test_de])))

            # Swissdial de
            f.write("\t" + str(text2test2scores_avg[text_model][test_de][0]))
            f.write("\t" + str(text2test2scores_avg[text_model][test_de][1]))
            f.write("\t")  # delta to text

            # Swissdial all dialects
            f.write("\t" + str(text2test2scores_avg[text_model][test_gsw][0]))
            f.write("\t" + str(text2test2scores_avg[text_model][test_gsw][1]))
            f.write("\t")  # delta to text
            ## delta to std
            f.write("\t" + str(text2test2scores_avg[text_model][test_gsw][0] - text2test2scores_avg[text_model][test_de][0]))

            # Swissdial per dialect
            for dial in dialects_ch:
                test_dial = test_gsw + "-" + dial
                f.write("\t" + str(text2test2scores_avg[text_model][test_dial][0]))
                f.write("\t" + str(text2test2scores_avg[text_model][test_dial][1]))
                f.write("\t")  # delta to text
                ## delta to std
                f.write("\t" + str(text2test2scores_avg[text_model][test_dial][0] - text2test2scores_avg[text_model][test_de][0]))
            f.write("\n")

            for asr_model in asr_models:
                f.write(model2pretty[text_model])
                f.write("\tgold text\tASR " + model2pretty[asr_model])
                f.write("\t" + str(len(asr2text2test2scores[asr_model][text_model][test_de])))

                # Swissdial de
                f.write("\t")
                f.write("\t")
                f.write("\t")

                # Swissdial all dialects
                f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][test_gsw][0]))
                f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][test_gsw][1]))
                ## delta to text
                f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][test_gsw][0] - text2test2scores_avg[text_model][test_gsw][0]))
                ## delta to std
                f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][test_gsw][0] - asr2text2test2scores_avg[asr_model][text_model][test_de][0]))

                # Swissdial per dialect
                for dial in dialects_ch:
                    test_dial = test_gsw + "-" + dial
                    f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][test_dial][0]))
                    f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][test_dial][1]))
                    ## delta to text
                    f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][test_dial][0] - text2test2scores_avg[text_model][test_dial][0]))
                    ## delta to std
                    f.write("\t" + str(asr2text2test2scores_avg[asr_model][text_model][test_dial][0] - asr2text2test2scores_avg[asr_model][text_model][test_de][0]))
                f.write("\n")


def write_table_by_setup(
        metric, text2test2scores, text2test2scores_avg,
        asr2text2test2scores, asr2text2test2scores_avg):
    with open(f"../scores/topics/topic_scores_{metric}-by-setup-type.tsv", "w") as f:
        f.write("Model\tN\t")
        f.write("Swissdial de (mean)\tSwissdial de (stdev)\t")
        f.write("Swissdial dial (mean)\tSwissdial dial (stdev)\tSwissdial dial (delta to std)")
        for dial in dialects_ch:
            f.write(f"\tSwissdial {dial} (mean)\tSwissdial {dial} (stdev)")
        f.write("\n")

        # Text-only
        rows = []
        all_de, all_dial = [], []
        dial2all = {}
        for text_model in text_models:
            row = model2pretty[text_model]
            row += "\t" + str(len(text2test2scores[text_model][test_de]))
            # Swissdial de
            all_de += text2test2scores[text_model][test_de].copy()
            row += "\t" + str(text2test2scores_avg[text_model][test_de][0])
            row += "\t" + str(text2test2scores_avg[text_model][test_de][1])
            # Swissdial all dialects
            all_dial += text2test2scores[text_model][test_gsw].copy()
            row += "\t" + str(text2test2scores_avg[text_model][test_gsw][0])
            row += "\t" + str(text2test2scores_avg[text_model][test_gsw][1])
            row += "\t" + str(text2test2scores_avg[text_model][test_gsw][0] - text2test2scores_avg[text_model][test_de][0])
            # Swissdial per dialect
            for dial in dialects_ch:
                test_dial = test_gsw + "-" + dial
                canton = text2test2scores[text_model][test_dial].copy()
                try:
                    dial2all[dial] += canton
                except KeyError:
                    dial2all[dial] = canton
                row += "\t" + str(text2test2scores_avg[text_model][test_dial][0])
                row += "\t" + str(text2test2scores_avg[text_model][test_dial][1])
            row += "\n"
            rows.append(row)
        f.write("*Text-only*")
        f.write("\t" + str(len(all_de)))
        f.write("\t" + str(sum(all_de) / len(all_de)))
        f.write("\t" + str(stdev(all_de)))
        f.write("\t" + str(sum(all_dial) / len(all_dial)))
        f.write("\t" + str(stdev(all_dial)))
        f.write("\t")
        for dial in dialects_ch:
            all_canton = dial2all[dial]
            f.write("\t" + str(sum(all_canton) / len(all_canton)))
            f.write("\t" + str(stdev(all_canton)))
        f.write("\n")
        for row in rows:
            f.write(row)

        # Cascaded
        rows_by_asr = []
        all_de, all_dial = [], []
        dial2all = {}
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
            row += "\t" + str(len(test2scores[test_de]))
            # Swissdial de
            all_de += test2scores[test_de].copy()
            test_de_mean = sum(test2scores[test_de]) / len(test2scores[test_de])
            row += "\t" + str(test_de_mean)
            row += "\t" + str(stdev(test2scores[test_de]))
            # Swissdial all dialects
            all_dial += test2scores[test_gsw].copy()
            test_gsw_mean = sum(test2scores[test_gsw]) / len(test2scores[test_gsw])
            row += "\t" + str(test_gsw_mean)
            row += "\t" + str(stdev(test2scores[test_gsw]))
            row += "\t" + str(test_gsw_mean - test_de_mean)
            # Swissdial per dialect
            for dial in dialects_ch:
                test_dial = test_gsw + "-" + dial
                canton = test2scores[test_dial].copy()
                try:
                    dial2all[dial] += canton
                except KeyError:
                    dial2all[dial] = canton
                row += "\t" + str(sum(canton) / len(canton))
                row += "\t" + str(stdev(canton))
            row += "\n"
            rows_by_asr.append(row)
        row_total_avg = "\t" + str(len(all_de))
        row_total_avg += "\t" + str(sum(all_de) / len(all_de))
        row_total_avg += "\t" + str(stdev(all_de))
        row_total_avg += "\t" + str(sum(all_dial) / len(all_dial))
        row_total_avg += "\t" + str(stdev(all_dial))
        row_total_avg += "\t"
        for dial in dialects_ch:
            all_canton = dial2all[dial]
            row_total_avg += "\t" + str(sum(all_canton) / len(all_canton))
            row_total_avg += "\t" + str(stdev(all_canton))
        row_total_avg += "\n"
        f.write("*Cascaded (averaged over ASR models)*")
        f.write(row_total_avg)
        for text_model in text_models:
            f.write(model2pretty[text_model])
            scores_test_de = cascaded_text2test2scores[text_model][test_de]
            f.write("\t" + str(len(scores_test_de)))
            scores_test_de_mean = sum(scores_test_de) / len(scores_test_de)
            f.write("\t" + str(scores_test_de_mean))
            f.write("\t" + str(stdev(scores_test_de)))
            scores_test_gsw = cascaded_text2test2scores[text_model][test_gsw]
            scores_test_gsw_mean = sum(scores_test_gsw) / len(scores_test_gsw)
            f.write("\t" + str(scores_test_gsw_mean))
            f.write("\t" + str(stdev(scores_test_gsw)))
            f.write("\t" + str(scores_test_gsw_mean - scores_test_de_mean))
            # Swissdial per dialect
            for dial in dialects_ch:
                scores_test_canton = cascaded_text2test2scores[text_model][test_gsw + "-" + dial]
                f.write("\t" + str(sum(scores_test_canton) / len(scores_test_canton)))
                f.write("\t" + str(stdev(scores_test_canton)))
            f.write("\n")
        f.write("*Cascaded (averaged over text models)*")
        f.write(row_total_avg)
        for row in rows_by_asr:
            f.write(row)


def write_deltas_setups(
        metric, text2test2scores, text2test2scores_avg,
        asr2text2test2scores, asr2text2test2scores_avg):
    with open(f"../scores/topics/topic_scores_{metric}-deltas-cascaded-text.tsv", "w") as f:
        f.write("Model\t")
        f.write("Swissdial de (mean)\tSwissdial de (stdev)")
        f.write("Swissdial dial (mean)\tSwissdial dial (stdev)\n")
        for asr_model in asr_models:
            test2diffs = {}
            for text_model in text_models:
                for test_set in test_sets:
                    asr_avg, _ = asr2text2test2scores_avg[asr_model][text_model][test_set]
                    text_avg, _ = text2test2scores_avg[text_model][test_set]
                    delta = asr_avg - text_avg
                    try:
                        test2diffs[test_set].append(delta)
                    except KeyError:
                        test2diffs[test_set] = [delta]
            f.write(model2pretty[asr_model])
            for test_set in test_sets:
                f.write("\t" + str(sum(test2diffs[test_set]) / len(test2diffs[test_set])))
                f.write("\t" + str(stdev(test2diffs[test_set])))
            f.write("\n")


def get_asr_results():
    asrhypo_model2test2scores = {}
    for asr_model in asr_models:
        asrhypo_model2test2scores[asr_model] = {}
        with open(f"../scores/asr/topics/{asr_model}.tsv") as f:
            for line in f:
                cells = line.split("\t")
                if not cells[0] in test_sets:
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
        for text_model in text_models:
            for test_set in test_sets:
                scores_asr = asr2text2test2scores[asr_model][text_model][test_set]
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
    with open("../scores/topics/correlations_intents_cascaded_asr.tsv", "w") as f:
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
    with open("../scores/topics/correlations_intents_cascaded-text-deltas_asr.tsv", "w") as f:
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


text_models = [
    "google-bert-bert-base-multilingual",
    "microsoft-mdeberta-v3-base",
    "FacebookAI-xlm-roberta-base",
    "FacebookAI-xlm-roberta-large",
]
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
model2pretty = {
    "google-bert-bert-base-multilingual": "mBERT",
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
test_de = "swissdial_test_deu"
test_gsw = "swissdial_test_dial"
test_sets = [
    test_de,
    test_gsw,
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
