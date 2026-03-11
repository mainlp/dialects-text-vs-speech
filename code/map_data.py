#!/usr/bin/env python3

import json
from collections import Counter


path_to_xsid_audio = ""
path_to_slurp_audio = ""


# Read in the translated MASSIVE subset (which was already manually)
# re-annotated with xSID labels.


def read_nalibasid(filename, nalibasid2details, eng2nalibasid):
    with open(filename) as f:
        intent_xsid = None
        id_ = None
        text_en = None
        text_ba = None
        counter = 0
        for line in f:
            line = line.strip()
            if not line:
                if text_en in eng2nalibasid:
                    eng2nalibasid[text_en] += [id_]
                    print("repetition")
                    print(text_en)
                    print(eng2nalibasid[text_en])
                else:
                    eng2nalibasid[text_en] = [id_]
                nalibasid2details[id_] = [intent_xsid, text_ba, text_en]
                if text_ba is None or text_en is None or intent_xsid is None or id_ is None:
                    print("Information missing (skipping sentence)")
                    print("ID", id_)
                    print("Label", intent_xsid)
                    print("BAR", text_ba)
                    print("ENG", text_en)
                intent_xsid = None
                text_en = None
                text_ba = None
                counter += 1
                continue
            if line.startswith("# id:"):
                id_ = line[len("# id:"):].strip()
            elif line.startswith("# intent:"):
                intent_xsid = line[len("# intent:"):].strip()
            elif line.startswith("# text-en:"):
                text_en = line[len("# text-en:"):].lower().strip()
            elif line.startswith("# text:"):
                text_ba = line[len("# text:"):].lower().strip()
            elif line.startswith("+# text:"):  # typo in one entry
                text_ba = line[len("+# text:"):].lower().strip()
    if text_en:
        if text_en in eng2nalibasid:
            eng2nalibasid[text_en] += [id_]
            print("repetition")
            print(text_en)
            print(eng2nalibasid[text_en])
        else:
            eng2nalibasid[text_en] = [id_]
        nalibasid2details[id_] = [intent_xsid, text_ba, text_en]
        counter += 1
    print(counter)
    return nalibasid2details, eng2nalibasid


nalibasid2details = {}
eng2nalibasid = {}
nalibasid_alt_translations = {}
nalibasid2details, eng2nalibasid = read_nalibasid(
    "../data/intents/nalibasid/de-ba.MAS.valid.conll",
    nalibasid2details, eng2nalibasid)
nalibasid2details, eng2nalibasid = read_nalibasid(
    "../data/intents/nalibasid/de-ba.MAS.test.conll",
    nalibasid2details, eng2nalibasid)
print(len(nalibasid2details), len(eng2nalibasid))


# Map the MASSIVE data according to the manual labels, and map the
# labels of the remaining items where possible.

nalibasid2massive = {}
massive2details = {}

skipped = 0
with open("../data/intents/massive1.1/data/en-US.jsonl") as f:
    for line in f:
        entries = json.loads(line)
        id_ = int(entries["id"])
        bar_id = None
        text = entries["utt"]
        intent = entries["intent"]
        annot = entries["annot_utt"]
        if text in eng2nalibasid:
            nalibasid_ids = eng2nalibasid[text]
            if len(nalibasid_ids) > 1:
                for bar_id_poss in nalibasid_ids:
                    if bar_id_poss not in nalibasid2massive:
                        bar_id = bar_id_poss
                        break
            else:
                bar_id = nalibasid_ids[0]
            if not bar_id:
                # shouldn't happen
                print("couldn't find a Bavarian ID")
                print(text)
            nalibasid2massive[bar_id] = id_
            xsid_intent, text_bar, _ = nalibasid2details[bar_id]
            massive2details[id_] = [xsid_intent, text_bar, text]

        elif intent == "play_music" or intent == "play_radio":
            massive2details[id_] = ["PlayMusic", None, text]
        elif intent == "alarm_remove":
            massive2details[id_] = ["alarm/cancel_alarm", None, text]
        elif intent == "alarm_set":
            massive2details[id_] = ["alarm/set_alarm", None, text]
        elif intent == "alarm_query":
            massive2details[id_] = ["alarm/show_alarms", None, text]
        elif intent == "weather_query":
            massive2details[id_] = ["weather/find", None, text]
        elif intent == "recommendation_movies" and (
                "place_name" in annot or "business_type" in annot):
            massive2details[id_] = ["SearchScreeningEvent", None, text]
        elif intent == "calendar_set" and (
                "remind" in text or "notif" in text):
            massive2details[id_] = ["reminder/set_reminder", None, text]
        elif intent == "calendar_query" and "remind" in text:
            massive2details[id_] = ["reminder/show_reminders", None, text]
        elif intent == "calendar_remove" and (
                "remind" in text or "notif" in text):
            massive2details[id_] = ["reminder/cancel_reminder", None, text]
        else:
            skipped += 1

print("Mapped", len(massive2details))
print("Skipped", skipped)


def print_intents(intents):
    counter = Counter(intents)
    for intent in ["AddToPlaylist", "PlayMusic",
                   "RateBook", "BookRestaurant",
                   "SearchScreeningEvent", "SearchCreativeWork",
                   "alarm/cancel_alarm", "alarm/modify_alarm",
                   "alarm/set_alarm", "alarm/show_alarms",
                   "alarm/snooze_alarm", "alarm/time_left_on_alarm",
                   "reminder/cancel_reminder", "reminder/set_reminder",
                   "reminder/show_reminders", "weather/find"]:
        print(intent, counter[intent])
    # For copy-pasting into a spreadsheet:
    for intent in ["AddToPlaylist", "PlayMusic",
                   "RateBook", "BookRestaurant",
                   "SearchScreeningEvent", "SearchCreativeWork",
                   "alarm/cancel_alarm", "alarm/modify_alarm",
                   "alarm/set_alarm", "alarm/show_alarms",
                   "alarm/snooze_alarm", "alarm/time_left_on_alarm",
                   "reminder/cancel_reminder", "reminder/set_reminder",
                   "reminder/show_reminders", "weather/find"]:
        print(counter[intent])
    print()


# Now going through the train split of MASSIVE
# (this info isn't given in the MASSIVE files, but it is identical
# in ITALIC and SLURP) in order to
# figure out which labels are decently represented in which splits

massive_id2split = {}
italic_train_intents = []
with open("../data/intents/italic/massive_train.json") as f_in:
    for line in f_in:
        entries = json.loads(line)
        id_ = entries["id"]
        if id_ in massive2details:
            massive_id2split[id_] = "train"
            intent = massive2details[id_][0]
            italic_train_intents.append(intent)

print("ITALIC train (= MASSIVE train)")
print_intents(italic_train_intents)

intents_final = []
counter = Counter(italic_train_intents)
for intent in set(italic_train_intents):
    if counter[intent] >= 10:
        intents_final.append(intent)

print("Final intent selection")
print(len(intents_final))
print(intents_final)
print()


# Going through the train/dev/test splits of MASSIVE
# (this info isn't given in the MASSIVE files, but it is identical
# in ITALIC and SLURP)
# so we can later split the MASSIVE translations
# (and the Bavarian translations) accordingly
massive_id2split = {}


def map_italic(split_in, split_out, massive_id2split, intents_final):
    intents = []
    with open(f"../data/intents/italic/massive_{split_in}.json") as f_in:
        for line in f_in:
            entries = json.loads(line)
            id_ = entries["id"]
            if id_ in massive2details:
                massive_id2split[id_] = split_out
                intent = massive2details[id_][0]
                if intent not in intents_final:
                    continue
                intents.append(intent)
    print(f"ITALIC {split_out} (= MASSIVE {split_out})")
    print_intents(intents)


map_italic("train", "train", massive_id2split, intents_final)
map_italic("validation", "dev", massive_id2split, intents_final)
map_italic("test", "test", massive_id2split, intents_final)


# Split and map MASSIVE

def remap_massive(lang_code_in, lang_code_out, massive_id2split,
                  massive2details, intents_final):
    with open(f"../data/intents/massive_{lang_code_out}_train_mapped.tsv", "w") as f_out_train:
        f_out_train.write("ID\tText\tIntent\n")
        with open(f"../data/intents/massive_{lang_code_out}_dev_mapped.tsv", "w") as f_out_dev:
            f_out_dev.write("ID\tText\tIntent\n")
            with open(f"../data/intents/massive_{lang_code_out}_test_mapped.tsv", "w") as f_out_test:
                f_out_test.write("ID\tText\tIntent\n")
                with open(f"../data/intents/massive1.1/data/{lang_code_in}.jsonl") as f_in:
                    for line in f_in:
                        entries = json.loads(line)
                        id_ = int(entries["id"])
                        if id_ in massive2details:
                            intent = massive2details[id_][0]
                            if intent not in intents_final:
                                continue
                            if massive_id2split[id_] == "train":
                                f_out_train.write(f"{id_}\t{entries['utt']}\t{intent}\n")
                            elif massive_id2split[id_] == "dev":
                                f_out_dev.write(f"{id_}\t{entries['utt']}\t{intent}\n")
                            elif massive_id2split[id_] == "test":
                                f_out_test.write(f"{id_}\t{entries['utt']}\t{intent}\n")
                            else:
                                print("Didn't find a split (skipping)")
                                print(id_)
                                print(entries["utt"])


remap_massive("de-DE", "deu", massive_id2split, massive2details, intents_final)


# Re-split the Bavarian translations of MASSIVE:

def remap_nalibasid(in_file, out_flag, nalibasid_train_intents,
                    nalibasid_dev_intents, nalibasid_test_intents,
                    nalibasid2massive, massive_id2split):
    with open("../data/intents/massive_deba_resplit_train.tsv", out_flag) as f_out_train:
        if out_flag == "w":
            f_out_train.write("ID\tText\tIntent\tNaLiBaSid ID\n")
        with open("../data/intents/massive_deba_resplit_dev.tsv", out_flag) as f_out_dev:
            if out_flag == "w":
                f_out_dev.write("ID\tText\tIntent\tNaLiBaSid ID\n")
            with open("../data/intents/massive_deba_resplit_test.tsv", out_flag) as f_out_test:
                if out_flag == "w":
                    f_out_test.write("ID\tText\tIntent\tNaLiBaSid ID\n")
                with open(in_file) as f_in:
                    id_ = None
                    intent = None
                    text_ba = None
                    text_en = None
                    for line in f_in:
                        line = line.strip()
                        if not line:
                            if id_ not in nalibasid2massive:
                                print("Not in dict (skipping)")
                                print(id_)
                                print(text_ba)
                                print(text_en)
                                id_ = None
                                intent = None
                                text_ba = None
                                text_en = None
                                continue
                            if intent not in intents_final:
                                id_ = None
                                intent = None
                                text_ba = None
                                text_en = None
                                continue
                            massive_id = nalibasid2massive[id_]
                            try:
                                split = massive_id2split[massive_id]
                            except KeyError:
                                print("Duplicate in NaLiBaSID that isn't a duplicate in MASSIVE (skipping)")
                                print(id_, massive_id)
                                print(text_ba)
                                print(text_en)
                                print(eng2nalibasid[text_en])
                            line = f"{massive_id}\t{text_ba}\t{intent}\t{id_}\n"
                            if split == "train":
                                f_out_train.write(line)
                                nalibasid_train_intents.append(intent)
                            elif split == "dev":
                                f_out_dev.write(line)
                                nalibasid_dev_intents.append(intent)
                            elif split == "test":
                                f_out_test.write(line)
                                nalibasid_test_intents.append(intent)
                            else:
                                print("Couldn't find sentence (skipping)")
                                print(massive_id)
                                print(text_ba)
                                print(text_en)
                            id_ = None
                            intent = None
                            text_ba = None
                            text_en = None
                            continue
                        if line.startswith("# id: "):
                            id_ = line[len("# id: "):]
                        elif line.startswith("# intent: "):
                            intent = line[len("# intent: "):]
                        elif line.startswith("# text: "):
                            text_ba = line[len("# text: "):]
                        elif line.startswith("# text-en: "):
                            text_en = line[len("# text-en: "):]
    return nalibasid_train_intents, nalibasid_dev_intents, nalibasid_test_intents

nalibasid_train_intents = []
nalibasid_dev_intents = []
nalibasid_test_intents = []
nalibasid_train_intents, nalibasid_dev_intents, nalibasid_test_intents = remap_nalibasid(
    "../data/intents/nalibasid/de-ba.MAS.valid.conll", "w",
    nalibasid_train_intents,
    nalibasid_dev_intents, nalibasid_test_intents,
    nalibasid2massive, massive_id2split)
nalibasid_train_intents, nalibasid_dev_intents, nalibasid_test_intents = remap_nalibasid(
    "../data/intents/nalibasid/de-ba.MAS.test.conll", "a",
    nalibasid_train_intents,
    nalibasid_dev_intents, nalibasid_test_intents,
    nalibasid2massive, massive_id2split)

print()
print("MASSIVE:de-ba resplit train")
print_intents(nalibasid_train_intents)

print("MASSIVE:de-ba resplit dev")
print_intents(nalibasid_dev_intents)

print("MASSIVE:de-ba resplit test")
print_intents(nalibasid_test_intents)


# Convert xSID to TSV and remove entries with rare labels

def reformat_xsid(lang, split_in, split_out, intents_final, sfx=""):
    intents = []
    with open(f"../data/intents/xsid_{lang}_{split_out}.tsv", "w") as f_out:
        if lang == "de-ba" or (lang == "de" and split_in in ["valid", "test"]):
            f_out.write("ID\tText\tIntent\tAudio\n")
        else:
            f_out.write("ID\tText\tIntent\n")
        with open(f"../data/intents/xsid/data/xSID-0.6/{lang}.{split_in}.conll{sfx}") as f_in:
            intent = None
            text = None
            id_ = None
            n_sentences = 0
            for line in f_in:
                line = line.strip()
                if not line:
                    if split_out == "train" or lang == "en":
                        n_sentences += 1
                        id_ = n_sentences
                    if intent in intents_final:
                        f_out.write(f"{id_}\t{text}\t{intent}")
                        if lang == "de" and split_in in ["valid", "test"]:
                            f_out.write(f"\t{path_to_xsid_audio}/04_de_{split_in}_wav_mulaw/04_de{id_:03d}_mulaw.wav")
                        elif lang == "de-ba":
                            f_out.write(f"\t{path_to_xsid_audio}/04_de-ba_{split_in}_wav_mulaw/04_ba{id_:03d}_mulaw.wav")
                        f_out.write("\n")
                        intents.append(intent)
                    intent = None
                    id_ = None
                    text = None
                    continue
                if line.startswith("# id ="):
                    id_ = int(line[len("# id ="):].strip())
                elif line.startswith("# id:"):
                    id_ = int(line[len("# id:"):].strip())
                elif line.startswith("# intent ="):
                    intent = line[len("# intent ="):].strip()
                elif line.startswith("# intent:"):
                    intent = line[len("# intent:"):].strip()
                elif line.startswith("# text ="):
                    text = line[len("# text ="):].lower().strip()
                elif line.startswith("# text:"):
                    text = line[len("# text:"):].lower().strip()
        if split_out == "train" or lang == "en":
            n_sentences += 1
            id_ = n_sentences
        if intent and intent in intents_final:
            f_out.write(f"{id_}\t{text}\t{intent}")
            if lang == "de" and split_in in ["valid", "test"]:
                f_out.write(f"\t{path_to_xsid_audio}/04_de_{split_in}_wav_mulaw/04_de{id_:03d}_mulaw.wav")
            elif lang == "de-ba":
                f_out.write(f"\t{path_to_xsid_audio}/04_de-ba_{split_in}_wav_mulaw/04_ba{id_:03d}_mulaw.wav")
            f_out.write("\n")
    print(f"xSID {lang} {split_out}")
    print_intents(intents)


reformat_xsid("de", "projectedTrain", "train", intents_final, ".fixed")
reformat_xsid("de", "valid", "dev", intents_final)
reformat_xsid("de", "test", "test", intents_final)
reformat_xsid("de-ba", "valid", "dev", intents_final)
reformat_xsid("de-ba", "test", "test", intents_final)
reformat_xsid("gsw", "valid", "dev", intents_final)
reformat_xsid("gsw", "test", "test", intents_final)
