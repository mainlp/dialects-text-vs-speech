import os
import pandas as pd
import argparse
from sklearn.utils import shuffle

parser = argparse.ArgumentParser(
    description="Convert SwissDial v1.0 data to a structured format.")
parser.add_argument(
    "--data_path", type=str, help="Path to the input data file")
parser.add_argument(
    "--seed", type=int, default=1234, help="Random seed for reproducibility")

args = parser.parse_args()

dialects = ["de", "ch_ag", "ch_be", "ch_bs", "ch_gr",
            "ch_lu", "ch_sg", "ch_vs", "ch_zh"]

topic_mapping = {"tiere-pflanzen-landwirtschaft": "Animals/farming",
                 "kultur": "Culture",
                 "erde-weltraum": "Earth/Space",
                 "wirtschaft": "Economics",
                 "internationale-politik": "International politics",
                 "medizin": "Medicine",
                 "srf_meteo": "Meteorology",
                 "meteo": "Meteorology",
                 "random": "Random",
                 "code-switching": "with Code-Switching",
                 "special": "Special",
                 "wissen": "Science",
                 "sport": "Sports",
                 "rottkaeppchen": "Story",
                 "frau-holle": "Story",
                 "schweizer-politik": "Swiss politics",
                 "regional": "Swiss regional"}

topic_ratios = {
    "Animals/farming": 4,
    "Culture": 3,
    "Earth/Space": 0.5,
    "Economics": 6,
    "International politics": 4,
    "Medicine": 3.5,
    "Meteorology": 6,
    "Random": 2,
    "with Code-Switching": 2,
    "Special": 43,
    "Science": 6,
    "Sports": 3.5,
    "Story": 1,
    "Swiss politics": 5.5,
    "Swiss regional": 10
}

# read data
# data_v1_0_path = "data_1.0/sentences_ch_de_numerics.json"
data_path = args.data_path

data_df = pd.read_json(data_path)
# extract parallel data that has all dialects
parallel_df = data_df.dropna(subset=[
    "de", "ch_ag", "ch_be", "ch_bs", "ch_gr",
    "ch_lu", "ch_sg", "ch_vs", "ch_zh"])

# Remap the topic names to the names used in the paper
for topic in data_df["thema"].unique():
    assert topic in topic_mapping, f"Topic {topic} not found in the mapping dictionary."
    data_df.loc[data_df["thema"] == topic, "thema"] = topic_mapping[topic]
    parallel_df.loc[parallel_df["thema"] == topic, "thema"] = topic_mapping[topic]

total_ratio = sum(topic_ratios.values())
assert total_ratio == 100, "Total topic ratio must sum to 100"
topic_percentages = {
    topic: ratio / total_ratio for topic, ratio in topic_ratios.items()}


# compute sizes for each split
split_fracs = (0.7, 0.1, 0.2)
total_samples = len(data_df)
train_size = int(split_fracs[0] * total_samples)
test_size = int(split_fracs[2] * total_samples)
# assign the remaining size to the dev set
dev_size = total_samples - train_size - test_size

# Shuffle the data for randomness
data_df = shuffle(data_df, random_state=args.seed)
parallel_df = shuffle(parallel_df, random_state=args.seed)


# initialise split storage
splits = {"train": [], "dev": [], "test": []}
rest_df = data_df.copy()
parallel_rest_df = parallel_df.copy()

for topic, percentage in topic_percentages.items():
    # fetch all samples for this topic
    topic_df = rest_df[rest_df["thema"] == topic]
    parallel_topic_df = parallel_rest_df[parallel_rest_df["thema"] == topic]

    topic_count = len(topic_df)
    assert topic_count > 0, f"No samples found for topic: {topic}"

    # target counts
    # calculate expected counts for test from parallel data
    expected_test_topic_total = int(percentage * test_size)
    if expected_test_topic_total > len(parallel_topic_df):
        expected_test_topic_total = len(parallel_topic_df)

    expected_train_topic_total = int(percentage * train_size)
    expected_dev_topic_total = int(percentage * dev_size)
    expected_total = expected_train_topic_total + expected_test_topic_total + expected_dev_topic_total

    # scale up or down proportionally to number of available topic instances
    if topic_count < expected_total:
        scale = topic_count / expected_total
        expected_train_topic_total = int(expected_train_topic_total * scale)
        expected_dev_topic_total = int(expected_dev_topic_total * scale)
        expected_test_topic_total = int(expected_test_topic_total * scale)
        expected_total = expected_train_topic_total + expected_test_topic_total + expected_dev_topic_total

    if topic_count > expected_total:
        # add rest to train set
        rest = topic_count - expected_total
        expected_train_topic_total += rest
        expected_total = expected_train_topic_total + expected_test_topic_total + expected_dev_topic_total

    # sample instances for train, dev and test in the corresponding sizes
    # extract test instances from parallel data
    splits["test"].append(parallel_topic_df.iloc[:expected_test_topic_total])
    # remove used samples from the whole topic_df
    used_parallel_indices = parallel_topic_df.iloc[:expected_test_topic_total].index
    topic_df = topic_df.drop(index=used_parallel_indices)

    splits["train"].append(topic_df.iloc[:expected_train_topic_total])
    splits["dev"].append(topic_df.iloc[expected_train_topic_total:expected_train_topic_total + expected_dev_topic_total])

    # remove used samples
    used_indices = topic_df.iloc[:-1].index
    rest_df = rest_df.drop(index=used_indices)

# concatenate topics and sort by id in ascending order
train_df = pd.concat(splits["train"]).reset_index(
    drop=True).sort_values(by="id")
dev_df   = pd.concat(splits["dev"]).reset_index(
    drop=True).sort_values(by="id")
test_df  = pd.concat(splits["test"]).reset_index(
    drop=True).sort_values(by="id")
assert sum([len(train_df), len(dev_df), len(test_df)]) == len(data_df), "Total samples in splits do not match original data length"


# write splits to formatted files at the location of the data_path
def write_file(df, filename):
    dial_outfile = filename + "_dial.tsv"
    de_outfile = filename + "_deu.tsv"

    with open(dial_outfile, "w", encoding="utf-8") as dial_f, open(de_outfile, "w", encoding="utf-8") as de_f:
        # write headers
        dial_f.write("ID\tText\tTopic\tAudio\n")
        de_f.write("ID\tText\tTopic\n")

        for row in df.itertuples():
            for dialect in dialects:
                if getattr(row, dialect) is not None and not pd.isna(getattr(row, dialect)):
                    if dialect == "de":
                        de_f.write(f"{row.id}\t{getattr(row, dialect)}\t{row.thema}\n")
                    else:
                        audio_name = f"{dialect}_{row.id:04d}.wav"
                        audio_path = f"../../../mainlp/corpora/sid/SwissDial/{dialect[-2:]}/{audio_name}"
                        dial_f.write(f"{row.id}\t{getattr(row, dialect)}\t{row.thema}\t{audio_path}\n")


write_file(train_df,
           os.path.dirname(args.data_path) + "/swissdial_train")
write_file(dev_df, os.path.dirname(args.data_path) + "/swissdial_dev")
write_file(test_df, os.path.dirname(args.data_path) + "/swissdial_test")
