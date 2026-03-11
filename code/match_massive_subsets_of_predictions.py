# The Bavarian test split of MASSIVE only contains a subset of the data.
# Get the corresponding subset of Std German instances.
from glob import glob


massive_indices_ba = []
with open("../data/intents/massive_deba_resplit_test.tsv") as f:
    first_line = True
    for line in f:
        if first_line:
            first_line = False
            continue
        massive_indices_ba.append(line.split("\t", 1)[0])


pred_files = glob("../predictions/intents/*/*massive_deu_test_mapped.tsv")
pred_files += glob("../predictions/intents/*/*speechmassive_de_test.tsv")
pred_files += glob("../data/intents/asr/*/*speechmassive_de_test.tsv")
for pred_file in pred_files:
    with open(pred_file[:-4] + "_matched.tsv", "w") as f_out:
        with open(pred_file) as f_in:
            first_line = True
            for line in f_in:
                if first_line:
                    first_line = False
                    f_out.write(line)
                    continue
                idx = line.split("\t", 1)[0]
                if idx in massive_indices_ba:
                    f_out.write(line)
