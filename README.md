# Standard-to-Dialect transfer trends differ across text and speech: A case study on intent and topic classification in German dialects

This repository contains code and detailed results for
> Verena Blaschke, Miriam Winkler, Barbara Plank. 2026. Standard-to-Dialect transfer trends differ across text and speech: A case study on intent and topic classification in German dialects. To appear in the *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics*. 

Please cite the paper if you use any of this data/code. A preprint is available at https://arxiv.org/abs/2510.07890.

## Repo structure
- `code`: Check the README in this repo for details on downloading the data and executing the code.
- `data/{intents,topics}`: The re-mapped data sets.
- `data/{intents,topics}/asr`: The ASR transcriptions of the evaluation data, by ASR model.
- `predictions`: Contains the intent classification predictions, in separate subfolders for ech set-up. Each filename encodes the LM, the maximum number of epochs, the batch size, the learning rate, the seed, and the test set.
- `scores/asr`: The WER/CER for the ASR models. For the Bavarian evaluation sets, the columns with "STD" in the names calculate the WER/CER relative to the parallel German sentence rather than the original Bavarian reference. The `_detailed` files show scores for each sentence.
- `scores/{intents,topics}`: The intent classification scores (unaggregated and aggregated over seeds).

All subfolders containing (automatic or gold-standard) transcriptions are in zip archives with the password `MaiNLP` so as to prevent potential inclusion in web-scraped datasets (cf. [Jacovi et al., 2023](https://aclanthology.org/2023.emnlp-main.308/)). 
Unzip them to get the subfolders with the same name.
Please also use a zip archive (or similar) if you re-distribute the transcriptions.

## Licenses + links to the datasets

- MASSIVE: https://github.com/alexa/massive, [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Speech-MASSIVE: https://github.com/hlt-mt/Speech-MASSIVE, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- xSID: https://github.com/mainlp/xsid, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- MAS:de-ba: https://github.com/mainlp/NaLiBaSID
- SwissDial: https://mtc.ethz.ch/publications/open-source/swiss-dial.html, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- xSID-audio: https://doi.org/10.5281/zenodo.19554427

## Known issues
The random seeds are not set properly. While the different runs are in fact seeded differently, the seed numbers in the prediction file names or de-aggregated results tables cannot be used to reproduce a run with exactly the same actual random seed.
