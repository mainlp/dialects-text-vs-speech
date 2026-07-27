# Metadata files for xsid-audio

These TSV files contain mappings between the sentences, intents, and audio files.
For the audio files, see https://zenodo.org/records/19554427.

These files contain the mapping of all xSID dev/test sentences (CC BY-SA 4.0) to intents and to the audio files in https://zenodo.org/records/19554427. **xsid_de-ba_test.tsv fixes an issue with misaligned sentence IDs in the Bavarian data in the Zenodo metadata files in xsid-audio 0.1.** (This fix has no bearing on the experiments in the ACL paper, which already used the correct mappings.)
Note that these misalignments are also present in the original xSID text data -- if using xSID, re-align the Bavarian version accordingly.
**This issue is also fixed in release 0.2 on Zenodo.**

The `Text_original` column contains the text versions as written in xSID. This includes some deliberate orthographic/grammatical errors in the translations (see Appendix F of [van der Goot et al., 2021](https://aclanthology.org/2021.naacl-main.197/)) and some abbreviations ("AK", "Nov."). When recording these sentences, we fixed these errors / expanded these abbreviations ("Alaska", "November") in order to have more natural-sounding audios. The `Text` column contains the fixed versions of the sentences. `Text` and `Text_original` are identical for nearly all sentences, however. 
In the ACL 2026 paper, we worked with the `Text_original` versions. Since the changes are minor and only affect a few sentences, we expect nearly the same ASR and classification scores when using the `Text` version instead.
The `Text` vs. `Text_original` distinction is also included in version 0.2 of the Zenodo dataset.

The files in this subdirectory and on Zenodo represent *all* instances of xSID(-audio), not just the MASSIVE-compatible subset used for the ACL experiments.

## Licenses + links to the datasets

- xSID: https://github.com/mainlp/xsid, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- xSID-audio: https://doi.org/10.5281/zenodo.19554427
