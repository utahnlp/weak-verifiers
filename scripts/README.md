# Running CRA Experiments for Gujarati SNACS

Please ensure that you have followed the setup instructions provided in [`weak_verifiers/README.md`](https://github.com/utahnlp/weak-verifiers/blob/main/README.md).

As an example instance, we provide details on how to run the CRA experiments on the Gujarati SNACS dataset. While we show the steps with respect to this particular dataset, the idea is much more general and can be extended to other datasets. 

The steps are as follows:
1. We have annotations from an annotator. The first step is to create three files that shall serve as input files to DirectProbe. <i>The first file contains the data about the target tokens/sentences and their label</i>. Each target should be line separated. In our dataset, each line will correspond to an adposition or case marker annotated. The info regarding the target and its annotation should be tab-separated. <i>The second file contains the embeddings corresponding to the targets</i>. These are contextual representations from a model for the targets mentioned in the first file. Naturally, these are parallel files meaning that the targets and its corresponding embedding need to be on the same line in both files.  <i>The third file contains a list of valid labels for the task (each label in a separate line)</i>. It does not matter if certain labels do not occur in your dataset.<br>
    To create these files, run the following command:
    ```console
       (<venv_name>)foo@bar:~$ python run.py --data_dir ./../data/annotated/ --model_name google/muril-large-cased --save_dir_prefix final_dataset/gu_muril-large_
    ```
    where `data_dir` is the location of the data files, `model_name` is the model being used from the [HuggingFace model library](https://huggingface.co/models), and `save_dir_prefix` is the directory under `scripts/DirectProbe/data/` sub-directory where your files should be stored.
   Once you run the above command, you will find the first two files created at `scripts/DirectProbe/data/final_dataset/gu_muril-large_SS-SR/entities/train.txt` and `scripts/DirectProbe/data/final_dataset/gu_muril-large_SS-SR/embeddings/train.txt` respectively for the Scene Role labels. Similarly, files for the Function label will also be generated. Finally, the third file (list of labels) is provided [here](https://github.com/utahnlp/weak-verifiers/blob/main/scripts/misc/tags.txt) and can be stored at some location, say, `scripts/DirectProbe/data/final_dataset/gu_muril-large_SS-SR/labels/tags.txt`.
