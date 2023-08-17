# Running CRA Experiments for Gujarati SNACS

Please ensure that you have followed the setup instructions provided in [`weak_verifiers/README.md`](https://github.com/utahnlp/weak-verifiers/blob/main/README.md).

As an example instance, we provide details on how to run the CRA experiments on the Gujarati SNACS dataset. While we show the steps with respect to this particular dataset, the idea is much more general and can be extended to other datasets. 

The steps are as follows:
1. We have annotations from an annotator. The first step is to create three files that shall serve as input files to DirectProbe. <i>The first file contains the data about the target tokens/sentences and their label</i>. Each target should be line separated. In our dataset, each line will correspond to an adposition or case marker annotated. The info regarding the target and its annotation should be tab-separated. <i>The second file contains the embeddings corresponding to the targets</i>. These are contextual representations from a model for the targets mentioned in the first file. Naturally, these are parallel files meaning that the targets and its corresponding embedding need to be on the same line in both files.  <i>The third file contains a list of valid labels for the task (each label in a separate line)</i>. It does not matter if certain labels do not occur in your dataset.<br>
    To create these files, run the following command:
    ```console
       (<venv_name>)foo@bar: weak-verifiers/scripts$ python run.py --data_dir ./../data/annotated/ --model_name google/muril-large-cased --save_dir_prefix final_dataset/gu_muril-large_
    ```
    where `data_dir` is the location of the data files, `model_name` is the model being used from the [HuggingFace model library](https://huggingface.co/models), and `save_dir_prefix` is the directory under `scripts/DirectProbe/data/` sub-directory where your files should be stored.
   Once you run the above command, you will find the first two files created at `scripts/DirectProbe/data/final_dataset/gu_muril-large_SS-SR/entities/train.txt` and `scripts/DirectProbe/data/final_dataset/gu_muril-large_SS-SR/embeddings/train.txt` respectively for the Scene Role labels. Similarly, files for the Function label will also be generated. We will consider only Scene Roles for the sake of this example. Finally, the third file (list of labels) is provided [here](https://github.com/utahnlp/weak-verifiers/blob/main/scripts/misc/tags.txt) and can be stored at some location, say, `scripts/DirectProbe/data/final_dataset/gu_muril-large_SS-SR/labels/tags.txt`.
2. Now that we have the input files, we can run DirectProbe. For any dataset, we need to just specify the relevant config file to run DirectProbe. An example config file for the dataset described here is given [here](https://github.com/utahnlp/weak-verifiers/blob/main/scripts/misc/config.ini). For any experiments concerning computing CRA score, only three fields are relevant and need to be changed. These are the `output_path` which specifies the folder where the clustering is stored for posterity, the `common` field which points to the location where the files generated in Step 1 reside, and `common2` which will be the same as `common` when we are computing `C_org`, but will point to a different folder where shuffled data resides for the computation of `C_rand`. (We'll talk about this in latter steps).<br> Note: `C_org` is the number of cluster obtained from DirectProbe for the original set of annotations. `C_rand` is the average number of clusters obtained when the annotations are shuffled preserving the label distribution. The averaging is across multiple shuffle implemented using multiple seeds. 
<br>To run the DirectProbe to compute `C_org`, place this file in the `scripts/DirectProbe/` directory and run:
    ```console
       (<venv_name>)foo@bar: weak-verifiers/scripts/DirectProbe$ python main.py
    ```
    Once this command is run, the number of clusters will show up on the console prefixed by `Final number of clusters:`. This is the `C_org` value. Alternatively, you can find the same in the `log.txt` file under the directory mentioned corresponding to `output_path` in the config file.
3. Next, we need to compute `C_rand`. We achieve this by simply changing the mapping between targets and labels in the first file (`*/entities/train.txt`) mentioned in Step 1. We provide a [python script](https://github.com/utahnlp/weak-verifiers/blob/main/scripts/misc/shuff.py) that allows to do random shuffling for different portions of the dataset for a specific seed while conserving the label distribution. Note that since the file is tab separated in the form of `<target>\t<label>`, the script is generic and can be applied to all datasets. We recommend running this for as many seeds as possible. For the paper, we used 42, 20, 1984, 11 and 1996 as random seeds. As an example to create a full shuffle of the dataset with the random seed 42 for our dataset, we can run:
    ```console
       (<venv_name>)foo@bar: weak-verifiers/scripts/$ python shuff.py --seed 42 --ratio 1 --tags_file ./DirectProbe/data/final_dataset/gu_muril-large_SS-SR/entities/train.txt --save_dir ./DirectProbe/data/final_dataset/gu_muril-large_SS-SR/
    ```
    where `ratio` specifies the ratio of the dataset to be shuffled, `tags_file` is the original annotation file created in Step 1 and `save_dir` specifies the directory where the shuffled file will be stored. After running the command, you'll see the shuffled file at `DirectProbe/data/final_dataset/gu_muril-large_SS-SR/100_perc/42/entities/train.txt`. For the sake of computing CRA scores, the ratio should be kept at 1. In order to replicate the trend plots in the paper where we consider different ratios of the dataset, you can change the ratio parameter. We used ratios of 0.05, 0.1, 0.25, 0.5, 0.75, and 1 in our plots. For each ratio, shuffles were created for each of the five seeds mentioned before.
4. After generating the random shuffle file(s), we can rerun the DirectProbe with a changed config pointing to the shuffled data file. This updated config file will have a change in the `common2` field pointing to the directory containing the shuffled data. We can also update the `output_path` so the results are stored in a different folder. Just to help you, we have provided the [updated config](https://github.com/utahnlp/weak-verifiers/blob/main/scripts/misc/config_shuff.ini) as well. Run DirectProbe with the updated config as shown in Step 2 and find out the number of clusters. Find out the clusters for multiple random seeds and average them. This average will be `C_rand`.
5. Once you have `C_org` and `C_rand`, CRA score = 1 - (`C_org`/`C_rand`). Unfortunately, the averaging to get `C_rand` and then computing the CRA score needs to be manually done. 



# Running Baseline Models
We release scripts to train two kinds of baseline models where i) gold adpositions are provided, and ii) gold adpositions are not provided. To run the training for all the representations mentioned in the paper, you can run:
```console
       (<venv_name>)foo@bar: weak-verifiers/scripts/$ sh run_snacs_classifier.sh
```
and
```console
       (<venv_name>)foo@bar: weak-verifiers/scripts/$ sh run_adpplussnacs_classifier.sh
```
respectively. Note that this runs for one single seed. To get the training done for other seeds, you can specific the global variable `SEED` in `adp_classifier.py` and `seq_tagger.py` respectively and re-run the shell script. <br>
The dev and test metrics for the best model alongside the best model parameters will be stored in the model directory specified with sub-folders created appropriately for the representation, the label set and the seed. <br><br>
More tweaks are possible for training. These can be passed as command line argumnents. See the [`parse_args` method](https://github.com/utahnlp/weak-verifiers/blob/a177a05820d43d7fe496188cd5a12ca30a13c71a/scripts/seq_tagger.py#L681) in either file for more details.

To evaluate a certain split (train/dev/test) using an existing model on disk, run:
```console
       (<venv_name>)foo@bar: weak-verifiers/scripts/$ python adp_classifier.py --model_name <hf_model_name> --label_set <label_set> --mode eval --eval_model_path <path_to_model> --eval_split <split>  
```
and
```console
       (<venv_name>)foo@bar: weak-verifiers/scripts/$ python seq_tagger.py --model_name <hf_model_name> --label_set <label_set> --mode eval --eval_model_path <path_to_model> --eval_split <split>  
```
