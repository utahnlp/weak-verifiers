import argparse
from collections import defaultdict
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml

from label import LABELS
from model import Model




class SNACSData():
    def __init__(
            self, 
            data_dir = Path("./../data/annotated/"),
            lang = "guj",
            demo = False,
            show_stats = True,
            model_name = "ai4bharat/indic-bert",
            init_model = True
        ):
        """ Constructor for the data class
        Inputs
        ------------------
        data_dir-    pathlib.Path. The directory where annotated data
                        resides,
        demo        - bool. If True, only considers a small amount of files
        show_stats  - bool. Show scene roles and function label statistics
        """
        self.data_dir = data_dir
        self.demo = demo
        self.lang = lang
        self.model = Model(model_name,lang, init_model)
        # This is the order in which files are processed. Fixed for consistency.
        file_ids = [5,16,22,8,24,12,11,7,1,21,25,13,10,15,27,2,4,17,19,18,26,20,23,14,3,9,6]
        self.files = [Path(f"{self.data_dir}/NR_Ch{f}.csv") for f in file_ids]

        if self.demo:
            self.files = self.files[:3]

        if show_stats:
            self.get_statistics()



    def get_statistics(self):
        """ Get statistics for scene role and functions
        """
        scene_roles_ctr = defaultdict(int)
        functions_ctr = defaultdict(int)
        non_construals_ctr = defaultdict(int)
        construals_ctr = defaultdict(int)
        targets = defaultdict(int)
        
        total_targets = 0
        total_tokens = 0

        for f in tqdm(self.files):
            try:
                data_df = pd.read_csv(f)
            except pd.errors.ParserError:
                data_df = pd.read_csv(f,error_bad_lines=False,engine="python")

            if self.lang == "hi":
                col_map = {"Unnamed: 0":"Word", "SS":"Scene role", "SS2":"Function"}
                data_df = data_df.rename(columns=col_map)            

            annotations  = data_df[["Scene role","Function"]].dropna(how='all')
            
            tokens = data_df["Word"].dropna()   # Removing non targets
            total_tokens += len(tokens)
            ch_wise = 0
            for ix, row in annotations.iterrows():
                total_targets += 1
                scene_role  = row["Scene role"]
                function    = row["Function"]

                scene_roles_ctr[scene_role] += 1
                
                # If a target is left unannotated
                if scene_role != scene_role:
                    print(f"Annotation Mistake\n{f}\n{function}\n")
                    exit()
                ch_wise+=1 
                if function != function:
                    if scene_role!= "`":
                        print(f"Annotation mistake\n{f}\n{scene_role}\n")
                    continue
               
                functions_ctr[function] += 1

                try:
                    if scene_role == function:
                        non_construals_ctr[scene_role+"->"+function] += 1
                    else:
                        construals_ctr[scene_role+"->"+function] += 1

                   
                except TypeError:
                    print(f"Annotation Mistake\n{f}\n{scene_role}\n{function}\n")
                    continue

                if (scene_role not in LABELS) or (function not in LABELS):
                    if scene_role not in ["`i"]: 
                        print(f"Annotation Mistake\n{f}\n{scene_role}\n{function}\n")
            
        
        def sort_dict(ctr):
            return dict(sorted(ctr.items(), key = lambda kv: kv[1], reverse=True))
        
        scene_roles_ctr = sort_dict(scene_roles_ctr)
        functions_ctr = sort_dict(functions_ctr)
        non_construals_ctr = sort_dict(non_construals_ctr)
        construals_ctr = sort_dict(construals_ctr)
        
        print(f"""Scene Role Histogram-\n{yaml.dump(scene_roles_ctr, sort_keys=False, default_flow_style=False)}\n\n""")
        print(f"""Function  Histogram-\n{yaml.dump(functions_ctr, sort_keys=False, default_flow_style=False)}\n\n""")       
        print(f"""Non-Construal Histogram-\n{yaml.dump(non_construals_ctr, sort_keys=False, default_flow_style=False)}\n\n""")
        print(f"""Construal Histogram-\n{yaml.dump(construals_ctr, sort_keys=False, default_flow_style=False)}\n\n""")       
        
        print(f"Total tokens- {total_tokens}")
        print(f"Total targets- {total_targets}")
        print(f"Total distinct Scene Roles(including `): {len(scene_roles_ctr.keys())}")
        print(f"Total distinct Functions: {len(functions_ctr.keys())}")
        sr_keys = set(scene_roles_ctr.keys())
        sr_keys.union(set(functions_ctr.keys()))
        print(f"Total distinct labels: {len(sr_keys)}")
        print(f"Total SR tokens: {sum(scene_roles_ctr.values())}")
        print(f"Total Function tokens: {sum(functions_ctr.values())}")

        print(f"Total distinct construals: {len(construals_ctr.keys())}")
        print(f"Total distinct non-construals: {len(non_construals_ctr.keys())}")
        print(f"Total Construal tokens: {sum(construals_ctr.values())}")
        print(f"Total Non-construal tokens: {sum(non_construals_ctr.values())}")
    


    def extract_sentences(self, tokenization_guj=True):
        """ This is the method to extract the sentences in the data and process for SNACS 
        targets and labels. At the end of the method, a list of dictionaries is returned where each 
        dictionary corresponds to a sentence in the dataset. 
        Input
        --------
        tokenization_guj: bool. If True, inflections are separated from head so that embeddings corresponding to 
                                the inflections (case markers) can be obtained. We highliy recommend this to be 
                                set to True.
        Output
        --------
        extracted_data: List[dict]. a processed list of dictionaries where each dictionary corresponds to a sentence
                                    in the dataset. Each dictionary has meta_data about the sentence and the processed
                                    output for each sentence. In the latter, we have a parallel lists of words,
                                    targets, labels and more. 
        """
        extracted_data = [] # This holds all extracted sentences
        for f in self.files:
            # Get chapter name. This is needed for 
            # met-data tagging
            if self.lang == "guj":
                ch_name = int(f.name.split(".")[0][5:])
            else:
                ch_name = int(f.name.split(".")[0])
       
            # Read data
            try:
                data = pd.read_csv(f)
            except pd.errors.ParserError:
                data = pd.read_csv(f,error_bad_lines=False,engine="python")

            # Mapping Hindi columns to a universal scheme
            if self.lang == "hi":
                col_map = {"Unnamed: 0":"Word", "SS":"Scene role", "SS2":"Function"}
                data = data.rename(columns=col_map)

            annotations = data[["Word","Target","Scene role","Function","MWE"]]     #Grab relevant columns
            # The fields which will be extracted
            input_data = {"words": [], "targets": [], "scene_roles" : [], "functions" : [],"target_ids":[]}
            
            mwe_flag = 0    # Flag indicating multi-word expression
            row_id = 0      # Indicates row index of the start of the sentence. Used for meta-data


            for idx, row in annotations.iterrows():
                # Sentence termination condition
                if row["Word"] != row["Word"]:
                    meta_data = {'ch':ch_name, 'row_id': row_id}
                    if self.lang == "guj" and tokenization_guj:
                        input_data = self.model.tokenize_guj(input_data) #This is where the tokenization for gujarati occurs where the head is separated from the inflected marker. This is done because the embedding for the inflection is of interest to us.
                   
                    # Appending the data to the final extarction list 
                    extracted_data.append({"input_data":input_data.copy(), "meta_data": meta_data.copy()})
                    # Reinitialize input_data dict, MWE flag and update row ID for future
                    input_data = {"words": [],"targets" : [],"scene_roles": [], "functions": [],"target_ids":[]}
                    mwe_flag = 0
                    row_id = idx+1
                else:
                    # Append intermediate words, targets, SRs and Functions
                    input_data["words"].append(row["Word"])
                    input_data["targets"].append(row["Target"])
                    input_data["scene_roles"].append(row["Scene role"])
                    input_data["functions"].append(row["Function"])

                    # For a target adposition/Case-marker
                    # We need to determine whether it is a MWE or not
                    if row["Target"] == row["Target"]:
                        # Is there a continuing MWE
                        if mwe_flag != 0:
                            # Check if the new target belongs to the same MWE (and it is a not a second consecutive MWE)
                            if (row["MWE"] == row["MWE"]) and (int(row["MWE"].split(":")[-1]) == (mwe_flag+1) ):
                                input_data["target_ids"][-1].append(len(input_data['words'])-1)
                                mwe_flag = int(row["MWE"].split(":")[-1])
                            else:
                                # New MWE
                                if row["MWE"] == row["MWE"]:
                                    mwe_flag = int(row["MWE"].split(":")[-1])
                                else:
                                    mwe_flag = 0
                                input_data["target_ids"].append([len(input_data["words"])-1])
                        else:
                            # MWE terminated
                            mwe_flag = 0
                            input_data["target_ids"].append([len(input_data["words"])-1])
                            if row["MWE"] == row["MWE"]:
                                mwe_flag = int(row["MWE"].split(":")[-1])

                    # Special Case for last sentence in a chapter
                    if idx+1 == len(annotations):
                        # Assign meta_data
                        meta_data = {'ch':ch_name, 'row_id': row_id}
                        if self.lang == "guj" and tokenization_guj:
                            input_data = self.model.tokenize_guj(input_data)
                        
                        # Append data to the extarction list
                        extracted_data.append({"input_data":input_data.copy(), "meta_data": meta_data.copy()})
    
                        #Reinitialize input_data dict, MWE flag and update row ID 
                        input_data = {"words": [],"targets" : [],"scene_roles": [], "functions": [],"target_ids":[]}
                        mwe_flag = 0
                        

        return extracted_data






    def prepare_data(self, save, save_dir,tokenization_guj=True):
        """ This is a method which creates the appropriate directories where the DirectProbe files shall
        be stored, process the data and generates data required for DirectProbe.
        Inputs
        -------------
        save: bool. If True, it generates the directories and files for DirectProbe. You would generally 
                    like to set it to True. Set False if debugging is required.
        save_dir: str. This is the prefix name for the directories being generated. 
        tokenization_guj: bool. If True, inflections are separated from head so that embeddings corresponding to 
                                the inflections (case markers) can be obtained. We highliy recommend this to be 
                                set to True.
        """
        # Creating the required dictionaries. 
        if save:
            Path(f"./DirectProbe/data/{save_dir}SS-SR/embeddings/").mkdir(exist_ok=True,parents=True)
            Path(f"./DirectProbe/data/{save_dir}SS-SR/entities/").mkdir(exist_ok=True,parents=True)
            Path(f"./DirectProbe/data/{save_dir}SS-Fn/embeddings/").mkdir(exist_ok=True,parents=True)
            Path(f"./DirectProbe/data/{save_dir}SS-Fn/entities/").mkdir(exist_ok=True,parents=True)
            
        extraction_list = self.extract_sentences(tokenization_guj)
        print(f"Number of sentences: {len(extraction_list)}")

        # The following loop creates embeddings for every target and generates the relevant input
        # files for DirectProbe to be stored in the directories created above.
        for sent in extraction_list:
            self.model.get_embeddings(sent['input_data'],sent['meta_data'], save, save_dir)
                                       

           



def preprocess_config(parser):
    """ Adds Command line arguments to an Empty Argument Parser
    Input 
    -------------
    parser: argparse.ArgumentParser. An empty initialized Argument Parser

    Output
    -------------
    parser: argparse.ArgumentParser. Return the argument parser with added arguments
    """
    parser.add_argument('--lang', default="guj", type=str)  # Support "guj" and "hi"
    # The default is the Guajrati SNACS dataset
    # This scripts support the Hindi annotation as well which can be downloaded 
    # from https://github.com/aryamanarora/carmls-hi/tree/master/annotations/lp_adjudicated_cleaned
    parser.add_argument('--data_dir', default="./../data/annotated/", type=str)
    # Supported models for now:
    # "google/muril-base-cased", "google/muril-large-cased", "ai4bharat/indic-bert"
    # "xlm-roberta-base", "xlm-roberta-large", "bert-base-multilingual-cased"
    parser.add_argument('--model_name', default="google/muril-large-cased", type=str)
    # The following argument specifies the directory prefix where the files will be stored. 
    # These files shall be  input to DirectProbe to compute the clusters 
    parser.add_argument('--save_dir_prefix', default="final_dataset/gu_muril-large_", type=str)

    return parser



if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser = preprocess_config(parser) 
    args = vars(parser.parse_args())

    data_dir    = Path(args["data_dir"])
    lang        = args["lang"]
    model_name  = args["model_name"] 
    save_dir    = args["save_dir_prefix"]
    save        = True

    data = SNACSData(data_dir=data_dir ,demo=False, show_stats=False, lang=lang, model_name=model_name)
    #data.get_statistics() #Uncomment if you are interested in data statistics
    data.prepare_data(save=save, save_dir=save_dir)

