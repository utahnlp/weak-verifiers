import math
import os
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

torch.manual_seed(42)

# Please change the device ID to the one required.
GPU_ID = '0'
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")


class Model():
    def __init__(self, model_name = "ai4bharat/indic-bert",lang="guj", init_model=True):
        """ Constructor for the Model class. Initializes the selected HF transformers model.
        Input
        --------
        model_name: str. Should be a valid model hosted on HugginFace transformers.
        lang: str in {"guj","hi"}. Language the data is in 
        init_model: bool. If True, the model will be initialized.
        """
        self.lang = lang
        # Initialize the tokenizer
        if model_name == "ai4bharat/indic-bert":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name,keep_accents=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Initialize the model
        if init_model:
            self.model = AutoModel.from_pretrained(model_name).to(device)
        
        # For Gujarati, case markers inflecting their head need to be separated so that
        # we obtain their  corresponding embedding. The following piece of code helps with it
        if lang == "guj":
            self.max_infl = 0
            with open("adpositions.txt","r") as f:
                adpositions  = f.readlines()
            self.preset_guj_adp = []    # This is the list of adpositions which are preset.
            for adp in adpositions:
                self.preset_guj_adp.append(adp.strip("\n"))
                if self.max_infl<len(adp.strip("\n")):
                    self.max_infl = len(adp.strip("\n"))
        
   

    def get_avg_embeddings(self, embs, ids, word_ids):
        """ Get average embedding of a target token 
        Inputs
        --------------------
        embs- torch.Tensor. Embeddings of final layer from the transformer
        ids - list. List of word IDs corresponding to a target
        word_ids - list. Index to word ID mappings obtained from tokenizer
        """
        tensor_ids = []
        # For every word ID in the target token, obtain the embedding indices
        for idx in ids:
            # Add all embedding indices corresponding to a target sub-token
            temp_ids = []
            for i, value in enumerate(word_ids):
                if value == idx:
                    temp_ids.append(i)
                
            tensor_ids.extend(temp_ids)
        
        # Select all tensors corresponding to the target
        selected_tensors = embs[tensor_ids,:]
        rep_vec = torch.mean(selected_tensors,dim=0,keepdim=True).squeeze() # Mean-pool
    
        return rep_vec



    def tokenize_guj(self, inp):
        """ Gujarati requires further tokenization for adpositions
        since they are often fused with their corresponding complements
        Input
        ------------
        inp - dict. Contains space separated word list in the key "words"

        Output
        ------------
        new_inp - dict. Similar dict as the input but now with the detached
                    adpositions.
        """
        new_inp = {'words':[],"targets":[],"scene_roles":[],"functions":[],"target_ids":inp["target_ids"].copy()}
        checklist = [i[0] for i in inp["target_ids"]]   #We just need the first token in case of a MWE
        
        for idx in range(len(inp["words"])):
            def_flag = True
            word_curr = inp['words'][idx]
            # Check if word is a target and if so a target in the adposition list
            if (idx in checklist) and (word_curr not in self.preset_guj_adp) and (word_curr != inp['targets'][idx]):
                ub = min(len(word_curr)-1,self.max_infl)# Biggest inflection is 6 characters
                for c_lim in range(ub,0,-1):
                    # If inflection is detected the the words are separated and all the other data is 
                    # updated accordingly.
                    if (word_curr[-c_lim:] in self.preset_guj_adp):
                        split_1 = word_curr[:-c_lim]
                        split_2 = word_curr[-c_lim:]
                        new_inp['words'].extend([split_1,split_2])
                        new_inp['targets'].extend([math.nan,inp['targets'][idx]])
                        new_inp['scene_roles'].extend([math.nan,inp['scene_roles'][idx]])
                        new_inp['functions'].extend([math.nan,inp['functions'][idx]])
                        for targ_ix, targ in enumerate(new_inp['target_ids']):
                            for el_targ_ix, el in enumerate(targ):
                                if el >= (len(new_inp['words'])-2):    #1 for index and len diff and 1 for the head removed
                                    new_inp['target_ids'][targ_ix][el_targ_ix] += 1  

                        def_flag = False
                        break

            # In case no inflection add the data as it was
            if def_flag:
                new_inp['words'].append(inp['words'][idx])
                new_inp['targets'].append(inp['targets'][idx])
            
                new_inp['scene_roles'].append(inp['scene_roles'][idx])
                new_inp['functions'].append(inp['functions'][idx])

        return new_inp



    def get_embeddings(self,inp,meta_data,save=True,save_dir=""):
        """ This method creates the files required to run DirectProbe
        Inputs
        ----------
        inp: dict. Dictionary containing data for each sentence in the dataset as generated in run.py
        meta_data: dict. Dictionary containing the meta data for the sentence. 
        save: bool.If True, directories are created and files are dumped
        saave_dir: str. Prefix denoting which directory should the files be stored in
        """
        sentence = " ".join(inp['words'])   # String form sentence
        # Tokenize the sentence
        processed_inp = self.tokenizer(inp['words'],is_split_into_words=True, return_tensors='pt')
        
        # Get the sub-wprd embeddings
        with torch.no_grad():
            embs = self.model(**processed_inp.to(device))
        hidden_states = embs.last_hidden_state.squeeze()
    
        # Process every target in the sentence iterartively
        for trgt_id in inp['target_ids']:
            sr = inp['scene_roles'][trgt_id[0]]
            fn = inp['functions'][trgt_id[0]]

            if (sr in ['`d','`i','NONSNACS','`']) or (sr!=sr):
                continue 
            
            # Get the mean embeddings for the target tokens
            avg_embedding = self.get_avg_embeddings(hidden_states, trgt_id, processed_inp.word_ids())

            # Prep the relevant meta data which can help traceback the target
            curr_target = "_".join(list(map(inp['targets'].__getitem__, trgt_id)))
            info = "({},{},{}):{}".format(meta_data["ch"],meta_data["row_id"],trgt_id[0],curr_target)
           
            # create the directories and create teh respective dumps
            if save:
                Path(f"./DirectProbe/data/{save_dir}SS-SR/embeddings/").mkdir(parents=True,exist_ok=True)
                Path(f"./DirectProbe/data/{save_dir}SS-SR/entities/").mkdir(parents=True,exist_ok=True)
                Path(f"./DirectProbe/data/{save_dir}SS-Fn/embeddings/").mkdir(parents=True,exist_ok=True)
                Path(f"./DirectProbe/data/{save_dir}SS-Fn/entities/").mkdir(parents=True,exist_ok=True)
                
                # Dump the embeddings and target info for Scene Roles
                sr_info = info+f"\t{sr}\n"
                with open(f"./DirectProbe/data/{save_dir}SS-SR/embeddings/train.txt","a+") as f:
                    f.write(" ".join(map(str,avg_embedding.tolist())))
                    f.write("\n")
                with open(f"./DirectProbe/data/{save_dir}SS-SR/entities/train.txt","a+") as f:
                    f.write(sr_info)

                # Dump the embeddings and target info for Functions
                fn_info = info + f"\t{fn}\n"
                with open(f"./DirectProbe/data/{save_dir}SS-Fn/entities/train.txt","a+") as f:
                    f.write(fn_info)
                with open(f"./DirectProbe/data/{save_dir}SS-Fn/embeddings/train.txt","a+") as f:
                    f.write(" ".join(map(str,avg_embedding.tolist())))
                    f.write("\n")
        
