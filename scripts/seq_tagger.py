from run import SNACSData

import argparse
import copy
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
import torch.optim as optim
from torchcrf import CRF
from tqdm import tqdm

GPU_ID = '0'
SEED = 20
torch.manual_seed(SEED)
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path("./../models/")

# Constructing a custom Dataset class function which would eventually
# be an input to create a DaatLoader
class SNACSDataset(data.Dataset):
    def __init__(self, inps, labs):
        self.inps = inps
        #self.word_ids = word_ids
        self.labels = labs
                
    def __getitem__(self, idx):
        return self.inps[idx], self.labels[idx]

    def __len__(self):
        return len(self.inps)



class SeqTagger(torch.nn.Module):
    def __init__(self, data, sentences, label_set="sr"):
        super(SeqTagger ,self).__init__()
        self.lm_model = data.model  # The transformers model intialized earlier
        self.label_set = label_set  # Label set for "Scene Role" or "Function"
        self.label_to_ix, self.labels = self.prep_label_set(sentences)  # Prepares label to id maps
    
        self.crf = CRF(len(self.labels), batch_first=True).to(device)   # CRF layer
        self.lm_hidden_size = self.lm_model.model.config.hidden_size    
        # Single feedforward network as a classification head
        self.ffn = torch.nn.Linear(self.lm_hidden_size, len(self.labels)).to(device)


    def prep_label_set(self, sentences):
        """ Create the label set for the scene role/ function
        Inputs
        -----------------
        sentences- list(Dict). Data from the Little Prince processed based on 
                    SNACSData clas. Please refer to run.py file for details. 
    
        
        Outputs
        ---------------
        label_to_ix - dict. Mapping between Roles and class indices
        labels - List. Reverse mapping of label_to_ix
        """
        labels = []
        label_to_ix = {}
        self.key = "scene_roles" if label_set == "sr" else "functions"

        for sent in sentences:
            for role in sent["input_data"][self.key]:
                # Considering non-NaN and non-sicourse markers
                if (role == role) and (role not in labels) and (role != "`"):
                    labels.append(role)
                    label_to_ix[role] = len(labels) - 1

        labels.extend(["I","O"])
        label_to_ix["I"] = len(labels)-2
        label_to_ix["O"] = len(labels)-1

        return label_to_ix, labels
            



    def tokenize_sent(self, sentence):
        """ Method to prepare the input data given information for a sentence. Tokenization and other
        preprocessing happens here.
        Inputs
        -------------
        sentence: dict{dict}. A dictionary containing information about the words and targets for a sentence

        Output
        ----------
        processed_inp: dict. Processed input which contains a tensor of sub-word ids and the list of indices
                        corresponding to target adpositions.
        """

        processed_inp = self.lm_model.tokenizer(sentence["input_data"]["words"], is_split_into_words=True, return_tensors="pt") # Tensor of tokenized words (sub-word) ids
        processed_inp["word_ids"] = processed_inp.word_ids()[1:-1] # removing special characters
        
        return processed_inp



    def translate_label(self, sentence):
        """ Method to prepare the labels for the sentence. Each label is converted to its corresponding class id.
        Inputs
        -------------
        sentence: dict{dict}. A dictionary containing information about the words and targets for a sentence

        Output
        ----------
        roles: list. List of labels ids occurring in the sentence
        """
        roles = [self.label_to_ix["O"]]*len(sentence["input_data"][self.key])   # Initializing everything with an 'O' label
        # Determine whether each sub-word is target or not. If Yes, give it the appropriate tag 
        for ix, role in enumerate(sentence["input_data"][self.key]):
            if role == role and role != "`":
                roles[ix] = self.label_to_ix[role]

        # For multi sub-word targets, all but the first should have an 'I' label
        for targ_id in sentence["input_data"]["target_ids"]:
            if sentence["input_data"][self.key][targ_id[0]] == "`":
                continue
            
            if len(targ_id) > 1:
                for el in targ_id[1:]:
                    roles[el] = self.label_to_ix["I"]
        
        return roles
            
            

    def get_mean_embeddings(self, emb, word_ids):
        """ Get average embeddings for all targets.
        Inputs
        ------------------
        emb: torch.Tensor. The embeddings for all subb-words in the sentence
        word_ids: torch.Tensor. List of ids mapping sub-words to the words
        targets: List. List of ids corresponding to target adpositions.

        Output
        -------------
        final: torch.Tensor . Tensor with average embeddings of each target
        """
        rel_emb = emb[1:-1,:]   # We do not need the special token embeddings
        idx_mean = []           # This keeps track of the previous token
        ix_list = []            # This keeps track of indices to mean_pool
        final = None            # The final mean_pooled vectors

        # Iterate over the word_ids
        for num, id in enumerate(word_ids):
            el = id.numpy()
            # Need to add the first element anyways
            if len(ix_list)==0:
                ix_list= [el]
                idx_mean = [num]
                continue
            # If the current and previous element are not the
            # same, we need to compute the mean-pool of all the vectors
            # corresponding to the indices in the idx_mean list. 
            # Hence, we are mean-pooling all the sub-word embeddings 
            # of a particular word.
            # If they are the same, we just note that they belong to 
            # the same word
            if el != ix_list[-1]:
                mean_vec = torch.mean(rel_emb[idx_mean,:],dim=0,keepdim=True)
                if final == None:
                    final = mean_vec
                else:
                    final = torch.cat((final,mean_vec),dim=0)
                ix_list = [el]
                idx_mean = [num]
            else:
                ix_list.append(el)
                idx_mean.append(num)
        
        mean_vec = torch.mean(rel_emb[idx_mean,:],dim=0,keepdim=True)
        if final == None:
            final = mean_vec
        else:
            final = torch.cat((final,mean_vec),dim=0)
        # The returned tensor should be of the size of words x emb_dim
        return final



    def create_dataloader(self, sentences_split, shuffle=False):
        """ Creates a torch data loader for a set of input sentences. 
        Inputs
        ----------------
        sentences_split: List[dict]. A split of data. The format as the same as returned by the 
                        `extract_sentences` method in `run.py`
        shuffle: bool. If True, the data loader is initialized such that data is shuffled at every epoch

        Output
        --------------
        split_dataloader: torch.data.DataLoader. A dataloader which can be used for iterating over the data
        """
        inps_split = list(map(self.tokenize_sent, sentences_split))     # Prepares input data
        gold_split = list(map(self.translate_label, sentences_split))   # Prepares targets
        split_dataset = SNACSDataset(inps_split, gold_split)    # Prepares the custom dataset
        # Create a datalader using the above dataset
        split_dataloader = data.DataLoader(dataset=split_dataset, shuffle=shuffle,  batch_size=1)

        return split_dataloader



    def train(self, sentences_train, sentences_dev, sentences_test, model_name, max_epochs=100, lr=0.00001, e_stop=5, save=True):
        """ Training loop for the model.
        Inputs
        -----------
        sentences_train,sentences_dev_sentences_test: List[dicti]. A split of data. The format as the same 
                                                    as returned by the `extract_sentences` method in `run.py`
        model_name: str. Model name from the HuggingFace Transformers model list
        max_epochs: int. Maximum number of epochs to run the training loop
        lr: float. Learning Rate
        e_stop: int. Early Stopping epochs
        save: bool. If True, models are saved to directory.
        """
        # Creates appropriate data loaders for the splits. The train data is shuffled at every epoch.
        train_dataloader = self.create_dataloader(sentences_train,shuffle=True)
        dev_dataloader = self.create_dataloader(sentences_dev)
        test_dataloader = self.create_dataloader(sentences_test)

        # List of parameters to optimize and subsequently initializing the optimizer.
        params = list(self.lm_model.model.parameters()) + list(self.crf.parameters()) + list(self.ffn.parameters())
        optimizer = optim.Adam(params, lr=lr)

        best_metric = 0 # Variable to record the best performance recorded on dev split
        no_improv = 0   # Counter which counts the number of consecutive epochs for which the best dev score isnot beaten
       
        # Makes the model folder where models will be dumped
        model_folder = Path(MODEL_DIR,f"""{model_name.split("/")[-1]}/{SEED}/{self.label_set}/""")
        if save:
            Path.mkdir(model_folder, exist_ok=True, parents=True)

        best_state_dict = {}    # Dictionary where best parameters will be stored

        # Training loop
        for ep in range(max_epochs):
            tr_loss = 0
            print(f"Epoch {ep+1}")

            # Iterating over the data
            for inp,gold_lab in tqdm(train_dataloader):
                optimizer.zero_grad()

                # Skipping sentences with no targets
                if len(gold_lab) == 0:
                    continue
                
                # Forward pass
                embs = self.lm_model.model(inp['input_ids'].squeeze(0).to(device), attention_mask=inp['attention_mask'].squeeze(0).to(device))
                # Get mean sub-words emnbeddings to construct embedding for a word
                embs_mean = self.get_mean_embeddings(embs.last_hidden_state.squeeze(), inp['word_ids'])
                emissions = self.ffn(embs_mean.unsqueeze(0))
                
                # Compute the log likelihood loss
                log_likelihood = -self.crf(emissions,torch.Tensor(gold_lab).long().unsqueeze(0).to(device))
                
                tr_loss += log_likelihood.item()
                log_likelihood.backward()
                optimizer.step()
             
            print(f"Training Loss: {tr_loss/len(train_dataloader)}")

            # Evaluate on the dev split
            dev_metrics = self.eval_split(dev_dataloader)
            dev_f1_macro = dev_metrics["Macro"]['f1']   # The primary metric to decide the best model
            print(f"""Dev Entity F1 Macro: {dev_f1_macro}""")

            # If the dev metric is metric than the previous best, we store the parameters and update the
            # relevant variables
            if dev_f1_macro > best_metric:
                best_metric = dev_f1_macro
                no_improv = 0       # Reset early stop counter
                test_metrics = self.eval_split(test_dataloader) # Evaluation is done on the test set (nopt displayed)

                # Best parameters are stored in the dictionary 
                best_state_dict = {
                    "lm_model_state_dict" : copy.deepcopy(self.lm_model.model.state_dict()),
                    "crf_state_dict"    : copy.deepcopy(self.crf.state_dict()),
                    "ffn_state_dict"    : copy.deepcopy(self.ffn.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(optimizer.state_dict())
                    }
                
                # Dump dev and test results into the JSON format
                dev_json = json.dumps(dev_metrics,indent=4)
                test_json = json.dumps(test_metrics, indent=4)

                best_dev_metrics = dev_json
                best_test_metrics = test_json
                best_ep = ep+1
            else:
                # If no improvements then earlyh stop counter is increased
                if best_metric != 0:
                    no_improv += 1

            # Early stop condition reached then the training stops
            if no_improv == e_stop:
                print("Early Stopping limit reached. Avoiding Overfitting!")
                break
            print("\n")


        # Save model parameters, dev and test metrics to disk
        if save:
            torch.save(best_state_dict, Path(model_folder,f"{best_ep}"))
            with open(Path(model_folder,f"{best_ep}_dev_metrics.json"),"w") as f:
                f.write(best_dev_metrics)
            with open(Path(model_folder,f"{best_ep}_test_metrics.json"),"w") as f:
                f.write(best_test_metrics)



    def load_model(self, model_path):
        """ Method to load a model on directory. File path to model specified by `model_path`
        """
        checkpoint = torch.load(model_path, map_location=device)
        self.lm_model.model.load_state_dict(checkpoint["lm_model_state_dict"])
        self.crf.load_state_dict(checkpoint["crf_state_dict"])
        self.ffn.load_state_dict(checkpoint["ffn_state_dict"])



    def evaluate(self, sentences):
        """ Method to evaluate a set of sentences
        """
        dataloader = self.create_dataloader(sentences)
        metrics = self.eval_split(dataloader)
        print(json.dumps(metrics,indent=4))



    def eval_split(self, loader):
        """" Method to evaluate a split.
        Inputs
        -------------
        loader: torch.data.DataLoader. Data loader for the split

        Outputs
        ------------
        metrics: dict. Dictionary containing evaluation metrics
        """
        gold_label_list = []
        prediction_list = []

        # Iterate over the data
        for inp, gold_lab in tqdm(loader):
            with torch.no_grad():
                # Forward pass as mentioned in the training loop
                embs = self.lm_model.model(inp['input_ids'].squeeze(0).to(device), attention_mask=inp['attention_mask'].squeeze(0).to(device))
                embs_mean = self.get_mean_embeddings(embs.last_hidden_state.squeeze(), inp['word_ids'])
                emissions = self.ffn(embs_mean.unsqueeze(0))
                preds = self.crf.decode(emissions)

                # Updating the gold and predicted labels
                prediction_list.append(preds[0])
                gold_label_list.append(np.array(gold_lab).tolist())

        # Compute the F1 metrics given the list of predicitons and gold labels
        metrics = self.calc_f1_ent(prediction_list, gold_label_list)
        
        return metrics



    def calc_f1_from_cmat(self, cmat):
        """ Method to compue F1 stat from a confusion matrix. 
        Input
        -------
        cmat: List[List]. Confusion matrix

        Output
        -------
        metrics: dict. Dictionary containing the F1 stats
        """
        metrics = {}

        tp = cmat[0][0]
        fp = cmat[1][0]
        fn = cmat[0][1]

        if (tp+fp) == 0:
            pr = 0
        else:
            pr = tp/(tp+fp)
        metrics['precision'] = pr

        if (tp+fn) == 0:
            re = 0
        else:
            re = tp/(tp+fn)
        metrics['recall'] = re

        if (pr+re) == 0:
            metrics['f1'] = 0
        else:
            metrics['f1'] = (2*pr*re)/(pr+re)

        if tp+fp+fn == 0:
            metrics["zero_points"] = True
        else:
            metrics["zero_points"] = False

        return metrics




    def calc_f1_ent(self, pred_list, gold_list):
        """ Method to compute F1 statistics for the model
        Input
        ---------
        pred_list: List. List of model predictions
        gold_list: List. List of gold predictions

        Output
        -------
        metrics: dict. Dictionary of class and corpus level F1 stats
       
        """
        # Intialize a list of matrices with zeros. One matrix for one class
        c_matrices = []
        for ix in range(len(self.labels)-2):
            c_matrices.append(np.zeros((2,2)))

        c_mat_extr = np.zeros((2,2))

        def extract_ent(lab_list):
            """ Extracts entites and their given labels. Returns a list of dictionaries where
            each dictionary corresponds to a target and each dictionary contains the label and start
            and end indices for the target.
            """
            entities = []
            entity_lab = None

            for lab_idx, lab in enumerate(lab_list):
                if lab == self.label_to_ix["O"]:
                    if entity_lab != None:
                        entities.append({"lab": entity_lab, "start_ix":start_ix, "end_ix": end_ix})
                        entity_lab = None
                        start_ix = None
                        end_ix = None
                elif lab == self.label_to_ix["I"]:
                    end_ix = lab_idx
                else:
                    if entity_lab != None:
                        entities.append({"lab": entity_lab, "start_ix":start_ix, "end_ix": end_ix})
                    entity_lab = lab
                    start_ix = lab_idx
                    end_ix = lab_idx
            
            if entity_lab != None:
                entities.append({"lab": entity_lab, "start_ix":start_ix, "end_ix": end_ix})
            
            return entities

        # For each sentence 
        for sent_ix in range(len(gold_list)):
            # Extract all entities from the sentence and their given labels
            g_ents = extract_ent(gold_list[sent_ix])
            p_ents = extract_ent(pred_list[sent_ix])
            
            for g_ent in g_ents:
                flag = False #Tracks if a match found or not
                for p_ent in p_ents:
                    # Condition for extraction match
                    if (g_ent["start_ix"] == p_ent["start_ix"]) and \
                            (g_ent["end_ix"]  == p_ent["end_ix"]):
                        flag = True #This entity will be processed
                        c_mat_extr[0][0] += 1
                        # Do the labels match as well
                        if g_ent["lab"] == p_ent["lab"]:
                            c_matrices[g_ent["lab"]][0][0] += 1 #True Positive
                        else:
                            c_matrices[g_ent["lab"]][0][1] += 1 #False Negative for that class
                            c_matrices[p_ent["lab"]][1][0] += 1 #False Positive for that class
                    
                    if flag:
                        break
                # If we don't find any matches, we know the gold
                # entity is a miss, i.e, False Negative
                if not flag:
                    c_matrices[g_ent["lab"]][0][1] += 1
                    c_mat_extr[0][1] += 1


            # Taking care of prediction which were not in the gold list
            for p_ent in p_ents:
                flag = False
                for g_ent in g_ents:
                     if (g_ent["start_ix"] == p_ent["start_ix"]) and \
                            (g_ent["end_ix"]  == p_ent["end_ix"]):
                        flag = True
                        break
                # Writing all remaining False Positives
                if not flag:
                    c_matrices[p_ent["lab"]][1][0] += 1
                    c_mat_extr[1][0] += 1


            
        metrics = {}
        f1_mac = 0
        pr_mac = 0
        re_mac = 0
        cnt = 0

        # For all confusion matrices compute the F1 stats
        for c_mat_ix, c_mat in enumerate(c_matrices):
            # Get F1, precision and recall stats for a confusion matrix
            metrics[self.labels[c_mat_ix]] = self.calc_f1_from_cmat(c_mat)
            if not metrics[self.labels[c_mat_ix]]['zero_points']:
                f1_mac += metrics[self.labels[c_mat_ix]]['f1']
                pr_mac += metrics[self.labels[c_mat_ix]]['precision']
                re_mac += metrics[self.labels[c_mat_ix]]['recall']
                cnt += 1

        # Compute macro averages
        f1_mac /= cnt
        pr_mac /= cnt
        re_mac /= cnt

        metrics["Macro"] = {"f1":f1_mac, "precision": pr_mac, "recall": re_mac}

        metrics["extractions"] = self.calc_f1_from_cmat(c_mat_extr)

        return metrics





    def get_majority_baseline(self,sentences_train,sentences_test):
        """ Method to get the majority baseline
        """
        adpositions = {"scene_roles":{} , "functions": {}}
    
        # Ietterate over sentences in the training set
        for row in sentences_train:
            for target in row['input_data']['target_ids']:
                # Construct a distibution of scene roles and function labels 
                # by adpositions
                sr = row['input_data']['scene_roles'][target[0]]
                fn = row['input_data']['functions'][target[0]]
            
                if row['input_data']['scene_roles'][target[0]] == '`':
                    continue

                adp = []
                for idx in target:
                    adp.append(row['input_data']['words'][idx])
                adp_key = " ".join(adp)
            
                if adp_key not in adpositions["scene_roles"].keys():
                    adpositions["scene_roles"][adp_key] = {}
                    adpositions["functions"][adp_key] = {}
            
                if sr not in adpositions["scene_roles"][adp_key].keys():
                    adpositions["scene_roles"][adp_key][sr] = 0
                if fn not in adpositions["functions"][adp_key].keys():
                    adpositions["functions"][adp_key][fn] = 0

                adpositions["scene_roles"][adp_key][sr] += 1
                adpositions["functions"][adp_key][fn] += 1
    
        # For every adposition compute the most frequent scene role and function label
        adp_sr_most_freq = {}
        adp_fn_most_freq = {}

        for key in adpositions["scene_roles"].keys():
            dictionary = adpositions["scene_roles"][key]
            adp_sr_most_freq[key] = max(dictionary,key=dictionary.get)
   
        for key in adpositions["functions"].keys():
            dictionary = adpositions["functions"][key]
            adp_fn_most_freq[key] = max(dictionary,key=dictionary.get)
   

        # Assign the target adpositon in the test set
        # with the most frequent lavbel found for that adposition in the train set 
        gold_sr = []
        gold_fn = []
        pred_sr = []
        pred_fn = []
        most_pop = "Focus"

        for row in sentences_test:
            for target in row['input_data']['target_ids']:
                sr = row['input_data']['scene_roles'][target[0]]
                fn = row['input_data']['functions'][target[0]]
            
                if row['input_data']['scene_roles'][target[0]] == '`':
                    continue

                adp = []
                for idx in target:
                    adp.append(row['input_data']['words'][idx])
                adp_key = " ".join(adp)
            
                gold_sr.append(sr)
                gold_fn.append(fn)

                if adp_key not in adp_sr_most_freq.keys():
                    pred_sr.append("Focus")
                    pred_fn.append("Focus")
                else:
                    pred_sr.append(adp_sr_most_freq[adp_key])
                    pred_fn.append(adp_fn_most_freq[adp_key])
                
        
        def get_cmats(pred,gold):
            c_matrices = {}
            for lab in self.labels:
                if lab not in ["I","O"]:
                    c_matrices[lab] = np.zeros((2,2))
            
            for ix in range(len(pred)):
                if pred[ix] == gold[ix]:
                    c_matrices[pred[ix]] += 1
                else:
                    c_matrices[gold[ix]][0][1] += 1
                    c_matrices[pred[ix]][1][0] += 1
            return c_matrices

        # Get all the confusion matrics as we did in the previous method
        cmats_sr = get_cmats(pred_sr,gold_sr)
        cmats_fn = get_cmats(pred_fn,gold_fn)

        metrics_sr = {}
        metrics_fn = {}
        
        f1_mac = 0
        pr_mac = 0
        re_mac = 0
        cnt = 0

        # For all confusion matrices compute the F1 stats
        for key in cmats_sr.keys():
            # Compute the F1 stats
            metrics_sr[key] =  self.calc_f1_from_cmat(cmats_sr[key])
            if not metrics_sr[key]['zero_points']:
                f1_mac += metrics_sr[key]['f1']
                pr_mac += metrics_sr[key]['precision']
                re_mac += metrics_sr[key]['recall']
                cnt += 1

        # Compute the maco averages
        f1_mac /= cnt
        pr_mac /= cnt
        re_mac /= cnt

        metrics_sr["Macro"] = {"f1":f1_mac, "precision": pr_mac, "recall": re_mac}
        print(metrics_sr["Macro"])
 
        f1_mac = 0
        pr_mac = 0
        re_mac = 0
        cnt = 0

        for key in cmats_fn.keys():
            metrics_fn[key] =  self.calc_f1_from_cmat(cmats_fn[key])
            if not metrics_fn[key]['zero_points']:
                f1_mac += metrics_fn[key]['f1']
                pr_mac += metrics_fn[key]['precision']
                re_mac += metrics_fn[key]['recall']
                cnt += 1

        f1_mac /= cnt
        pr_mac /= cnt
        re_mac /= cnt

        metrics_fn["Macro"] = {"f1":f1_mac, "precision": pr_mac, "recall": re_mac}
        print(metrics_fn["Macro"])





def parse_args(parser):
    parser.add_argument('--lang',default=f"guj", type =str)
    parser.add_argument('--data_dir',default=Path("./../data/annotated/"), type =str)   # Location of data
    parser.add_argument('--model_name', default="ai4bharat/indic-bert", type =str)  # model from the HF library
    parser.add_argument('--mode', default="train", type=str)    # "train" or "eval" mode
    parser.add_argument('--label_set', default="sr", type=str)  # "sr" or "fn" for Scene Roles and Functions
    parser.add_argument('--eval_model_path', default="./../models/indic-bert/42/sr/2", type=str)    # In "eval" mode, the location from where model is loaded
    parser.add_argument('--eval_split',default="dev",type=str)  # Eval split used for evaluation in "eval" mode
    parser.add_argument('--e_stop', default=5,type=int)         # Epochs for early stopping
    parser.add_argument('--save_model', dest='save_model', action='store_true') # If set models are saved to disk
    parser.add_argument('--no-save_model', dest='save_model', action='store_false') # If set, mdoels are not saved
    parser.add_argument('--lr', default=0.00001, type=float)    # learning rate
    parser.add_argument('--max_epochs', default=100, type=int)  # maximum epochs for training
    parser.set_defaults(save_model=True)
    
    return parser




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_args(parser)
    args = vars(parser.parse_args())

    lang = args['lang']    # Specifying language
    data_dir = args['data_dir'] # Data directory for the data
    model_name = args['model_name']     # Language Model to train on
    mode = args['mode']
    label_set = args['label_set']

    # Processing the data to list format
    snacs_data = SNACSData(data_dir=data_dir, demo=False, show_stats=False, lang=lang, model_name=model_name)
    sentences = snacs_data.extract_sentences(tokenization_guj=True)
    
    # Preparing data splits
    sentences_train, sentences_inter = train_test_split(sentences, test_size=0.3, random_state=42)
    sentences_dev, sentences_test = train_test_split(sentences_inter, test_size=0.5, random_state=42)

    # Intializing the tagger
    tagger = SeqTagger(snacs_data, sentences, label_set=label_set)
   
    print("***Majority Baseline**")
    tagger.get_majority_baseline(sentences_train, sentences_test)

    # Training module
    if mode == "train":
        tagger.train(sentences_train, sentences_dev, sentences_test, model_name=model_name,\
                max_epochs=args['max_epochs'], lr=args['lr'], e_stop=args['e_stop'], save=args["save_model"])
    else:
        tagger.load_model(args['eval_model_path'])
        if args["eval_split"] == "dev":
            tagger.evaluate(sentences_dev)
        else:
            tagger.evaluate(sentences_test)
