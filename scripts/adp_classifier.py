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
SEED = 11
torch.manual_seed(SEED)
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path("./../models/adp_tagger/")


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



class AdpClassifier(torch.nn.Module):
    def __init__(self, data, sentences, label_set="sr"):
        super(AdpClassifier ,self).__init__()
        self.lm_model = data.model      # The underlying transformer model 
        self.label_set = label_set      # Determines which label set to take
        self.label_to_ix, self.labels = self.prep_label_set(sentences)  
    
        self.lm_hidden_size = self.lm_model.model.config.hidden_size
        # Single layer feed forward network acts as the supersense classification head
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
        processed_inp = self.lm_model.tokenizer(sentence["input_data"]["words"], is_split_into_words=True, return_tensors="pt")     # Outputs of the model tokenizer. A list of sub-words indices
        processed_inp["word_ids"] = processed_inp.word_ids()[1:-1]  # The special tokens are removed
        
        # This list contains all indices from the sub-word index list which correspond to 
        # adpositions. Note that the adpsitions need to be labeled.
        targets = []
        for t_ids in sentence["input_data"]["target_ids"]:
            # We only want to consider supersense and not discourse markers
            if sentence["input_data"]["scene_roles"][t_ids[0]] != "`":
                targets.append(t_ids)
            
        processed_inp["target_ids"] = targets

        return processed_inp



    def translate_label(self, sentence):
        """ Method to prepare the labels for the sentence. Each label is converted to its corresponding class id.
        Inputs
        -------------
        sentence: dict{dict}. A dictionary containing information about the words and targets for a sentence

        Output
        ----------
        processed_inp: list. List of labels ids occurring in the sentence
        """

        roles = [] 
        for ix, role in enumerate(sentence["input_data"][self.key]):
            if role == role and role != "`":
                roles.append(self.label_to_ix[role])
        
        return roles
            
            

    def get_mean_embeddings(self, emb, word_ids, targets):
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

        targ_ix = 0

        # Iterate over the word_ids
        for num, id in enumerate(word_ids):
            el = id.numpy()
            # take all ids corresponding to a target
            if el >= targets[targ_ix][0].numpy() and el <=targets[targ_ix][-1].numpy():
                idx_mean.append(num)
            elif el < targets[targ_ix][0].numpy():
                continue
            else:
                # Take mean of all embeddings from the releavant ids
                mean_vec = torch.mean(rel_emb[idx_mean,:], dim=0,keepdim=True)
                if final == None:
                    final = mean_vec
                else:
                    final = torch.cat((final,mean_vec),dim=0)
                targ_ix += 1
                idx_mean = []
                # Break when all ids are processed
                if targ_ix+1 > len(targets):
                    break
                elif el>=targets[targ_ix][0].numpy():
                    idx_mean.append(num)

        # Processing for the last target token
        if len(idx_mean) > 0:
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
        gold_split = list(map(self.translate_label, sentences_split))   # Prepares target labels 
        split_dataset = SNACSDataset(inps_split, gold_split)    # Prepare the dataset for the input and target
        # Create the dataloader from the dataset created above
        split_dataloader = data.DataLoader(dataset=split_dataset, shuffle=shuffle,  batch_size=1)

        return split_dataloader

    
    def get_weighted_losses(self,sentences):
        """ Method to obtain weights for the loss function .
        Input
        --------
        sentences: List[dict]. A split of data. The format as the same as returned by the 
                        `extract_sentences` method in `run.py`

        Output
        --------
        class_weights: torch.Tensor. A tensor of loss weights per class 
        """
        # The following lines generates a distribution of labels
        occurences = [0]*len(self.labels) 
        total = 0
        for sent in sentences:
            for role in sent["input_data"][self.key]:
                if role == role and role!='`':
                    occurences[self.label_to_ix[role]] += 1
                    total+=1
        # Inverting the distribution to obtain weights for loss
        class_weights = torch.Tensor(list(map(lambda x: 1-(x/total),occurences)))
        return class_weights



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
        params = list(self.lm_model.model.parameters()) + list(self.ffn.parameters())
        optimizer = optim.Adam(params, lr=lr)

        # We have an unbalanced dataset in terms of label distriution. Hence, having a weighted
        # loss is important. 
        class_weights = self.get_weighted_losses(sentences_train)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device)) # Initilizing the loss function
        
        best_metric = 0 # Variable to record the best performance recorded on dev split
        no_improv = 0   # Counter which counts the number of consecutive epochs for which the best dev score isnot beaten
       
        # Makes the model folder where models will be dumped
        model_folder = Path(MODEL_DIR,f"""{model_name.split("/")[-1]}/{SEED}/{self.label_set}/""")
        if save:
            Path.mkdir(model_folder, exist_ok=True, parents=True)

        best_state_dict = {}    # Retains best model and optimizer parameters

        # Training loop
        for ep in range(max_epochs):
            tr_loss = []
            print(f"Epoch {ep+1}")
            # iterating over data
            for inp,gold_lab in tqdm(train_dataloader):
                optimizer.zero_grad()

                # If no targets then we skip the sentence
                if len(gold_lab) == 0:
                    continue
                
                # Obtain embeddings for the sub-words
                embs = self.lm_model.model(inp['input_ids'].squeeze(0).to(device), attention_mask=inp['attention_mask'].squeeze(0).to(device))
                
                batch_loss = 0
                # Get embeddings for the target adpositions
                embs_mean = self.get_mean_embeddings(embs.last_hidden_state.squeeze(),inp['word_ids'],inp["target_ids"])
                pred = self.ffn(embs_mean)  # Forward pass
                batch_loss = loss_fn(pred,torch.Tensor(gold_lab).long().to(device)) #  Get loss
                
                tr_loss.append(batch_loss.item()/embs_mean.shape[0]) 
                # Update
                batch_loss.backward()
                optimizer.step()
 
            print(f"Training Loss: {np.mean(tr_loss)}")

            # Evalauate the dev split
            dev_metrics = self.eval_split(dev_dataloader)
            dev_f1_macro = dev_metrics["Macro"]['f1']   # The primary metric to decide the best model
            print(f"""Dev Entity F1 Macro: {dev_f1_macro}""")

            # If the dev metric is metric than the previous best, we store the parameters and update the
            # relevant variables
            if dev_f1_macro > best_metric:
                best_metric = dev_f1_macro
                no_improv = 0   # Reset the early stop counter
                test_metrics = self.eval_split(test_dataloader) # Evaluation is done on the test set (nopt displayed)

                # Best parameters are stored in the dictionary
                best_state_dict = {
                    "lm_model_state_dict" : copy.deepcopy(self.lm_model.model.state_dict()),
                    "ffn_state_dict"    : copy.deepcopy(self.ffn.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(optimizer.state_dict())
                    }
                
                # Dev and test metrics are dumped in the JSON format
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
        self.ffn.load_state_dict(checkpoint["ffn_state_dict"])



    def evaluate(self, sentences):
        """ Method to evaluate a set of sentences
        """
        dataloader = self.create_dataloader(sentences_dev)
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

        # iterate over data
        for inp, gold_lab in tqdm(loader):
            with torch.no_grad():
                if len(gold_lab) == 0:
                    continue
                # Forward pass as mentioned in the training loop
                embs = self.lm_model.model(inp['input_ids'].squeeze(0).to(device), attention_mask=inp['attention_mask'].squeeze(0).to(device))
                embs_mean = self.get_mean_embeddings(embs.last_hidden_state.squeeze(),inp['word_ids'],inp["target_ids"])
 
                out = self.ffn(embs_mean)
                preds = torch.argmax(out, dim=1)
                
                # Predictions and gold labels are recorded to compute the 
                # final metrics
                prediction_list.extend(preds.tolist())
                gold_label_list.extend(torch.Tensor(gold_lab).int().tolist())
       
        # Compute metrics
        metrics = self.calc_f1(prediction_list, gold_label_list)
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




    def calc_f1(self, pred_list, gold_list):
        """ Calculates several class-level and corpus-level F1 stats.
        Inputs
        --------
        pred_list: List. List of model predictions
        gold_list: List. List of gold predictions

        Output
        -------
        metrics: dict. Dictionary of class and corpus level F1 stats
        """
        # Intialize confusion matrics for each class
        c_matrices = []
        for ix in range(len(self.labels)):
            c_matrices.append(np.zeros((2,2)))

        # Populate the confusion matrices
        for ix in range(len(pred_list)):
            if pred_list[ix] == gold_list[ix]:
                c_matrices[pred_list[ix]][0][0] += 1
            else:
                c_matrices[pred_list[ix]][1][0] += 1
                c_matrices[gold_list[ix]][0][1] += 1

            
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

        # Compute Macro averages
        f1_mac /= cnt
        pr_mac /= cnt
        re_mac /= cnt

        metrics["Macro"] = {"f1":f1_mac, "precision": pr_mac, "recall": re_mac}

        return metrics



def parse_args(parser):
    parser.add_argument('--lang',default=f"guj", type =str)
    parser.add_argument('--data_dir',default=Path("./../data/annotated/"), type =str)  # Directory where data resides 
    parser.add_argument('--model_name', default="ai4bharat/indic-bert", type =str)  # HF model name
    parser.add_argument('--mode', default="train", type=str)    # "train" or "eval" mode
    parser.add_argument('--label_set', default="sr", type=str)  # "sr" or "fn" label set for Scene Role and Function
    parser.add_argument('--eval_model_path', default="./../models/indic-bert/42/sr/2", type=str)    # For eval which model to use
    parser.add_argument('--eval_split',default="dev",type=str)  # Which split are we evaluating on
    parser.add_argument('--e_stop', default=5,type=int)         # Number of epochs for early stopping
    parser.add_argument('--save_model', dest='save_model', action='store_true') # Saves models to directory if set
    parser.add_argument('--no-save_model', dest='save_model', action='store_false') # Doesn't save model to directory 
    parser.add_argument('--lr', default=0.00001, type=float)    # Learning rate
    parser.add_argument('--max_epochs', default=100, type=int)  # Max number of epochs to run training for
    parser.set_defaults(save_model=True)
    
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_args(parser)
    args = vars(parser.parse_args())

    lang = args['lang']    # Specifying language
    data_dir = args['data_dir'] # Data directory for the data
    model_name = args['model_name']     # Language Model to train on
    #model_name = "google/muril-large-cased" 
    mode = args['mode']
    label_set = args['label_set']

    # Processing the data to list format
    snacs_data = SNACSData(data_dir=data_dir, demo=False, show_stats=False, lang=lang, model_name=model_name)
    sentences = snacs_data.extract_sentences(tokenization_guj=True)

    # Preparing data splits
    sentences_train, sentences_inter = train_test_split(sentences, test_size=0.3, random_state=42)
    sentences_dev, sentences_test = train_test_split(sentences_inter, test_size=0.5, random_state=42)
    
    # Intializing the tagger
    classifier = AdpClassifier(snacs_data, sentences, label_set=label_set)
   
    # Training module
    if mode == "train":
        classifier.train(sentences_train, sentences_dev, sentences_test, model_name=model_name,\
                max_epochs=args['max_epochs'], lr=args['lr'], e_stop=args['e_stop'], save=args["save_model"])
    else:
        classifier.load_model(args['eval_model_path'])
        if args["eval_split"] == "dev":
            classifier.evaluate(sentences_dev)
        else:
            classifier.evaluate(sentences_test)
