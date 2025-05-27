import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '..', 'ns_lib'))

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_soft_device_placement(True)
print("GPUs used: ", gpus)
# tf.config.run_functions_eagerly(True)
import copy
from itertools import product
from train import main
import shutil as sh
import ns_lib as ns
from ns_lib.utils import NSParser
import numpy as np
import ast
import argparse
from model_utils import * 
import wandb 
wandb.login()

if __name__ == '__main__':

    use_logger = True
    use_WB = False
    load_model_ckpt = False
    load_kge_ckpt = False
    save_model_ckpt = True
    save_kge_ckpt = save_model_ckpt
    log_folder :str = "./experiments/runs/"
    ckpt_folder = "./../checkpoints/"
    data_path :str = "experiments/data"
    epochs: int = 100
    EARLY_STOPPING = True
    DATASET_NAME = ['countries_s3'] # ['ablation_d','ablation_d2','ablation_d3'] #['countries_s2','countries_s3','nations','kinship_family','pharmkg_small','pharmkg_full','wn18rr','nations','FB15k237']
    GROUNDER = ['backward_1_1'] #['backwardnoprune_1_1','backward_1_1','backward_1_2','backward_1_3','backward_2_1','backward_2_2','backward_2_3']  #  'domainbody','relationentity','full']
    KGE = ['complex'] # ["distmult", "transe","complex", "rotate"]
    MODEL_NAME = ['dcr'] #,['no_reasoner','dcr','sbr','r2n'] 
    RULE_MINER = ['amie','None'] 
    E = [100]
    DEPTH = [1]
    SEED = [[0]]# [[0,1,2,3,4]]
    NEG_PER_SIDE = [1]
    WEIGHT_LOSS = [.5]  
    DROPOUT = [0.0]
    R = [0.0]
    RR = [0.0]
    LR = [0.01]
    LR_SCHEDULER = ['plateau'] # None
    OPTIMIZER = ['adam'] #['None','adam']
    NUM_RULES = [1] 
    VALID_SIZE = [None]

    
    
    
    # This is in case I want to take inputs from the command line

    parser = argparse.ArgumentParser(description='Description of your script')  
    parser.add_argument("--d", default = None, help="dataset",nargs='+')
    parser.add_argument("--m", default = None, help="model",nargs='+')
    parser.add_argument("--g", default = None, help="grounder",nargs='+')
    parser.add_argument("--s", default = None, help="seed")

    args = parser.parse_args()
    if args.s is not None:
        SEED = [ast.literal_eval(args.s) ]
        assert isinstance(SEED, list) 
    if args.m is not None:
        MODEL_NAME = args.m
    if args.d is not None:
        DATASET_NAME = args.d
    if args.g is not None:
        GROUNDER = args.g

    del args.s
    del args.m
    del args.d  
    del args.g
    print('Running experiments for the following parameters:','DATASET_NAME:',DATASET_NAME,'GROUNDER:',GROUNDER,'MODEL_NAME:',MODEL_NAME,'SEED:',SEED)
    
    # Do the hparam search
    all_args = []
    for dataset_name,grounder, kge, model_name, rule_miner, e, dp, seed, neg, w_loss,  dropout, r, lr,lr_sched,optimizer, nr, rr in product(
            DATASET_NAME,GROUNDER, KGE, MODEL_NAME, RULE_MINER, E, DEPTH, SEED, NEG_PER_SIDE, WEIGHT_LOSS, DROPOUT, R,
            LR,LR_SCHEDULER,OPTIMIZER, NUM_RULES, RR ):  

        run_vars = (dataset_name,grounder, kge, model_name, rule_miner, neg, e)
        if not os.path.exists(os.path.join(data_path, dataset_name)):
            print('skipping, dataset not existing', run_vars)
            continue
        # Discern the datasets for which the grounders full, domainbody, and relationentity are too heavy to run
        if model_name == 'no_reasoner' and grounder not in {'backward_1_1'}:
            continue

        args.dataset_name = dataset_name
        args.grounder = grounder
        args.kge = kge
        args.model_name = model_name 
        args.rule_miner = rule_miner 
        args.seed = seed
        if (dataset_name == 'pharmkg_full' or dataset_name == 'wn18rr' or dataset_name == 'FB15k237'): # For heavy datasets, run only one seed
            args.seed = [0]
        args.kge_atom_embedding_size = e
        args.batch_size = 256 # Full batch only for explain.
        args.val_batch_size = 256
        if dataset_name in 'wn18rr' or dataset_name == 'FB15k237' or dataset_name == 'pharmkg_full' :
            args.test_batch_size = 1
        elif dataset_name == 'kinship_family':
            args.test_batch_size = 4
        elif dataset_name in 'countries_s3':
            args.test_batch_size = 128
        else:
            args.test_batch_size = 256
        args.facts_file = 'facts.txt'
        args.train_file = 'train.txt'  
        args.valid_file = 'valid.txt'
        args.test_file = 'test.txt'
        args.domain_file = 'domain2constants.txt'
        args.rules_file = 'rules.txt' 

        # Select the rules file
        rules_files = {'amie': 'rules_amie.txt', 'ncrl': 'rules_ncrl.txt', 'None': 'rules.txt'}
        args.rules_file = rules_files.get(rule_miner) or ValueError(f'Rule miner not recognized for {dataset_name}')
        if not os.path.exists(os.path.join(data_path, dataset_name, args.rules_file)):
            continue

        # Data params
        args.corrupt_mode = 'TAIL' if ('countries' in dataset_name or 'ablation' in dataset_name or 'test' in dataset_name) else 'HEAD_AND_TAIL'
        args.num_negatives = neg  
        args.valid_negatives = 100
        args.test_negatives = 1000 if (dataset_name=='FB15k237' or dataset_name=='wn18rr') else None # all possible negatives
        args.ragged = True
        args.format = "functional"
        args.engine_num_negatives = 0
        args.engine_num_adaptive_constants = 0
        args.constant_embedding_size = (
            2 * args.kge_atom_embedding_size
            if args.kge == "complex" or args.kge == "rotate"
            else args.kge_atom_embedding_size)
        args.predicate_embedding_size = (
            2 * args.kge_atom_embedding_size
            if args.kge == 'complex'
            else args.kge_atom_embedding_size)
        # KGE params
        args.dropout_rate_embedder = dropout
        args.kge_regularization = r
        # Model params
        args.learning_rate = lr
        args.lr_sched = lr_sched
        args.optimizer = 'adam'
        args.early_stopping = EARLY_STOPPING
        args.epochs = epochs #if not args.early_stopping else 1500
        args.num_rules = 0 if model_name == "no_reasoner"  else nr
        args.loss = "binary_crossentropy"
        args.weight_loss = w_loss
        args.cdcr_use_positional_embeddings = False
        args.cdcr_num_formulas = 3
        args.valid_frequency = 5 if not EARLY_STOPPING else 1
        args.resnet = True if 'ablation' not in dataset_name else False
        args.r2n_prediction_type = 'head' if 'ablation' in dataset_name else 'full'
        args.reasoner_depth = dp if nr > 0 else 0
        args.reasoner_regularization_factor = rr
        args.reasoner_formula_hidden_embedding_size = args.kge_atom_embedding_size
        args.reasoner_dropout_rate = dropout
        args.kge_dropout_rate = dropout
        args.reasoner_atom_embedding_size = args.kge_atom_embedding_size
        # DCR/R2N params
        args.signed = True
        args.temperature = 0.0
        args.aggregation_type = "max"
        args.filter_num_heads = 3
        args.filter_activity_regularization = 0.0
        # Other
        # args.adaptation_layer = "identity"  # "dense", "sigmoid","identity"
        # args.output_layer = "dense" # "wmc" or "kge" or "positive_dense" or "max"
        # args.relation_entity_grounder_max_elements = 20
        # args.semiring = "product"

        run_vars = (args.dataset_name,grounder, kge, model_name, rule_miner, neg, e, args.batch_size, args.val_batch_size, args.test_batch_size)
        args.keys_signature = ['dataset_name','grounder', 'kge', 'model_name', 'rule_miner','neg','e','train_batch_size','val_batch_size','test_batch_size']
        args.run_signature = '-'.join(f'{v}' for v in run_vars)    

        args.ckpt_folder = ckpt_folder
        args.load_model_ckpt = load_model_ckpt
        args.save_model_ckpt = save_model_ckpt
        args.load_kge_ckpt = load_kge_ckpt  
        args.save_kge_ckpt = save_kge_ckpt
        all_args.append(copy.deepcopy(args)) # append a hard copy of the args to the list of all_args




    def main_wrapper(args): 

        print("\nRun vars:", args.run_signature+'\n')

        if use_logger:
            logger = ns.utils.FileLogger(base_folder=log_folder)
            if logger.exists_experiment(args.__dict__):
                return

        for seed in args.seed:
            args.seed_run_i = seed
            print("Seed", seed, " in ", args.seed)
            if use_logger:
                date = logger.date
                log_filename_tmp = os.path.join(log_folder,'_tmp_log-{}-{}-seed_{}.csv'.format(args.run_signature,date,seed))
                if logger.exists_run(args.run_signature,seed):   
                    continue
                # else:
                #     print("Seed number ", seed, " not done. Exit")
                #     continue
            else:   
                log_filename_tmp = None

            train_eval_metrics,valid_eval_metrics, test_eval_metrics, training_info = main(data_path,log_filename_tmp,use_WB,args)
            

            if use_logger:
                # Include the results in the logger
                logged_data = copy.deepcopy(args)
                dicts_to_log = {'train': train_eval_metrics, 'valid': valid_eval_metrics, 'test': test_eval_metrics, 'training_info': training_info}
                # write the info about the results in the tmp file 
                logger.log(log_filename_tmp,logged_data.__dict__,dicts_to_log)
                # Rename to not be temporal anymore
                task_mrr = np.round(test_eval_metrics['task_mrr'],3)
                log_filename_run_name = os.path.join(log_folder,'indiv_runs', '_ind_log-{}-{}-{}-seed_{}.csv'.format(
                                                            args.run_signature,date,task_mrr,seed))
                logger.finalize_log_file(log_filename_tmp,log_filename_run_name)
        # If we have done all the seeds in args.seed, we can get the average results
        logger.get_avg_results(args.__dict__, args.run_signature,args.seed) if use_logger else None

                
    for args in all_args:
        print('Experiment number ', all_args.index(args), ' out of ', len(all_args), ' experiments.')
        main_wrapper(args)
