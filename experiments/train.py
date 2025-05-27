import os
import tensorflow as tf
import ns_lib as ns
from itertools import product
import numpy as np
from os.path import join
import random
import pickle
from typing import List, Tuple, Dict

from dataset import KGCDataHandler
from model import CollectiveModel
# from keras.callbacks import CSVLogger
from ns_lib.logic.commons import Atom, Domain, FOL, Rule, RuleLoader
from ns_lib.grounding.grounder_factory import BuildGrounder
from ns_lib.utils import MMapModelCheckpoint, KgeLossFactory, get_arg, load_model_weights
import time
from model_utils import * 
import wandb
from wandb.integration.keras import WandbCallback
from wandb.integration.keras import WandbMetricsLogger
from ns_lib.utils import save_embeddings_from_model

explain_enabled: bool = False



def main(data_path, log_filename, use_WB, args):

    # # Start the TensorFlow Profiler server
    # tf.profiler.experimental.server.start(6009)  # Choose any available port, e.g., 6009

    print('\nARGS', args,'\n')
    seed = get_arg(args, 'seed_run_i', 0)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    ragged = get_arg(args, 'ragged', None, True)
    start_train = time.time()



    # DATASET PREPARATION
    data_handler = KGCDataHandler(
        dataset_name=args.dataset_name,
        base_path=data_path,
        format=get_arg(args, 'format', None, True),
        domain_file= args.domain_file,
        train_file= args.train_file,
        valid_file=args.valid_file,
        test_file= args.test_file,
        fact_file= args.facts_file)
    
    dataset_train = data_handler.get_dataset(split="train",number_negatives=args.num_negatives)
    dataset_valid = data_handler.get_dataset(split="valid",number_negatives=args.valid_negatives, corrupt_mode=args.corrupt_mode)
    dataset_test = data_handler.get_dataset(split="test",  number_negatives=args.test_negatives,  corrupt_mode=args.corrupt_mode)
    if explain_enabled and enable_rules and (args.model_name == 'dcr' or args.model_name == 'cdcr'):
        dataset_test_positive_only = data_handler.get_dataset(split="test", number_negatives=0, corrupt_mode=args.corrupt_mode)

    fol = data_handler.fol
    domain2adaptive_constants: Dict[str, List[str]] = None
    dot_product = get_arg(args, 'engine_dot_product', False)

    num_adaptive_constants = get_arg(args, 'engine_num_adaptive_constants', 0)


    # DEFINING RULES AND GROUNDING ENGINE
    rules = []
    engine = None

    enable_rules = (args.reasoner_depth > 0 and args.num_rules > 0)
    if enable_rules:
        rules = ns.utils.read_rules(join(data_path, args.dataset_name, args.rules_file),args)
        facts = list(data_handler.train_known_facts_set)
        engine = BuildGrounder(args, rules, facts, fol, domain2adaptive_constants)
    serializer = ns.serializer.LogicSerializerFast(
        predicates=fol.predicates, domains=fol.domains,
        constant2domain_name=fol.constant2domain_name,
        domain2adaptive_constants=domain2adaptive_constants)



    # DATA GENERATORS
    print('***********Generating train data**************')
    start = time.time()
    data_gen_train = ns.dataset.DataGenerator(
        dataset_train, fol, serializer, engine,
        batch_size=args.batch_size, ragged=ragged)
    end = time.time()
    args.time_ground_train = np.round(end - start,2)
    print("Time to create data generator train: ", np.round(end - start,2),'\n************************************')
    start = time.time()
    data_gen_valid = ns.dataset.DataGenerator(
       dataset_valid, fol, serializer, engine,
       batch_size=args.val_batch_size, ragged=ragged)
    end = time.time()
    args.time_ground_valid = np.round(end - start,2)
    print("Time to create data generator valid: ",  np.round(end - start,2),'\n************************************') 

    start = time.time()
    data_gen_test = ns.dataset.DataGenerator(
        dataset_test, fol, serializer, engine,
        batch_size=args.test_batch_size, ragged=ragged)
    end = time.time()
    args.time_ground_test = np.round(end- start,2)
    print("Time to create data generator test: ",  np.round(end - start,2),'\n************************************')


    # COMPILING MODEL
    model = CollectiveModel(
        fol, rules,
        kge=args.kge,
        kge_regularization=args.kge_regularization,
        model_name=get_arg(args, 'model_name', 'dcr'),
        constant_embedding_size=args.constant_embedding_size,
        predicate_embedding_size=args.predicate_embedding_size,
        kge_atom_embedding_size=args.kge_atom_embedding_size,
        kge_dropout_rate=args.kge_dropout_rate,
        reasoner_single_model=get_arg(args, 'reasoner_single_model', False),
        reasoner_atom_embedding_size=args.reasoner_atom_embedding_size,
        reasoner_formula_hidden_embedding_size=args.reasoner_formula_hidden_embedding_size,
        reasoner_regularization=args.reasoner_regularization_factor,
        reasoner_dropout_rate=args.reasoner_dropout_rate,
        reasoner_depth=args.reasoner_depth,
        aggregation_type=args.aggregation_type,
        signed=args.signed,
        resnet=get_arg(args, 'resnet', False),
        embedding_resnet=get_arg(args, 'embedding_resnet', False),
        temperature=args.temperature,
        filter_num_heads=args.filter_num_heads,
        filter_activity_regularization=args.filter_activity_regularization,
        num_adaptive_constants=num_adaptive_constants,
        dot_product=dot_product,
        cdcr_use_positional_embeddings=get_arg(
            args, 'cdcr_use_positional_embeddings', True),
        cdcr_num_formulas=get_arg(args, 'cdcr_num_formulas', 3),
        r2n_prediction_type=get_arg(args, 'r2n_prediction_type', 'full'),
    )


    #LOSS
    loss_name = get_arg(args, 'loss', 'binary_crossentropy')
    loss = KgeLossFactory(loss_name)

    loss = { 'concept': loss, 'task': loss}
    loss_weights = { 'concept': 1-args.weight_loss, 'task': args.weight_loss  }

    metrics = {
        'concept': [
            ns.utils.MRRMetric(),
            ns.utils.HitsMetric(1),
            ns.utils.HitsMetric(3),
            ns.utils.HitsMetric(10)
        ],
        'task': [
            ns.utils.MRRMetric(),
            ns.utils.HitsMetric(1),
            ns.utils.HitsMetric(3),
            ns.utils.HitsMetric(10)
        ]
    }

    optimizer,lr_scheduler = choose_optimizer_scheduler(args.optimizer,args.lr_sched,args.learning_rate)
    model.compile(optimizer=optimizer,
                    loss=loss,
                    loss_weights = loss_weights,
                    metrics=metrics,
                    run_eagerly=False)



    # CHECKPOINT HANDLING

    # create model's layers and weights to be able to load the weights 
    if not model.built:
        _ = model(next(iter(data_gen_train))[0])  # force building the model.

    # LOAD CKPT
    assert not args.load_model_ckpt or not args.load_kge_ckpt, "Only one of ckpt_load and load_kge_ckpt can be set."
    ckpt_filepath = os.path.join(args.ckpt_folder, args.run_signature+'_seed_'+str(seed), args.run_signature+'_seed_'+str(seed))

    # If checkpoint_load is not None, try to load the weights
    if args.load_model_ckpt or args.load_kge_ckpt:
        path_ = ckpt_filepath if args.load_model_ckpt else ckpt_filepath+'_kge_model'
        success = load_model_weights(model, path_, verbose=True)

        # update the arg that was true
        args.load_model_ckpt = success if args.load_model_ckpt else args.load_model_ckpt
        args.load_kge_ckpt = success if args.load_kge_ckpt else args.load_kge_ckpt



    # CALLBACKS
    callbacks = []
    if log_filename is not None:
        csv_logger = ns.utils.CustomCSVLogger(log_filename, append=True, separator=';') 
        callbacks.append(csv_logger)

    if args.lr_sched=='plateau':
      callbacks.append(lr_scheduler)
    
    if args.early_stopping:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=50,
            verbose=1)
        callbacks.append(early_stopping)
    
    model_checkpoint = MMapModelCheckpoint(
        model=model,
        monitor='val_task_mrr',
        filepath= ckpt_filepath if args.save_model_ckpt else None,
        save_best_only=True,
        save_weights_only=True,
        name='model'
    )

    kge_model_checkpoint = MMapModelCheckpoint(
        model=model.kge_model,
        monitor='val_concept_mrr',
        filepath= ckpt_filepath+'_kge_model' if args.save_kge_ckpt else None,
        save_best_only=True,
        save_weights_only=True,
        name='kge_model'
    )

    callbacks.append(model_checkpoint)
    callbacks.append(kge_model_checkpoint)

    # # Set up logging for TensorBoard
    # import datetime
    # log_dir = os.path.join("./../logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=(1, 19))
    # callbacks.append(tensorboard_callback)
    
    # Initialize a W&B run
    if use_WB:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dir = os.path.join(current_dir, '../..')
        run = wandb.init(project = "Grounders-exp", name=args.run_signature,
                dir=dir,  config = dict(
                shuffle_buffer = 1024,
                batch_size = args.batch_size,
                learning_rate = args.learning_rate,
                epochs = args.epochs)) 
        callbacks.append(WandbMetricsLogger(log_freq=10))



    # TRAIN
    do_training = args.epochs > 0 #and not (args.load_model_ckpt or args.load_kge_ckpt):
    if do_training: 

        history = model.fit(data_gen_train,
                epochs=args.epochs,
                callbacks=callbacks,
                validation_data=data_gen_valid,
                validation_freq=args.valid_frequency)

        end_train = time.time()

        # Close the W&B run
        if use_WB:
            run.finish()
    
        args.time_train = np.round(end_train - start_train,2)
        print('Training time:', np.round(end_train - start_train,2), 'seconds')

        # Restore the best weights after training
        kge_model_checkpoint.restore_weights() if args.load_kge_ckpt else model_checkpoint.restore_weights()
        model_checkpoint.restore_weights()
        model_checkpoint.write_train_time(args.time_train)

    else:
        if use_WB:
            run.finish
        args.time_train = 0
    
    # save_embeddings_from_model(model, fol, serializer, save_dir="./../embeddings/"+args.run_signature)



    # EVALUATION
    print("\nEvaluation train", flush=True)
    train_metrics = model.evaluate(data_gen_train)#,train_data=True,testing=True) 
    print("\nEvaluation val", flush=True)
    if do_training:
        valid_metrics =  model.evaluate(data_gen_valid)#,val_data=True,testing=True) 
    else:
        valid_metrics = [0.0]*len(train_metrics)    
    print("\nEvaluation test", flush=True)
    start_inf = time.time()
    test_metrics  =  model.evaluate(data_gen_test)#,test_data=True,testing=True)
    end_inf = time.time()
    args.time_inference = np.round(end_inf - start_inf,2)
    print('Inference time:', np.round(end_inf - start_inf,2), 'seconds')

    print('\nMetrics names:',model.metrics_names)
    train_eval_metrics = dict(zip(model.metrics_names,train_metrics))
    valid_eval_metrics = dict(zip(model.metrics_names,valid_metrics))
    test_eval_metrics = dict(zip(model.metrics_names,test_metrics))
    training_info = history.history if do_training else None

    print('\nMetrics:',train_eval_metrics.keys()) 
    print('\nResults',
          '\nTrain', np.round(train_metrics,3),
          '\nVal', np.round(valid_metrics,3),
          '\nTest', np.round(test_metrics,3),
          flush=True)


    if explain_enabled and enable_rules and (args.model_name == 'dcr' or args.model_name == 'cdcr'):
        model.explain_mode(True)
        print('\nExplain Train', flush=True)
        print(model.predict(data_gen_train)[-1])

        print('\nExplain Test', flush=True)
        data_gen_test_explain = ns.dataset.DataGenerator(
            dataset_test, fol, serializer, engine, batch_size=-1, ragged=ragged)
        print(model.predict(data_gen_test_explain)[-1])

        data_gen_test_positive_only = ns.dataset.DataGenerator(
            dataset_test_positive_only, fol, serializer, engine,
            batch_size=args.test_batch_size, ragged=ragged)
        for r in model.reasoning[-1].rule_embedders.values():
            r._verbose=True
        print(model.predict(data_gen_test_positive_only)[-1])
    return train_eval_metrics,valid_eval_metrics, test_eval_metrics, training_info