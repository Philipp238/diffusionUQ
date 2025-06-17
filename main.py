# Main function to run the experiments.

import yaml
import numpy as np
import torch
import gc
from torch.utils.data import DataLoader, random_split
import pandas as pd
from time import time
import os
import sys
import datetime
import pathlib
import logging
import argparse
import configparser
import ast
import shutil
from data.data_utils import UCI_DATASET_NAMES, get_data, get_uci_data
from train import trainer, using
from utils import train_utils
from evaluate import start_evaluation
# from models import LA_Wrapper

# print(os.getcwd())
# sys.path[0] = os.getcwd()
# import utils

torch.autograd.set_detect_anomaly(False)

msg = 'Start main'

# initialize parser
parser = argparse.ArgumentParser(description=msg)
default_config = 'debug.ini'

parser.add_argument('-c', '--config', help='Name of the config file:', default=default_config)
parser.add_argument('-f', '--results_folder', help='Name of the results folder (only use if you only want to evaluate the models):', default=None)
args = parser.parse_args()

config_name = args.config
config = configparser.ConfigParser()
config.optionxform = str # otherwise capital letters will be transformed to lower case

config.read(os.path.join('config', config_name))
results_path = config['META']['results_path']
experiment_name = config['META']['experiment_name']

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'Using {device}.')


def construct_result_dict(entry_names, data_parameters_dict, training_parameters_dict):
    results_dict = {**{key: [] for key in data_parameters_dict[0].keys()},
                    **{key: [] for key in training_parameters_dict[0].keys()}}
    for entry_name in entry_names:
        results_dict[entry_name] = []
    return results_dict

def append_results_dict(results_dict, data_parameters, training_parameters, t_training):
    for key in data_parameters.keys():
        results_dict[key].append(data_parameters[key])
    for key in training_parameters.keys():
        results_dict[key].append(training_parameters[key])
    results_dict['t_training'].append(t_training)
    
def get_weight_filenames_directory(directory):
    # Get the names of all weight files in a directory 
    filenames = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.pt')]
    filenames = [filename for filename in filenames if (not filename[:-3].endswith('la_state'))]
    return filenames

def find_files_by_ending(folder_path, ending):
    files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(ending):
            files.append(filename)
    return files

if __name__ == '__main__':
    d_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
    directory = os.path.join(results_path, d_time + experiment_name)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    shutil.copy(os.path.join('config', config_name), directory)
    print(f'Created directory {directory}')

    # https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    logging.basicConfig(filename=os.path.join(directory, 'experiment.log'), level=logging.INFO, force=True)
    logging.info('Starting the logger.')
    logging.debug(f'Directory: {directory}')
    logging.debug(f'File: {__file__}')

    logging.info(f'Using {device}.')
    
    logging.info(using(''))

    logging.info(f'############### Starting experiment with config file {config_name} ###############')

    training_parameters_dict = dict(config.items("TRAININGPARAMETERS"))
    training_parameters_dict = {key: ast.literal_eval(training_parameters_dict[key]) for key in
                                training_parameters_dict.keys()}
    
    # In case you ONLY want to validate all models in a certain directory:
    # This prepares the filename_to_validate field in training_parameters_dict to contain the names of all weight files in the directory you want to validate
    if config['META'].get('only_validate', None):
        filename_to_validate = config['META']['only_validate']
        if not filename_to_validate.endswith('.pt'):
            filename_to_validate = get_weight_filenames_directory(os.path.join(results_path, filename_to_validate))
        else:
            filename_to_validate = os.path.join(results_path, filename_to_validate)
        training_parameters_dict['filename_to_validate'] = filename_to_validate
                
    # except_keys for keys that are coming as a list for each training process
    training_parameters_dict = train_utils.get_hyperparameters_combination(training_parameters_dict, 
                                                                           except_keys=['uno_out_channels', 'uno_scalings', 'uno_n_modes'])
    
    data_parameters_dict = dict(config.items("DATAPARAMETERS"))
    data_parameters_dict = {key: ast.literal_eval(data_parameters_dict[key]) for key in
                            data_parameters_dict.keys()}
    data_parameters_dict = train_utils.get_hyperparameters_combination(data_parameters_dict) # except_keys for keys that are coming as a list for each training process
    
    entry_names = ['t_training'] 
    
    results_dict = construct_result_dict(entry_names, data_parameters_dict, training_parameters_dict)
    data_dir = config['META']['data_path']

    for i, data_parameters in enumerate(data_parameters_dict):
        logging.info(f"###{i + 1} out of {len(data_parameters_dict)} data set parameter combinations ###")
        print(f'Data parameters: {data_parameters}')
        logging.info(f'Data parameters: {data_parameters}')
        
        dataset_name = data_parameters['dataset_name']
        dataset_size = data_parameters['max_dataset_size']
        if dataset_name in UCI_DATASET_NAMES:
            split = data_parameters['yarin_gal_uci_split_indices']
            validation_ratio_on_train_set = data_parameters["validation_ratio"] / (1 - data_parameters["validation_ratio"])
            uci_data = get_uci_data(dataset_name, 
                                    splits=split, 
                                    standardize=data_parameters['standardize'], 
                                    validation_ratio=validation_ratio_on_train_set)
            dataset, image_dim, label_dim = uci_data
        else:
            dataset, image_dim, label_dim = get_data(dataset_name, data_dir, dataset_size, data_parameters['standardize'], image_size=None)
        
        logging.info(using('After loading the datasets'))

        for i, training_parameters in enumerate(training_parameters_dict):
            logging.info(f"###{i + 1} out of {len(training_parameters_dict)} training parameter combinations ###")
            print(f'Training parameters: {training_parameters}')
            logging.info(f'Training parameters: {training_parameters}')
            
            seed = training_parameters['seed']
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            filename_ending = f"{data_parameters['dataset_name']}_{training_parameters['model']}_{training_parameters['uncertainty_quantification']}_"
            
            batch_size = training_parameters['batch_size']
            eval_batch_size = training_parameters['eval_batch_size']
            
            if dataset_name in UCI_DATASET_NAMES:
                logging.info(f"Using split-{split} for UCI dataset {dataset_name}") # type: ignore
                if data_parameters["validation_ratio"] > 0:
                    training_dataset, validation_dataset, test_dataset = dataset
                else:
                    training_dataset, test_dataset = dataset
                    validation_dataset = None
            else:
                logging.info(f"Use random split for dataset {dataset_name}")
                training_dataset, validation_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])

            train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
            if validation_dataset is not None:
                val_loader = DataLoader(validation_dataset, batch_size=eval_batch_size, shuffle=True)
            else:
                val_loader = None
            logging.info(using('After creating the dataloaders'))
            
            if training_parameters['regressor'] == "orig_CARD_pretrain":
                logging.info(f"Using pre-trained regressor from the CARD repo")
                model_path = os.path.join("models", "orig_CARD_pretrain", dataset_name, f"split_{split}", "aux_ckpt.pth")
                aux_states = torch.load(model_path)

                config_path = os.path.join("models", "orig_CARD_pretrain", dataset_name, f"split_{split}", "config.yml")
                with open(os.path.join(config_path), "r") as f:
                    card_config = yaml.unsafe_load(f)

                regressor = train_utils.setup_CARD_model(
                    image_dim=image_dim,
                    label_dim=label_dim,
                    hidden_layers=card_config.diffusion.nonlinear_guidance.hid_layers,
                    use_batchnorm=card_config.diffusion.nonlinear_guidance.use_batchnorm,
                    negative_slope=card_config.diffusion.nonlinear_guidance.negative_slope,
                    dropout_rate=card_config.diffusion.nonlinear_guidance.dropout_rate,
                ).to(device)

                regressor.load_state_dict(aux_states[0])
                regressor.eval()

                # Eval regressor on test set
                test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=True)
                test_performance = train_utils.evaluate_CARD_model(
                    model=regressor,
                    loader=test_loader,
                    device=device,
                    standardized=data_parameters['standardize'],
                )
                rmse = np.sqrt(test_performance)
                logging.info(f"Performance of pre-trained regressor on test set: RMSE={round(rmse, 2)}")
            elif training_parameters['regressor']:
                folder_path = os.path.join('models', training_parameters['regressor'])
                ini_file = find_files_by_ending(folder_path, '.ini')[0]
                weight_file = find_files_by_ending(folder_path, '.pt')[0]

                config = configparser.ConfigParser()
                config.read(os.path.join(folder_path, ini_file))      
                
                regressor_parameters_dict = dict(config.items("TRAININGPARAMETERS"))
                regressor_parameters_dict = {key: ast.literal_eval(regressor_parameters_dict[key]) for key in
                                        regressor_parameters_dict.keys()}
                # except_keys for keys that are coming as a list for each training process
                regressor_parameters_dict = train_utils.get_hyperparameters_combination(regressor_parameters_dict, 
                                                                                except_keys=['uno_out_channels', 'uno_scalings', 'uno_n_modes'])
                
                
                regressor = train_utils.setup_model(regressor_parameters_dict[0], device, image_dim, label_dim)
                train_utils.resume(regressor, os.path.join(folder_path, weight_file))
                regressor.eval()
            else:
                regressor = None
            
            if training_parameters.get('filename_to_validate', None):
                # In case you ONLY want to validate all models in a certain directory; loads the model (instead of training it)
                # in_channels = next(iter(train_loader))[0].shape[1]
                # out_channels = next(iter(train_loader))[1].shape[1]
                
                model = train_utils.setup_model(training_parameters, device, image_dim, label_dim)
                filename = training_parameters['filename_to_validate']
                if training_parameters['uncertainty_quantification'] == 'laplace':
                    raise NotImplementedError('Laplace not implemented yet.')
                    model = LA_Wrapper(
                                model,
                                n_samples=training_parameters["n_samples_uq"],
                                method="last_layer",
                                hessian_structure="full",
                                optimize=True,
                            )
                train_utils.resume(model, filename)
                t_training = -1
            else:
                # In case you want to train the models
                t_0 = time()
                d_time_train = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                model, filename = trainer(train_loader, val_loader, directory=directory, training_parameters=training_parameters,
                                          data_parameters = data_parameters, logging=logging, filename_ending=filename_ending, d_time=d_time_train,
                                          image_dim=image_dim, label_dim=label_dim, results_dict=results_dict, regressor=regressor)
                            
                t_1 = time()
                t_training = np.round(t_1 - t_0, 3)
                logging.info(f'Training the model took {t_training}s.')
                t_0 = time()
                torch.cuda.empty_cache()
                t_1 = time()
                logging.info(f'Emptying the cuda cache took {np.round(t_1 - t_0, 3)}s.')
            
            train_loader = DataLoader(training_dataset, batch_size=eval_batch_size, shuffle=True)
            if validation_dataset is not None:
                val_loader = DataLoader(validation_dataset, batch_size=eval_batch_size, shuffle=True)
            else:
                val_loader = None
            test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=True)
            
            # Fix the seed again just that the evaluation without training yields the same results as training + evaluation
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            if training_parameters['evaluate']:
                start_evaluation(model, training_parameters, data_parameters, train_loader, val_loader, 
                                test_loader, results_dict, device, logging, filename, regressor)
                            
                append_results_dict(results_dict, data_parameters, training_parameters, t_training)
                results_pd = pd.DataFrame(results_dict)
                results_pd.T.to_csv(os.path.join(directory, 'test.csv'))
                
                logging.info(using('After validation'))
                
                del model
                torch.cuda.empty_cache()
                gc.collect()