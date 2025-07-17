def process_data_parameters(data_dict: dict) -> dict:
    processed_dict = dict()
    # Assertions
    assert "dataset_name" in data_dict.keys(), "Specify a dataset"
    
    #  Values
    processed_dict["dataset_name"] = data_dict["dataset_name"]
    processed_dict["downscaling_factor"] = data_dict.get("downscaling_factor",1)
    processed_dict["yarin_gal_uci_split_indices"] = data_dict.get("yarin_gal_uci_split_indices", 0)
    processed_dict["max_dataset_size"] = data_dict.get("max_dataset_size",1000)
    processed_dict["standardize"] = data_dict.get("standardize", True)
    processed_dict["select_timesteps"] = data_dict.get("select_timesteps", "zero")
    processed_dict["temporal_downscaling_factor"] = data_dict.get("temporal_downscaling_factor", 1)
    return processed_dict

def process_training_parameters(train_dict: dict)-> dict:
    processed_dict = dict()
    # Assertions
    assert "model" in train_dict.keys(), "Specify a model"
    assert "uncertainty_quantification" in train_dict.keys(), "Specify a UQ method" 

    # Values
    processed_dict["model"] = train_dict["model"]
    processed_dict["uncertainty_quantification"] = train_dict["uncertainty_quantification"]
    processed_dict["report_every"] = train_dict.get("report_every", 50)
    processed_dict["seed"] = train_dict.get("seed", [1234])
    processed_dict["backbone"] = train_dict.get("backbone", ["default"])
    processed_dict["batch_size"] = train_dict.get("batch_size", [128])
    processed_dict["eval_batch_size"] = train_dict.get("eval_batch_size", 512)
    processed_dict["n_epochs"] = train_dict.get("n_epochs", 1000)
    processed_dict["early_stopping"] = train_dict.get("early_stopping", 100)
    processed_dict["init"] = train_dict.get("init", "default")
    processed_dict["learning_rate"] = train_dict.get("learning_rate", 0.0001)
    processed_dict["lr_schedule"] = train_dict.get("lr_schedule", "step")
    processed_dict["optimizer"] = train_dict.get("optimizer", "adam")
    processed_dict["gradient_clipping"] = train_dict.get("gradient_clipping", 1)
    processed_dict["distributed_training"] = train_dict.get("distributed_training", False)
    processed_dict["alpha"] = train_dict.get("alpha", 0.05)
    processed_dict["n_samples_uq"] = train_dict.get("n_samples_uq", 100)
    processed_dict["weight_decay"] = train_dict.get("weight_decay", 0.0)

    # Model Parameters
    processed_dict["dropout"] = train_dict.get("dropout", [0.1])
    processed_dict["hidden_dim"] = train_dict.get("hidden_dim", [64])
    processed_dict["n_layers"] = train_dict.get("n_layers", [2])

    # Diffusion Parameters
    processed_dict["n_timesteps"] = train_dict.get("n_timesteps", [50])
    processed_dict["distributional_method"] = train_dict.get("distributional_method", ["mvnormal"])
    processed_dict["loss"] = train_dict.get("loss", ["crps"])
    processed_dict["gamma"] = train_dict.get("gamma", 1.0)
    processed_dict["rank"] = train_dict.get("rank",10)
    processed_dict["mvnormal_method"] = train_dict.get("mvnormal_method", ["lora"])
    processed_dict["concat_condition_diffusion"] = train_dict.get("concat_condition_diffusion", True)
    processed_dict["evaluate"] = train_dict.get("evaluate", True)
    processed_dict["x_T_sampling_method"] = train_dict.get("x_T_sampling_method", ["standard"])
    processed_dict["conditional_free_guidance_training"] = train_dict.get("conditional_free_guidance_training", False)
    processed_dict["ddim_sigma"] = train_dict.get("ddim_sigma", 1.0)
    processed_dict["noise_schedule"] = train_dict.get("noise_schedule", "linear")
    processed_dict["regressor"] = train_dict.get("regressor", None)

    return processed_dict
