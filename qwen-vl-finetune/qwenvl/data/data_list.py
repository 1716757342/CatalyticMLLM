# qwenvl/data/data_list.py

"""
Dataset configuration file
Define paths and configuration information for different datasets
"""

def data_list(dataset_names):
    """
    Return dataset configuration information by dataset name
    
    Args:
        dataset_names: Dataset name list
        
    Returns:
        Dataset configuration information list
    """
    
    # Dataset configuration mapping
    dataset_configs = {
        "MOLECULE_RELAXED_ENERGY": {
            "annotation_path": "/path/to/training_data.json",
            "data_type": "molecule",
            "task_type": "energy_prediction"
        },
        "MOLECULE_RELAXED_ENERGY_CELL": {
            "annotation_path": "/path/to/training_data.json",
            "data_type": "molecule_with_cell",
            "task_type": "energy_prediction"
        },
        "MOLECULE_RELAXED_ENERGY_CELL_24k": {
            "annotation_path": "/path/to/training_data.json",
            "data_type": "molecule_with_cell",
            "task_type": "energy_prediction"
        }
    }
    
    # Build the result list
    result = []
    for name in dataset_names:
        if name in dataset_configs:
            result.append(dataset_configs[name])
        else:
            # If the dataset name is not in the predefined list, create a default configuration
            result.append({
                "annotation_path": f"/path/to/{name}.json",
                "data_type": "unknown",
                "task_type": "unknown"
            })
            print(f"Warning: Unknown dataset '{name}', using default config")
    
    return result
