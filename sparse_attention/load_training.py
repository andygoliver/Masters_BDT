import numpy as np
import h5py
import sys
import os

def main(train_files_dir, training_split= 0.1, classes= ['g', 'q', 'w', 'z', 't'], 
         positive_class = 't', save: bool = False, output_dir= "/work/aoliver/BDT/sparse_attention/data/"):

    x_train_full, y_train_full = load_files_in_directory(train_files_dir, nb_files = 4)
    y_train_full = [1*(np.argmax(i) == classes.index(positive_class)) for i in y_train_full]
    print('==================')

    print("Splitting training data")
    split_events = int(len(x_train_full)*training_split)

    x_train_small = x_train_full[:split_events]
    x_train_large =  x_train_full[split_events:]

    y_train_small = y_train_full[:split_events]
    y_train_large =  y_train_full[split_events:]
    print("Successfully split training data.")
    print('==================')

    print("Making data for the small model...")
    x_constituent_small, y_constituent_small, structure_memory_small = make_sparse_attention_data(x_train_small, y_train_small)
    x_constituent_large, structure_memory_large = make_sparse_attention_x_data(x_train_large, y_train_large)

    if save:
        print(f"Data successfully made: now saving arrays to '{output_dir}'")
        np.save(os.path.join(output_dir, "x_constituent_small"), x_constituent_small)
        np.save(os.path.join(output_dir, "y_constituent_small"), y_constituent_small)
        np.save(os.path.join(output_dir, "structure_memory_small"), structure_memory_small)

        np.save(os.path.join(output_dir, "x_constituent_large"), x_constituent_large)
        # np.save(os.path.join(output_dir, "y_constituent_large"), y_constituent_large)
        np.save(os.path.join(output_dir, "y_train_large"), y_train_large)
        np.save(os.path.join(output_dir, "structure_memory_large"), structure_memory_large)
    print("Process complete!")

    return x_constituent_small, y_constituent_small, structure_memory_small, x_constituent_large, structure_memory_large, y_train_large

def get_file_paths(data_file_dir: str) -> list:
    """Gets path to the data files inside a given directory.

    Args:
        data_file_dir: Path to directory containing the data files.

    Returns:
        Array of paths to the data files themselves.
    """
    file_names = os.listdir(data_file_dir)
    file_paths = [os.path.join(data_file_dir, file_name) for file_name in file_names]

    return file_paths

def load_data(data_path: str) -> tuple([np.ndarray, np.ndarray]):
    """Loads the full dataset."""
    data = h5py.File(data_path)
    x_data = data["jetConstituentList"][:, :, :]
    y_data = data["jets"][:, -6:-1]

    return x_data, y_data

def load_files_in_directory(data_file_dir: str, nb_files: int = None, verbosity: int = 1):
    data_file_paths = get_file_paths(data_file_dir)

    if nb_files == None:
        max_files = len(data_file_paths)
    else:
        max_files = nb_files

    if verbosity > 0:
        print('\nReading files...')

    # x_data, y_data = select_features(args.type, data_file_paths[0])
    x_data, y_data = load_data(data_file_paths[0])
    n_file = 1
    
    if verbosity > 0:
        sys.stdout.write('\r')
        sys.stdout.write(f"Read file [{n_file}/{max_files}]")
        sys.stdout.flush()

    for file_path in data_file_paths[1:nb_files]:
        n_file +=1

        # add_x_data, add_y_data = select_features(args.type, file_path)
        add_x_data, add_y_data = load_data(file_path)
        x_data = np.concatenate((x_data, add_x_data), axis=0)
        y_data = np.concatenate((y_data, add_y_data), axis=0)

        if verbosity > 0:
            sys.stdout.write('\r')
            sys.stdout.write(f"Read file [{n_file}/{max_files}]")
            sys.stdout.flush()
    
    if verbosity > 0:
        print('\nAll files read')

    return x_data, y_data

def make_sparse_attention_data(x_data, y_data):
    boolean_mask = np.any(x_data, axis = 2)
    structure_memory = boolean_mask.sum(axis=1)
    x_constituent_data = x_data[boolean_mask, :]

    y_constituent_data = np.array([], dtype=int)
    for i in range(len(structure_memory)):
        y_constituent_data = np.append(y_constituent_data, structure_memory[i]*[y_data[i]])

    return x_constituent_data, y_constituent_data, structure_memory

def make_sparse_attention_x_data(x_data):
    boolean_mask = np.any(x_data, axis = 2)
    structure_memory = boolean_mask.sum(axis=1)
    x_constituent_data = x_data[boolean_mask, :]

    return x_constituent_data, structure_memory

if __name__ == "__main__":
    train_files_dir = "/work/aoliver/Data/train/"
    output_dir = "data/"
    training_split = 0.1

    classes = ['g', 'q', 'w', 'z', 't']
    positive_class = 't'
    main(train_files_dir, training_split=training_split, classes=classes, 
         positive_class=positive_class, save= True, output_dir= output_dir)