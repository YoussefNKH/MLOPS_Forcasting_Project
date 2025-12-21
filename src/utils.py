import pickle as pkl 
import os 
import glob 
import re 

def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        pkl.dump(model, f)

def load_data(data_path):
    with open('data/CA_1_0.pkl', 'rb') as f:
        df_train = pkl.load(f)
    return df_train 

def get_latest_data_file(data_dir="data", file_pattern="CA_1_*.pkl"):

    search_path = os.path.join(data_dir, file_pattern)
    files = glob.glob(search_path)

    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {search_path}")

    # Regex to extract the digit(s) between 'CA_1_' and '.pkl'
    # This handles CA_1_0.pkl, CA_1_1.pkl, CA_1_10.pkl, etc.
    regex_pattern = re.compile(r"CA_1_(\d+)\.pkl")

    max_index = -1
    latest_file_path = None

    print(f"Scanning directory '{data_dir}' for files...")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        match = regex_pattern.search(filename)
        if match:
            # Extract the number (e.g., '3' from 'CA_1_3.pkl')
            current_index = int(match.group(1))
            
            # Check if this is the highest number we've seen so far
            if current_index > max_index:
                max_index = current_index
                latest_file_path = file_path
    
    if latest_file_path is None:
        raise ValueError(f"Files found {files}, but none matched the expected numerical format (CA_1_X.pkl).")

    print(f"-> Selected latest file (Index {max_index}): {latest_file_path}")
    return latest_file_path