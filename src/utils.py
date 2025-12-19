import pickle as pkl 

def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        pkl.dump(model, f)

def load_data(data_path):
    with open('data/CA_1_0.pkl', 'rb') as f:
        df_train = pkl.load(f)
    return df_train 