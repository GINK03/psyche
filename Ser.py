import pickle
import gzip
import pandas as pd


def ser_pickle(**var):
    name_type = {}
    for name, obj in var.items():
        #print(name, type(obj))
        if isinstance(obj, pd.DataFrame):
            print(name, 'is pd.DataFrame')
            obj.to_pickle(f'/tmp/{name}')
            name_type[name] = 'pd.DataFrame'
        else:
            with open(f'/tmp/{name}', 'wb') as fp:
                fp.write(pickle.dumps(obj))
            name_type[name] = 'other'
    with open(f'/tmp/name_type', 'wb') as fp:
        fp.write(pickle.dumps(name_type))

def deser_pickle():
    with open(f'/tmp/name_type', 'rb') as fp:
        name_type = pickle.load(fp)

    ret = {}
    for name, type in name_type.items():
        if type == 'pd.DataFrame':
            print(name, 'is pd.DataFrame')
            df = pd.read_pickle(f'/tmp/{name}')
            ret[name] = df
        else:
            with open(f'/tmp/{name}', 'rb') as fp:
                obj = pickle.load(fp)
            ret[name] = obj
    return ret 
