import os

import numpy as np
import pandas as pd

from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

GLOBAL_SEED = 1234
np.random.seed(GLOBAL_SEED)

def training(model_data, force_train, model_dir, model_load=False, seed=1234, vector_size=None):
    
    train_corpus = [TaggedDocument(x, [str(y)]) for i, (x, y) in enumerate(model_data[['static_user_profile_list', 'Customer_ID']].values)]
    print(f'train_corpus[0]:{train_corpus[0]}')
    
    fixed_params = {'window':4, 'vector_size':64, 'ns_exponent': 0.254610887, 'min_count': 4, 'epochs': 30, 'alpha': 0.016691564, 'min_alpha': 0.000591644, 'seed': seed, 'dm': 0, 'dbow_words': 1, 'hs': 1}

    window = fixed_params['window']
    vector_size = vector_size if vector_size else fixed_params['vector_size']
    ns_exponent = fixed_params['ns_exponent']
    min_count = fixed_params['min_count']
    epochs = fixed_params['epochs']
    alpha = fixed_params['alpha']
    min_alpha = fixed_params['min_alpha']
    seed = fixed_params['seed']
    dm = fixed_params['dm']
    dbow_words = fixed_params['dbow_words']
    hs = fixed_params['hs']
    
    callbacks=[]

    print(f'epochs:{epochs}, window:{window}, vector_size:{vector_size}, ns_exponent:{ns_exponent}, min_count:{min_count}')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_file_name = f'{model_dir}/model_dm{dm}_vector_size{vector_size}_min_count{min_count}_epochs{epochs}_window{window}_ns_exponent{ns_exponent}_alpha{alpha}_min_alpha{min_alpha}_hs{hs}_seed{seed}.model'

    if (not model_load) and (not os.path.exists(model_file_name) or force_train):
        print('model training')
        model = Doc2Vec(train_corpus,
                        dm=dm, 
                        vector_size=vector_size, 
                        min_count=min_count, 
                        epochs=epochs,
                        workers=1, 
                        window=window,
                        ns_exponent=ns_exponent,
                        alpha=alpha,
                        min_alpha=min_alpha,
                        compute_loss=False,
                        hs=hs,
                        dbow_words=dbow_words,
                        seed=seed,
                        callbacks=callbacks
                        )
    
        model.save(model_file_name)
        print(f'Model Saved at: {model_file_name}')
    
        all_Customer_id_list = list(model_data.Customer_ID.unique())
        n_test_sample = 2000
        corpus_len = len(train_corpus)-1
        n_correct_prediction = np.zeros(21)
        for indi_n_sample in range(n_test_sample):
            Customer_id = all_Customer_id_list[indi_n_sample]
            index_Customer_id = all_Customer_id_list.index(Customer_id)
            model.random.seed(seed)
            inferred_vector = model.infer_vector(train_corpus[index_Customer_id].words)
            sim_Customer_id_list = model.docvecs.most_similar([inferred_vector], topn=20)
            for n_top in [1, 2, 3, 5, 10, 20]:
                for sim_Customer_id in sim_Customer_id_list[0:n_top]:
                    if sim_Customer_id[0] == Customer_id:
                        n_correct_prediction[n_top] +=1
                        break
        n_correct_prediction=n_correct_prediction/n_test_sample
        print(f'n_correct_prediction: {n_correct_prediction}')
    else:
        model = None
        try:
            print(f'loading stored model at: {model_file_name}')
            model = Doc2Vec.load(model_file_name) 
        except:
            pass

    return model