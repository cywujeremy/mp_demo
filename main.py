import time
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from multiprocessing import Pool, cpu_count


### Modify Configurations Here to Test Different Scenarios
N_PROCESSES = 10
N_TREES_LIST = [400, 400, 400, 400, 400, 400, 400, 400, 400, 400]


def trainRFAsync(n_trees):
    
    with open('data.pkl', 'rb') as f:
        X_train, X_val, y_train, y_val = pickle.load(f)
    
    model = RandomForestClassifier(n_estimators=n_trees, n_jobs=1)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    
    return model, acc


def trainRFSync(params):
    
    with open('data.pkl', 'rb') as f:
        X_train, X_val, y_train, y_val = pickle.load(f)
    
    for param in params:
        
        model = RandomForestClassifier(n_estimators=param, n_jobs=1)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
    

if __name__ == '__main__':
    
    # asynchronous execution
    cores = cpu_count()
    pool = Pool(processes=N_PROCESSES)
    start_time = time.time()
    with pool as p:
        model_parallel = p.map_async(trainRFAsync, N_TREES_LIST).get()
    print(f"Async Tasks Completed!")
    print('elapsed: ', round(time.time() - start_time, 6), 's')
    
    # synchrounus execution
    start_time = time.time()
    trainRFSync(N_TREES_LIST)
    print(f"Sync Tasks Completed!")
    print('elapsed: ', round(time.time() - start_time, 6), 's')