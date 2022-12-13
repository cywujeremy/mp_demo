import time
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from multiprocessing import Pool, cpu_count


### Modify Configurations Here to Test Different Scenarios
N_PROCESSES = 10
N_TREES_LIST = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]


def trainAdaBoostAsync(n_trees):
    
    with open('data.pkl', 'rb') as f:
        X_train, X_val, y_train, y_val = pickle.load(f)
    
    model = AdaBoostClassifier(n_estimators=n_trees)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    
    return model, acc


def trainAdaBoostSync(params):
    
    with open('data.pkl', 'rb') as f:
        X_train, X_val, y_train, y_val = pickle.load(f)
    
    for param in params:
        
        model = AdaBoostClassifier(n_estimators=param)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
    

if __name__ == '__main__':
    
    # asynchronous execution
    cores = cpu_count()
    start_time = time.time()
    
    with Pool(processes=N_PROCESSES) as p:
        model_parallel = p.map_async(trainAdaBoostAsync, N_TREES_LIST).get()
    print(f"Async Tasks Completed!")
    print('elapsed: ', round(time.time() - start_time, 6), 's')
    
    # synchronous execution
    start_time = time.time()
    trainAdaBoostSync(N_TREES_LIST)
    print(f"Sync Tasks Completed!")
    print('elapsed: ', round(time.time() - start_time, 6), 's')