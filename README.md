# multiprocessing_demo

This repository is a demonstration of using process-based parallelism in machine learning model training.
In certain cases, properly splitting computation tasks into processes can significantly save training time.

As an example for demonstration purpose, we set up the benchmark task as training 10 random forest models on a training dataset with size `(7000, 1000)` and running the inference on the validation dataset with size `(3000, 1000)`. We run this task in both asynchronous and synchronous way and record the time elapsed to compare the efficiency.

## Usage

* Clone the repository by running:

```{shell}
$ git clone git@github.com:cywujeremy/mp_demo.git
```

* On the root directory of the repo, run the following command to execute the test:

```{shell}
$ python main.py
```