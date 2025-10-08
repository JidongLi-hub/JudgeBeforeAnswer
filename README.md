
<center><img src="./images/JBA_logo.png" alt="描述" width="150" ></center>

# Judge Before Answer: Can MLLM Discern the False Premise in Question?
The original code of constructing JBA dataset is given in this repository, which is an evaluation set of false premise problems for MLLM.


# Quick Start
Find our JBA dataset in `dataset/Judge_Before_Answer.json`, image ids are from [`Visual Genome`](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html). 
Also, you can run `main.py` to constrcut your own JBA dataset.

Run `test.py` to generate test results for MLLM.

Run `evaluate.py` to evaluate the results and get metrics.