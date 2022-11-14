Toy examples to evaluate different uncertainty estimation methods for neural networks.

To get started, go inside ```mlp/create_data.py```, change the directory where you'd like to store your toy dataset at the bottom of the script, and run this script. Then, there are three current implementations:
    1. An MLP predictor that does not estimate variance (```mlp/driver.py```)
    2. An MLP predictor that predicts the mean of a Gaussian and a variance (```mlp/gaussian_driver.py```)
    3. A predictor that uses an ensemble of MLPs to predict single values, from which we obtain a std deviation of the predictions to quantify uncertainty (```mlp/ensemble_driver.py```)

To run any of these files, first go into the scripts and change the directories for the dataset at the bottom of the script. Also, change the value of ```USE_WANDB``` depending on your preference. Leaving it to true will require WANDB to be first setup, which you can do by following instructions on their website.