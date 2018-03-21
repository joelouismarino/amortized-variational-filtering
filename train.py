from config import run_config, train_config, data_config, model_config
from util.logging import init_log, load_checkpoint, save_checkpoint
from util.plotting import init_plot
from util.data.load_data import load_data
from lib.models import load_model
from util.optimizers import load_opt_sched
from util.train_val import train, validate

# initialize logging and plotting
# log_dir = init_log(run_config)
# init_plot(log_dir)

# load the data
train_data, val_data, test_data = load_data(data_config, run_config)

# load the model, optimizers
if run_config['resume_path']:
    model, optimizers, schedulers = load_checkpoint(run_config['resume_path'])
else:
    model = load_model(model_config)
    optimizers, schedulers = load_opt_sched(train_config, model)

# train the model
while True:
    train(train_data, model, optimizers, schedulers)
    if val_data:
        validate(val_data, model)
    save_checkpoint(model, optimizers, schedulers)
