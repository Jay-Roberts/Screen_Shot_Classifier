import argparse
import os
import gmodel
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)
from tensorflow.contrib.training.python.training import hparam


def run_experiment(hparams):
    """Run the training and evaluate using the high level API"""
    
    # Get the train_input function
    train_input = lambda: gmodel.my_input_fn('train',file_dir = hparams.file_dir,
        num_epochs=hparams.num_epochs,
        batch_size=hparams.train_batch_size)

    # Don't shuffle evaluation data
    eval_input = lambda: gmodel.my_input_fn('val', file_dir= hparams.file_dir,
        batch_size=hparams.eval_batch_size,
        shuffle=False)

    # Set the traning specs
    train_spec = tf.estimator.TrainSpec(train_input,
                                        max_steps=hparams.train_steps)
    
    # Set up the configuration to run gmodel
    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(model_dir=hparams.job_dir)
    print('model dir {}'.format(run_config.model_dir))
    
    # Construct the estimator
    estimator = gmodel.construct_model(config = run_config)

    #Train the model
    estimator.train(train_input,
                    steps=hparams.train_steps)
    
    # Evaluate the model
    estimator.evaluate(eval_input,
                        steps = hparams.eval_steps,
                        name='gm_eval')
    

    # Save the model
    export_dir = hparams.job_dir+'/Saved_Estimators'
    #export_dir = 'Saved_Estimators/'
    print(export_dir)

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    with tf.Session(graph=tf.Graph()) as sess:

        builder.add_meta_graph_and_variables(sess,
               ['testit_tag'],
                )
        
    builder.save()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--file-dir',
        help='GCS or local paths to data. Must be organized data/gameID/{test,train,val}',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--num-epochs',
        help="""\
        Maximum number of training data epochs on which to train.
        If both --max-steps and --num-epochs are specified,
        the training job will run for --max-steps or --num-epochs,
        whichever occurs first. If unspecified will run for --max-steps.\
        """,
        type=int,
    )
    parser.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=1
    )
    parser.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=40
    )
    # Training arguments
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )

    # Argument to turn on all logging
    parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
    )
    # Experiment arguments
    parser.add_argument(
        '--train-steps',
        help="""\
        Steps to run the training job for. If --num-epochs is not specified,
        this must be. Otherwise the training job will run indefinitely.\
        """,
        type=int
    )
    parser.add_argument(
        '--eval-steps',
        help='Number of steps to run evalution for at each checkpoint',
        default=100,
        type=int
    )
   


    args = parser.parse_args()

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[args.verbosity] / 10)

    # Run the training job
    hparams=hparam.HParams(**args.__dict__)
    run_experiment(hparams)
