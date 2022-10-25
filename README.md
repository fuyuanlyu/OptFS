# OptFS
This repository contains PyTorch Implementation of WWW 2023 submission paper:
  - **OptFS**: Optimizing Feature Set for Click-Through Rate Prediction

### Data Preprocessing

You can prepare the Criteo data in the following format. Avazu and KDD12 datasets can be preprocessed by calling its own python file.

```
python datatransform/criteo2tf.py --store_stat --stats PATH_TO_STORE_STATS
		--dataset RAW_DATASET_FILE --record PATH_TO_PROCESSED_DATASET \
		--threshold 2 --ratio 0.8 0.1 0.1 \
```

Then you can find a `stats` folder under the `PATH_TO_STORE_STATS` folder and your processed files in the tfrecord format under the `PATH_TO_PROCESSED_DATASET` folder. 

### Run

Running Backbone Models:
```
python -u trainer.py $YOUR_DATASET $YOUR_MODEL \
        --feature $NUM_OF_FEATURES --field $NUM_OF_FIELDS \
        --data_dir $PATH_TO_PROCESSED_DATASET \
        --cuda 0 --lr $LR --l2 $L2
```

You can choose `YOUR_DATASET` from \{Criteo, Avazu, KDD12\} and `YOUR_MODEL` from \{FM, DeeepFM, DCN, IPNN\}


Running OptFS Models:
```
python -u maskTrainer.py $YOUR_DATASET $YOUR_MODEL \
        --feature $NUM_OF_FEATURES --field $NUM_OF_FIELDS \
        --data_dir $PATH_TO_PROCESSED_DATASET \
        --cuda 0 --lr $LR --l2 $L2 \
        --reg_lambda $LAMBDA --final_temp $TEMP \
        --max_epoch $EPOCH --rewind_EPOCH $REWIND
```

### Hyperparameter Settings

Here we list the hyper-parameters we used in the following table.

| Model\Dataset | Criteo | Avazu | KDD12 |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| FM            | _lr_=3e-4, l<sup>2</sup>=1e-5, _lambda_=2e-9, _temp_=1000, _epoch_=10, _rewind_=1 | _lr_=3e-4, l<sup>2</sup>=1e-6, _lambda_=2e-9, _temp_=5000, _epoch_=5, _rewind_=4 | _lr_=3e-5, l<sup>2</sup>=1e-5, _lambda_=2e-9, _temp_=1000, _epoch_=10, _rewind_=0 |
| DeepFM        | _lr_=3e-4, l<sup>2</sup>=3e-5, _lambda_=5e-9, _temp_=200,  _epoch_=15, _rewind_=1 | _lr_=3e-4, l<sup>2</sup>=1e-6, _lambda_=2e-9, _temp_=5000, _epoch_=5, _rewind_=3 | _lr_=3e-5, l<sup>2</sup>=1e-5, _lambda_=2e-9, _temp_=1000, _epoch_=10, _rewind_=0 |
| DCN           | _lr_=3e-4, l<sup>2</sup>=3e-6, _lambda_=1e-8, _temp_=10000, _epoch_=5, _rewind_=2 | _lr_=3e-4, l<sup>2</sup>=1e-6, _lambda_=2e-9, _temp_=5000, _epoch_=5, _rewind_=2 | _lr_=3e-5, l<sup>2</sup>=1e-5, _lambda_=5e-9, _temp_=1000, _epoch_=5,  _rewind_=0 |
| IPNN          | _lr_=3e-4, l<sup>2</sup>=3e-6, _lambda_=5e-9, _temp_=2000, _epoch_=10, _rewind_=1 | _lr_=3e-4, l<sup>2</sup>=1e-6, _lambda_=2e-9, _temp_=5000, _epoch_=5, _rewind_=2 | _lr_=3e-5, l<sup>2</sup>=1e-5, _lambda_=2e-9, _temp_=1000, _epoch_=10, _rewind_=0 |

The following procedure describes how we determine these hyper-parameters:

First, we determine the hyper-parameters of the basic models by grid search: learning ratio and l<sub>2</sub> regularization. We select the optimal learning ratio _lr_ from \{1e-3, 3e-4, 1e-4, 3e-5, 1e-5\} and l<sub>2</sub> regularization from \{3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6\}. Adam optimizer and Xavier initialization are adopted. We empirically set the batch size to be 2048, embedding dimension to be 16, MLP structure to be [1024, 512, 256].

Second, we tune the hyper-parameters introduced by the OptFS method. We select the regularization lambda _lambda_ from \{1e-8, 5e-9, 2e-9, 1e-9\}, final temperature _temp_ from \{10000, 5000, 2000, 1000, 500, 200, 100\}, maximum epoch _epoch_ from \{5, 10, 15, 20\} and rewind epoch _rewind_ from \{0, 1, ..., _epoch_\}. During tuning process, we fix the optimal learning ratio _lr_ and l<sub>2</sub> regularization determined in the first step.
