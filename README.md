# tf-keras-stacked-convolutional-autoenconder

- Implementation of Stacked Convolutional Autoencoder.
- Evaluated the denoising performance using MNIST.
- Used Python3.7 + TensorFlow 2.3 + tf.keras.

## Settings

- I recommend to make virtual environment by venv.
    1. ```python -m venv .venv```

0. Enter to virtual environment ```.venv/Scripts/activate.ps1``` (Windows) or ```.venv/Scripts/activate``` (Linux)
1. ```pip install -r requirements.txt```

## Run

1. ```python train.py --epochs 20 --batch_size 100 --stacked 1 --snr -40``` (defalut options)
    - --epochs or -e: The number of epochs
    - --batch_size or -b: The number of batch size
    - --stacked or -s: 1 or 0 (stacked or not)
    - --snr or -n: The value of SNR for denoising autoencoder. The smaller value is noisy.

1. ```python eval.py --dir_model path_to_hdf5 --snr -40```
    - --dir_model or -d: path to model.
        - hdf5 file will generate in ../dist/train_conditions/models
    - --snr or n: same to train.py
    - SNR value will printed.

## Results

- Denoising performance improved slightly.
  - SNR comparison:
    - stacked: 9.307
    - no_stacked: 9.184

## Others

- If you have any problems, please let me know.
- Please send me the pull request if you find wrong.