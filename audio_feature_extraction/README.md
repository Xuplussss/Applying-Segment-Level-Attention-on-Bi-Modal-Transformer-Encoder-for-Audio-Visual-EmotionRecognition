# Fine-tuning Wav2vec 2.0 for SER

!["our audio feature extraction frameworks"](https://github.com/Xuplussss/Applying-Segment-Level-Attention-on-Bi-Modal-Transformer-Encoder-for-Audio-Visual-EmotionRecognition/blob/main/audio_feature_extraction/wav2vec2.PNG?raw=true)

Official implementation for the paper [Exploring Wav2vec 2.0 fine-tuning for improved speech emotion recognition](http://arxiv.org/abs/2110.06309).
Submitted to ICASSP 2022.

## Libraries and dependencies
 - [pytorch](https://github.com/pytorch/pytorch)
 - [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)
 - [fairseq](https://github.com/pytorch/fairseq) (For Wav2vec)
 - [huggingface transformers](https://huggingface.co) (For Wav2vec2)
 - [faiss](https://github.com/facebookresearch/faiss) (For running clustering)

Faiss can be skipped if you are not running clustering scripts.
Or you can simply check the DockerFile at `docker/Dockerfile` for our setup.
To train the first phase wav2vec model of P-TAPT, you'll need the the pretrained wav2vec model checkpoint from Facebook AI Research, which can be obtained [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt).

 - Code just switched from hard-coding to passing arguments, not sure every scripts are working as expected.

### Prepare IEMOCAP
Obtain [IEMOCAP](https://sail.usc.edu/iemocap/) from USC
```
cd Dataset/IEMOCAP &&
python make_16k.py IEMOCAP_DIR &&
python gen_meta_label.py IEMOCAP_DIR &&
python generate_labels_sessionwise.py &&
cd ../..
```

### Run scripts
 - V-FT: `bash bin/run_exp_iemocap_vft.sh Dataset/IEMOCAP/Audio_16k/ Dataset/IEMOCAP/labels_sess/label_{SESSION_TO_TEST}.json OUTPUT_DIR GPU_ID V-FT NUM_EXPS`

The OUTPUT_DIR should not be exist and different for each method, note that it will take a long time since we need to run NUM_EXPS and average. The statistics will be at `OUTPUT_DIR/{METHOD}.log` along with some model checkpoints. Note that it takes a long time and lots of VRAM, if you are concerned at computation, try lower the batch size (but the results may not be as expected).

## Run the training scripts on your own dataset
You will need a directory containing all the training wave files sampled at 16kHz, and a json file which contains the emotion label, and the *training/validation/testing* splits in the following format:
```
{
    "Train": {
        audio_filename1: angry,
        audio_filename2: sad,
        ...
    }
    "Val": {
        audio_filename1: neutral,
        audio_filename2: happy,
        ...
    }
    "Test": {
        audio_filename1: neutral,
        audio_filename2: angry,
        ...
    }
}
```
 - If the Test has zero elements `"Test: {}"`, no testing will be performed, same rule holds for validation.
 - Put all your dataset in the following structure, we will be mounting this directory to the container.

## V-FT
```
python run_downstream_custom_multiple_fold.py --precision 16 \
                                              --num_exps NUM_EXP \
                                              --datadir Audio_Dir \
                                              --labeldir LABEL_DIR \
                                              --saving_path SAVING_CKPT_PATH \
                                              --outputfile OUTPUT_FILE
```
 - `--max_epochs`: The epoch to train on the custom dataset, default to 15
 - `--maxseqlen`: maximum input duration in sec, truncate if exceed, default to 12
 - `--labeldir`: A directory contains all the label files to be evaluate (in folds)
 - `--saving_path` Path for audio generated checkpoints
 - `--outputfile`: A log file for outputing the test statistics
