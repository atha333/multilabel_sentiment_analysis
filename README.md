# Emotion Recognition using BERT and its Variants
***

<p align="center">
  <img src="Emo_arch.PNG?raw=true" alt="Photo" border="5" width=40%/> 
</p>

# Link to the Presentation
https://www.youtube.com/watch?v=AUmesEOhLlw

# Link to the PDF
https://github.iu.edu/rgampa/multilabel-sentiment-analysis/blob/main/Project%20Report-final.pdf


# Dependencies

```angular2html
pip install -r requirements.txt
```
also run the following command:
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
The model was trained on an Nvidia Tesla P4,V100, Google cloud Vertex AI.

***

# Usage


Next, run the main script to do the followings:
 * data loading and preprocessing
 * model creation and training

### Training
```
python scripts/train.py --dataset=twitter --algo=bert

Options:
    -h --help                         show this screen
    --loss-type=<str>                 Which loss to use cross-ent|corr|joint. [default: cross-entropy]
    --max-length=<int>                text length [default: 128]
    --output-dropout=<float>          prob of dropout applied to the output layer [default: 0.1]
    --seed=<int>                      fixed random seed number [default: 42]
    --train-batch-size=<int>          batch size [default: 32]
    --eval-batch-size=<int>           batch size [default: 32]
    --max-epoch=<int>                 max epoch [default: 20]
    --ffn-lr=<float>                  ffn learning rate [default: 0.001]
    --bert-lr=<float>                 bert learning rate [default: 2e-5]
    --dev-path=<str>                  file path of the dev set [default: '']
    --train-path=<str>                file path of the train set [default: '']
    --alpha-loss=<float>              weight used to balance the loss [default: 0.2]
    --dataset=<str>                   Name of the dataset [default: twitter]
    --dataset-folder=<str>            folder that contains the data [default: ./data/]
    --algo=<str>                      type of bert algorithm used [default: bert]
```



Once the above step is done, you can then evaluate on the test set using the trained model:

## Evaluation
```
python scripts/test.py --dataset=twitter --algo=bert 

Options:
    -h --help                         show this screen
    --model-path=<str>                path of the trained model
    --max-length=<int>                text length [default: 128]
    --seed=<int>                      seed [default: 0]
    --test-batch-size=<int>           batch size [default: 32]
    --lang=<str>                      language choice [default: English]
    --test-path=<str>                 file path of the test set [default: ]
    --algo=<str>                      type of bert algorithm used [default: bert]
    --dataset=<str>                   Name of the dataset [default: twitter]
    --dataset-folder=<str>            folder that contains the data [default: ./data/]
```

To run various other combinations of models and datasets please look at run.sh
***

