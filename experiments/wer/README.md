# Fine-tuning a letter classifier and estimating WER with a Wav2Letter decoder

### Requirements
* wav2letter python bindings: [(how-to)](https://github.com/facebookresearch/wav2letter/tree/master/bindings/python)
* KenLM-based Librispeech language model, can be found [here](http://www.openslr.org/11/)
* jiwer, installable via `pip install jiwer` (TODO: remove the dependency)

The `4-gram.bin` and `lexicon.txt` files are expected to be placed in `./data/` folder, alongside with `letters.lst`.

### Running examples
Training a letter classifier on top of a pre-trained CPC model:
```
python letter_ctc_train.py --path_train=<path to fine-tuning dataset> --path_val=<validation dataset> --path_checkpoint=<path to a pre-trained CPC model> --lr=1e-3  --n_epochs=50 --p_dropout=0.1 --output=<path where the classifier would be saved>
```
Evaluating it with wav2letter decoder:
```
python letter_ctc_train.py --path_wer=<path to evaluation data, e.g. librispeech test-clean> --path_wer=<optional path to other evaluation data, e.g. librispeech dev-other>  --output=<path to the saved classifier> --batch_size=32
```

You can also train and evaluate afterwards, in a single command:
```
python letter_ctc_train.py --path_train=<path to fine-tuning dataset> --path_val=<validation dataset> --path_checkpoint=<path to a pre-trained CPC model> --lr=1e-3  --n_epochs=50 --p_dropout=0.1 --output=<path where the classifier would be saved> --path_wer=<path to evaluation data, e.g. librispeech test-clean
```
