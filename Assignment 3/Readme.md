# ANLP Assignment - 3

Evaluating different finetuning methods for summarization

## PROMPT TUNING:
- **prompt_utils.py**: Contains data preparation, model class, training and testing functions.
- **prompt_tuning.py**: Sets hyperparameters and uses functions defined in promp_utils.py to finetune the GPT2 model using prompt tuning. 

- To fine-tune the model using prompt tuning, run
```
python3 prompt_tuning.py
```
- This script saves the trained model and its loss curve and also prints the total GPU memory used, trainable parametrs and the training time.

## Traditional finetuning:
- **FT_utils.py**: Contains data preparation, training and testing functions.
- **ft.py**: Sets hyperparameters and uses functions defined in promp_utils.py to finetune the GPT2 model using traditional fine-tuning. 

- To fine-tune the model using traditional fine-tuning, run
```
python3 ft.py
```
- This script saves the trained model and its loss curve and also prints the total GPU memory used, trainable parametrs and the training time.

## LORA:
- **lora_utils.py**: Contains function to load the peft model with lora configuration. (Uses dataset class, train and test functions present in FT_utils.py)
- **lora.py**: Sets hyperparameters and uses functions defined in lora_utils.py, FT_utils.py to finetune the GPT2 model using LORA. 

- To fine-tune the model using Lora, run
```
python3 lora.py
```
- This script saves the trained model and its loss curve and also prints the total GPU memory used, trainable parametrs and the training time.

## Testing:
-**test.py**: Loads all the saved models and calculates loss and rouge scores.

To test the models, run
```
python3 test.py
```

- This prints the test loss and corresponding rouge scores for the models finetuned using above 3 methods.

### Report:
- **Report.pdf**: Contains answers to theory questions and inferences made.

### Link to Saved Models:
- [Download the saved models](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/kukkapalli_shravya_students_iiit_ac_in/EuyQPmTy1OVLqRcHHCID0VcBdRcng2q9s-eQbG1J6y0QJw?e=BbP8Xn)