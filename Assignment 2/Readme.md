# ANLP Assignment - 2

Transformer built from scratch for Machine Translation (English to French)

## File Structure:
- **utils.py**: Contains data preprocessing functions and utility methods.
- **encoder.py**: Defines the Encoder class and its helper functions.
- **decoder.py**: Defines the Decoder class and its helper functions.
- **transformer.py**: Implements the Transformer model and includes functions to train it.
  
### Training the Model:
- **train.py**: This script sets the parameters and calls necessary functions to train the Transformer model. Once training is complete, the model is saved as `transformer.pt`, along with the vocabularies.
  
  To train the model, run:
  ```bash
  python3 train.py
  ```

  After execution, the model and vocabularies will be saved.

### Hyperparameter Tuning:
- **tuning.py**: This script performs hyperparameter tuning and saves the best-performing model. It also outputs the best hyperparameters found during the tuning process.
  
  To tune the model, run:
  ```bash
  python3 tuning.py
  ```

### Testing the Model:
- **test.py**: This script loads the saved model and vocabularies to make predictions on the test dataset. It calculates BLEU scores and writes the predicted sentences along with their corresponding BLEU scores to a file named `testbleu.txt`.
  
  To test the model or make predictions on the test set, run:
  ```bash
  python3 test.py
  ```

### Report:
- **Report.pdf**: Contains answers to theory questions and inferences made.

### Link to Saved Models and Vocabularies:
- [Download the saved vocabularies and model](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/kukkapalli_shravya_students_iiit_ac_in/EgpE5CrJI1hBszf_Uty1190BVRgVYAZEaJMFCftKbKCdqQ?e=6nLz0S)