# Python Code Generation using Character-level RNN

This project is a Python-based tool for generating Python code using a character-level Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers. It allows users to generate Python code snippets based on user input, train new models, continue training existing models, and evaluate the model's performance.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage) 
- [Menu Options](#menu-options)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Model Updating](#model-updating)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python 3.7+**: The script is written in Python and requires Python 3.7 or higher.
- **TensorFlow**: Install TensorFlow 2.x for deep learning.
- **spaCy**: Install spaCy and download the English language model (`en_core_web_sm`).
- **GPU (optional)**: If you have a GPU, TensorFlow can leverage it for faster training.
- **GPU Driver**: You will need to install nvidia gpr drivers and cuda drivers and cuda toolkit
  (please note you will need a cuda capable GPU(s))

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Shadowdrums/RNN-Duo
   cd your-repo
Install Dependencies:

Install the required Python packages using pip:

Copy code
```bash
pip install tensorflow spacy
python -m spacy download en_core_web_sm
```

to install the python libraries with requirements.txt
Copy code
```bash
pip install -r requirements.txt
```
## Usage
Menu Options
Generate Python Code: Input a task or functionality, and the model will generate a Python code snippet based on that input. You can then provide feedback to update the model.

Train a New Model: Use this option to train a new character-level RNN model from scratch using Python code files in the current directory.

Continue Training the Existing Model: Continue training an existing model using the dataset.

Evaluate the Model: Evaluate the performance of the current model on the dataset.

Exit: Exit the program.

## Steps to Run the Program
Run the main script:

Copy code
```bash
python main.py
```
Select an option from the menu.

Follow the on-screen prompts to interact with the tool.

## Model Training
When training a new model, the program:

- Reads and preprocesses Python code from .py files in the specified directory.
- Converts the code into numerical sequences using character-level encoding.
- Builds an LSTM-based RNN model.
- Trains the model and saves it for future use.
- Multiple GPU Support
If multiple GPUs are available, the program can leverage them for data parallelism to speed up training.

### Model Evaluation
You can evaluate the model's performance using the "Evaluate the Model" option, which outputs the loss and accuracy metrics.

### Model Updating
After generating code, the program requests user feedback. If positive feedback is received, the model is updated using the current training data.

## File Structure
- main.py: The main script that runs the program.
- requirements.txt
- trained_model.keras: The saved model file (generated after training).
- README.md: This readme file.
## Contributing
  Contributions are welcome! Please fork this repository and submit a pull request.

# MIT License

Copyright (c) [2024] [Shadowdrums]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

