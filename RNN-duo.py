import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import spacy

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    # Allow memory growth to avoid allocating all GPU memory at once
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU available and configured.")
    
# Function to read and preprocess Python code from all .py files in a directory and its subdirectories
def read_and_preprocess_python_code(directory):
    code_snippets = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                        # Preprocess the code (you may need to customize this based on your requirements)
                        code_snippets.append(code)
                except UnicodeDecodeError as e:
                    print(f"Error reading file {file_path}: {e}")
    return code_snippets

# Function to build and train a character-level RNN model
def build_and_train_model(input_sequences, target_sequences, unique_chars, epochs, model_path):
    strategy = tf.distribute.MirroredStrategy()
    
    with strategy.scope():
        model = keras.Sequential([
            layers.Embedding(input_dim=len(unique_chars), output_dim=128),
            layers.LSTM(256, return_sequences=True),
            layers.LSTM(256),
            layers.Dense(256, activation='relu'),
            layers.Dense(len(unique_chars), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Define callbacks
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            model_path, 
            monitor='val_loss', 
            save_best_only=True, 
            save_weights_only=False,  # Save the entire model
            verbose=1  # Print messages when saving the model
        )

        # Train the model with ModelCheckpoint callback
        history = model.fit(
            input_sequences, 
            target_sequences, 
            epochs=epochs, 
            batch_size=128, 
            callbacks=[checkpoint_callback], 
            validation_split=0.2
        )

    return model, history

# Function to preprocess and extract meaningful information using spaCy
def preprocess_and_extract_info(user_input):
    doc = nlp(user_input)
    # Extract relevant information from the parsed spaCy doc
    # Add more specific processing based on your needs
    return doc

# Function to generate Python code based on the trained model and user input
def generate_code(user_input, model, char_to_index, index_to_char, num_chars=300):
    # Extract meaningful information from user input
    user_info = preprocess_and_extract_info(user_input)

    # Modify code generation based on user_info
    # Example: Adjust seed_text based on the extracted information
    seed_text = f"create a Python script to {user_input.lower()}"
    generated_code = seed_text

    for _ in range(num_chars):
        # Ensure that the sequence has the correct size
        sequence = [char_to_index[char] for char in seed_text]
        if len(sequence) != 100:
            # Pad the sequence if it's shorter than the expected size
            sequence = [0] * (100 - len(sequence)) + sequence

        sequence = np.array(sequence).reshape((1, 100))
        predicted_index = np.argmax(model.predict(sequence), axis=-1)
        predicted_char = index_to_char[predicted_index[0]]
        seed_text += predicted_char
        seed_text = seed_text[1:]
        generated_code += predicted_char

    return generated_code

# Function to get user feedback on generated code
def get_user_feedback(generated_code):
    print("\nGenerated Python code:")
    print(generated_code)
    feedback = input("\nPlease provide feedback (positive/negative/neutral): ")
    return feedback.lower()

# Function to update the model based on user feedback
def update_model(model, input_sequences, target_sequences, feedback, epochs=1, batch_size=128):
    if feedback == 'positive':
        print("Updating model based on positive feedback...")
        # Assuming you have labeled feedback data, use it to further train the model
        model.fit(input_sequences, target_sequences, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        print("Model updated successfully.")
    elif feedback == 'negative':
        print("Negative feedback received. The model will not be updated.")
    else:
        print("Neutral feedback received. No model update.")

# Define the directory containing Python code files
code_directory = os.getcwd()  # Use the current working directory

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Read and preprocess Python code
code_snippets = read_and_preprocess_python_code(code_directory)

# Combine all code snippets into a single string
corpus = " ".join(code_snippets)

# Generate unique character indices
unique_chars = sorted(set(corpus))
char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
index_to_char = {idx: char for idx, char in enumerate(unique_chars)}

# Convert the corpus into numerical sequences
sequences = [char_to_index[char] for char in corpus]

# Prepare input and target sequences
input_sequences = [sequences[i:i+100] for i in range(0, len(sequences)-100, 1)]
target_sequences = [sequences[i+100] for i in range(0, len(sequences)-100, 1)]

# Convert to numpy arrays
input_sequences = np.array(input_sequences)
target_sequences = np.array(target_sequences)

# Check if a trained model exists
model_path = 'trained_model.keras'  # Changed file extension to .keras
if os.path.exists(model_path):
    # Load the model
    model = keras.models.load_model(model_path)
    print("Found a trained model.")
else:
    print("No trained model found. Training a new model...")
    # Ask the user for the number of epochs
    epochs = int(input("Enter the number of training epochs: "))
    # Build and train the model
    model, history = build_and_train_model(input_sequences, target_sequences, unique_chars, epochs, model_path)
    print(f"Trained model saved with {epochs} epochs.")

# Menu to choose between options
while True:
    print("\nMenu:")
    print("1. Generate Python code using the trained model and user input")
    print("2. Train a new model")
    print("3. Continue training the existing model")
    print("4. Evaluate the model")
    print("5. Exit")

    choice = input("Enter your choice (1/2/3/4/5): ")

    if choice == '1':
        # Generate Python code using the trained model and user input
        user_prompt = input("Enter a specific task or functionality you want in the Python code: ")
        generated_code = generate_code(user_prompt, model, char_to_index, index_to_char, num_chars=300)

        # Get user feedback
        feedback = get_user_feedback(generated_code)

        # Update the model based on user feedback
        update_model(model, input_sequences, target_sequences, feedback)

    elif choice == '2':
        # Train a new model
        print("Training a new model...")
        # Read and preprocess Python code
        code_snippets = read_and_preprocess_python_code(code_directory)

        # Combine all code snippets into a single string
        corpus = " ".join(code_snippets)

        # Generate unique character indices
        unique_chars = sorted(set(corpus))
        char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
        index_to_char = {idx: char for idx, char in enumerate(unique_chars)}

        # Convert the corpus into numerical sequences
        sequences = [char_to_index[char] for char in corpus]

        # Prepare input and target sequences
        input_sequences = [sequences[i:i+100] for i in range(0, len(sequences)-100, 1)]
        target_sequences = [sequences[i+100] for i in range(0, len(sequences)-100, 1)]

        # Convert to numpy arrays
        input_sequences = np.array(input_sequences)
        target_sequences = np.array(target_sequences)

        # Ask the user for the number of epochs
        epochs = int(input("Enter the number of training epochs: "))

        # Build and train the model
        model, history = build_and_train_model(input_sequences, target_sequences, unique_chars, epochs, model_path)

        print(f"Trained model saved with {epochs} epochs.")

    elif choice == '3':
        # Continue training the existing model
        if not os.path.exists(model_path):
            print("No existing model to continue training.")  
        else:
            # Ask the user for the number of additional epochs
            additional_epochs = int(input("Enter the number of additional training epochs: "))
            # Load the existing model
            model = keras.models.load_model(model_path)

            # Check if multiple GPUs are available
            if len(physical_devices) > 1:
                # Use data parallelism for multi-GPU training
                strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
                with strategy.scope():
                    # Re-create the model and compile within the strategy scope
                    model = keras.Sequential([
                        layers.Embedding(input_dim=len(unique_chars), output_dim=128),
                        layers.LSTM(256, return_sequences=True),
                        layers.LSTM(256),
                        layers.Dense(256, activation='relu'),
                        layers.Dense(len(unique_chars), activation='softmax')
                    ])
                    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                # Continue training the model
                model.fit(input_sequences, target_sequences, epochs=additional_epochs, batch_size=128, validation_split=0.2)

                # Save the updated model
                model.save(model_path)
                print(f"Continued training and saved the model with an additional {additional_epochs} epochs.")
            else:
                # Continue training without multiple GPUs
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                model.fit(input_sequences, target_sequences, epochs=additional_epochs, batch_size=128, validation_split=0.2)
                model.save(model_path)
                print(f"Continued training and saved the model with an additional {additional_epochs} epochs.")

    elif choice == '4':
        # Evaluate the model
        evaluation_loss, evaluation_accuracy = model.evaluate(input_sequences, target_sequences, verbose=0)
        print(f"Model Evaluation - Loss: {evaluation_loss}, Accuracy: {evaluation_accuracy}")

    elif choice == '5':
        print("Exiting the program.")
        break

    else:
        print("Invalid choice. Please enter a valid option.")

