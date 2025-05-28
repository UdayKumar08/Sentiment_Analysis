""" 
Importing required functions from custom modules
"""
from Data_Loader import load_and_prepare
from model_trainer import train_model, prepare_tf_dataset, encode_labels, save_model


""" 
Load and preprocess data 
This function returns tokenized input encodings and corresponding labels 
for both the training and testing datasets 
"""
print("Loading and preprocessing data...")
train_encodings, train_labels, test_encodings, test_labels = load_and_prepare()


""" 
Encode labels
LabelEncoder is used to convert string labels into numeric format 
so that they can be used by the model during training 
"""
train_labels, label_encoder = encode_labels(train_labels)
test_labels, _ = encode_labels(test_labels)


""" 
Prepare TensorFlow datasets 
Converts tokenized inputs and encoded labels into TensorFlow tf.data.Dataset objects 
These datasets will be fed into the training loop 
"""
train_dataset = prepare_tf_dataset(train_encodings, train_labels)
test_dataset = prepare_tf_dataset(test_encodings, test_labels)


""" 
Train the model 
This will train a transformer-based model using the prepared train and test datasets 
"""
print("Training the model...")
model = train_model(train_dataset, test_dataset)


""" 
Save the trained model 
The trained model will be saved locally for later inference or deployment 
"""
print("Saving model...")
save_model(model)

""" 
Final message indicating that training is complete and the model has been saved 
"""
print("Training complete. Model saved to ./saved_model")
