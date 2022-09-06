# University Chatbot

This is a chatbot made for universities for handling question answers such as where to find a place, how to get there and also if deployed in the departments how the chatbot can be helpful if a physical body is built.


# Files and Usage

The `intents.json` file contains the intents which was used to train the chatbot. This file can be edited and the model can be retrained accordingly. Right now it only contains a few intents but as more intents are added the chatbot will become more general.

The packages i had in my virtual environment when building the chatbot are listed in the `requirements.txt` file and can be installed using:

    pip install -r requirements.txt

The `nltk_utils.py` file is used for pre-processing the data by applying procedures such as tokenization and lemmatization.

The `model.py` defines the architecture of the chatbot. Right now a Feed forward neural network(FFN) is being used.

The `train.py` file is to be executed when after defining the model architecture it needs to be trained on the data.

And finally the `Chat.py` is to be executed when you want to converse with the chatbot. To test the current chatbot you can directly execute the current file once you have cloned the repository and have the requirements in the `requirements.txt` file. 

# Future Works

   In the future i will be integrating the current chatbot with a Speech-to-text model (mostly will be using online available speech to text models) so that instead of typing the query, we can just speak and the chatbot will reply.
