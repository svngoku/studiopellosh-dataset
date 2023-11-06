import os
import replicate

# Load the REPLICATE_API_TOKEN from the environment
REPLICATE_API_TOKEN = os.environ['REPLICATE_API_TOKEN']

# Import the file `studiopellosh.zip` into the current directory using os
training_file = "https://github.com/svngoku/studiopellosh-dataset/raw/0d37e39ddb9a116f08d068af6ee0e5865ff9b624/studiopellosh.zip"



# Start the training and make the code asynchronous
def start_training_and_monitor(file=training_file, model='svngoku/sdxl-africans'):
    training = replicate.trainings.create(
        version="stability-ai/sdxl:7ca7f0d3a51cd993449541539270971d38a24d9a0d42f073caf25190d41346d7",
        input={
            "input_images": file,
        },
        destination=model
    )

    return training

if __name__ == '__main__':
    start_training_and_monitor()