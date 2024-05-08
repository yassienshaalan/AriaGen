# AriaGen: Audio to Music AI Transformation

Welcome to AriaGen, an advanced audio-to-music transformation application that turns your simple hums into stylized musical compositions. Built with Python, AriaGen records your humming live through the microphone, applies AI-driven techniques to generate music in your chosen style, and outputs the creation as a high-quality WAV file. Supported music styles include jazz, pop, classical, and more, allowing for a personalized musical experience.

## Features

- **Custom Audio Input**: Record your humming or any simple melody directly through your system's microphone as input.
- **Diverse Music Styles**: Choose from a variety of music styles to transform your audio input into, such as jazz, pop, and classical.
- **AI-Powered Music Generation**: Leverage a pre-trained Generative Adversarial Network (GAN) to generate music based on the recorded audio.
- **Automated File Management**: Automatically save both the original recording and the AI-generated music with timestamped filenames for easy tracking.

## Features

- **Audio Recording**: Record live audio directly through your system's microphone.
- **Music Generation**: Transform recorded audio into music using a pre-trained Generative Adversarial Network (GAN).
- **Multiple Music Styles**: Generate music in styles such as jazz, pop, and classical.
- **Automatic Saving**: Save both the original recording and generated music with timestamped filenames for organization.

## Installation

To set up AriaGen on your local machine, follow these steps:

1. Clone the repository:
```
git clone https://github.com/yourusername/AriaGen.git
```
2. Navigate to the AriaGen directory.
3. Install the required Python libraries and run app:
```
pip install -r requirments.txt
```
### Running the Application

After installation, you can run the application in either generation mode or training mode:

- **To generate music**:
  ```
  python app.py --mode generate --style your_music_style
  ```
  Replace `your_music_style` with your desired style, such as 'jazz', 'pop', or 'classical'.

- **To train a new model**:
  ```
  python app.py --mode train
  ```
## Retraining the GAN Model

AriaGen uses a pre-trained Generative Adversarial Network (GAN) to transform recorded audio into music. If you wish to further improve or personalize the music generation capabilities, you may consider retraining the GAN model with your own dataset.

### Requirements for Retraining

- A dataset of audio files and their corresponding musical style labels.
- TensorFlow 2.x and additional libraries as needed (e.g., librosa for audio processing).
- Adequate computational resources (GPU recommended for training).

## Retraining the GAN Model

AriaGen uses a pre-trained Generative Adversarial Network (GAN) to transform recorded audio into music. If you wish to further improve or personalize the music generation capabilities, you may consider retraining the GAN model with your own dataset.

### Requirements for Retraining

- A dataset of audio files and their corresponding musical style labels.
- TensorFlow 2.x and additional libraries as needed (e.g., librosa for audio processing).
- Adequate computational resources (GPU recommended for training).

### Steps to Retrain the Model

1. **Prepare Your Dataset**:
 - Collect a diverse set of audio recordings and categorize them by musical style.
 - Preprocess the data into a suitable format for training. This typically involves converting audio files into spectrograms or Mel-frequency cepstral coefficients (MFCCs).

2. **Modify Training Scripts**:
 - Adjust the existing training scripts to accommodate your dataset and training parameters.
 - You can find the training scripts in the `train` directory within the repository.

3. **Run the Training**:
 - Execute the training script with the prepared data:
   ```
   python train/train_model.py --dataset path/to/your/dataset
   ```
 - Monitor the training process and adjust parameters as necessary to improve model performance.

4. **Evaluate the Model**:
 - After training, evaluate the new model using a separate validation set to ensure it generates music accurately reflecting the intended styles.

5. **Integrate the New Model**:
 - Once retraining is complete and the model performs satisfactorily, replace the existing model file in `gan_generator.h5` with the new model file.
 - Test the integration to ensure that the application correctly utilizes the new model.

### Tips for Successful Retraining

- Focus on a high-quality, varied dataset to train the model.
- Regularly save model checkpoints to avoid losing progress.
- Experiment with different architectures and hyperparameters to find the best setup for your specific use case.

By following these steps, you can enhance AriaGen's ability to generate music that better matches your preferences or improves upon the pre-trained model's capabilities.


## Follow the prompts in the terminal to choose the duration of the recording and the style of music you want to generate.

## Dependencies

- Python 3.x
- NumPy
- SoundDevice
- SoundFile
- TensorFlow

## Using Docker

To simplify the installation and execution environment for AriaGen, you can use Docker. Follow these steps to build and run AriaGen using a Docker container.

### Build the Docker Image

First, build the Docker image from your project directory where the Dockerfile resides:

```
docker build -t ariagen .
```
Run the Container
Once the image is built, you can run it:
```
docker run -p 4000:80 ariagen
```


## Contributing

Contributions are welcome! Please fork the repository and submit pull requests with your proposed changes.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Author

- Yassien Shaalan https://github.com/yassienshaalan

## Acknowledgements

- Thank you to the open-source community for the various tools and libraries that make projects like this possible.
- Special thanks to TensorFlow team for providing the deep learning framework used in this project.

