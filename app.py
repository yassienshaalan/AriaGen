import os
import sounddevice as sd
import numpy as np
import logging
import time
from datetime import datetime
import soundfile as sf
import tensorflow as tf

# Ensure directories exist
os.makedirs('logs', exist_ok=True)
os.makedirs('inputs', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Set up logging to file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('logs/recording.log'),
                              logging.StreamHandler()])

class AriaGen:
    def __init__(self, sample_rate=44100, channels=1, duration=15):
        """
        Initialize the music generator with specified audio settings.
        
        :param sample_rate: Sampling rate for audio recording in Hz.
        :param channels: Number of audio channels.
        :param duration: Duration to record in seconds.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration = duration
        logging.info("AriaGen initialized with sample_rate=%s, channels=%s, duration=%s",
                     sample_rate, channels, duration)

    def record_audio(self):
        """
        Record audio from the microphone while displaying a countdown timer.
        
        :return: NumPy array with the recorded audio data.
        """
        logging.info("Starting audio recording for %s seconds...", self.duration)
        recording = np.zeros(int(self.duration * self.sample_rate) * self.channels, dtype='float32')
        with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, dtype='float32') as stream:
            for i in range(self.duration, 0, -1):
                print(f"Recording in {i} seconds...")
                time.sleep(1)
                data, _ = stream.read(self.sample_rate)
                recording[(self.duration - i) * self.sample_rate:(self.duration - i + 1) * self.sample_rate] = data.flatten()
        logging.info("Recording stopped.")
        return recording
    '''
    def generate_music(self, audio_data, music_style):
        """
        Generate music based on the recorded audio and specified music style.
        This method should be implemented based on the specific AI model used.
        
        :param audio_data: The recorded audio data as a NumPy array.
        :param music_style: The style of music to generate (e.g., 'jazz', 'pop').
        :return: Dummy data; replace with actual model output.
        """
        logging.info("Generating music in the style: %s", music_style)
        return np.random.rand(self.sample_rate * self.duration)  # Dummy data
    '''
    
    def generate_music(self, audio_data, music_style):
        # Load the pre-trained generator
        generator = tf.keras.models.load_model('gan_generator.h5')

        # Generate music using the generator
        random_latent_vectors = np.random.normal(0, 1, (1, LATENT_DIM))
        generated_music = generator.predict(random_latent_vectors)[0]

        return generated_music

    def save_music(self, music_data, music_style):
        """
        Save the generated music data to a WAV file in the outputs directory with a timestamped filename.
        
        :param music_data: The music data to save.
        :param music_style: The style of music, used in the filename.
        """
        filename = f'outputs/{music_style}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.wav'
        logging.info("Saving music to %s", filename)
        sf.write(filename, music_data, self.sample_rate)
        logging.info("Music saved successfully.")

if __name__ == "__main__":
    music_style = input("Enter the music style you want (e.g., jazz, pop, classical): ")
    mg = AriaGen(duration=5)  # Record for 5 seconds
    audio_data = mg.record_audio()
    # Optionally save the input recording
    input_filename = f'inputs/{music_style}_input_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.wav'
    sf.write(input_filename, audio_data, mg.sample_rate)
    music_data = mg.generate_music(audio_data, music_style)
    mg.save_music(music_data, music_style)
    logging.info("Process completed.")