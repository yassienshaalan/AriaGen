import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import soundfile as sf
import IPython.display as ipd

class MusicGAN:
    def __init__(self, directory='input_data/fma_small', duration=5, sr=22050, latent_dim=100, epochs=10000, batch_size=16):
        self.directory = directory
        self.duration = duration
        self.sr = sr
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.compile_gan()

    def load_data(self):
        """
        Load and preprocess audio data from the specified directory.

        Returns:
        np.array: Preprocessed audio data as spectrograms.
        """
        files = glob.glob(os.path.join(self.directory, '*.mp3'))[:1000]  # Limit to 1000 files for demonstration
        data = []
        for file in files:
            audio, _ = librosa.load(file, sr=self.sr, duration=self.duration)
            spectrogram = librosa.stft(audio)
            spectrogram = np.abs(spectrogram)
            spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
            data.append(spectrogram)
        data = np.array(data)
        print("Data Length ",data.shape)
        return data

    def build_generator(self):
        model = tf.keras.Sequential([
            layers.Dense(512, activation='relu', input_dim=self.latent_dim),
            layers.BatchNormalization(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(2048, activation='sigmoid')
        ])
        return model

    def build_discriminator(self, data_shape=2048):
        model = tf.keras.Sequential([
            layers.Dense(1024, activation='relu', input_shape=(data_shape,)),
            layers.Dense(512, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def compile_gan(self):
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam')
        self.discriminator.trainable = False
        gan_input = layers.Input(shape=(self.latent_dim,))
        gan_output = self.discriminator(self.generator(gan_input))
        gan = models.Model(gan_input, gan_output)
        gan.compile(loss='binary_crossentropy', optimizer='adam')
        return gan

    def train(self, data):
        if len(data)==0:
            print("Exiting no data to train the model")
            return False
        for epoch in range(self.epochs):
            # Sample random points in the latent space
            random_latent_vectors = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            # Generate fake audio data
            generated_data = self.generator.predict(random_latent_vectors)
            # Mix them with real data
            real_data = data[np.random.randint(0, data.shape[0], self.batch_size)]
            combined_data = np.vstack((generated_data, real_data))
            # Assemble labels discriminating real from fake data
            labels = np.concatenate([np.zeros(self.batch_size), np.ones(self.batch_size)])
            # Train discriminator
            d_loss = self.discriminator.train_on_batch(combined_data, labels)
            # Train generator
            misleading_targets = np.ones(self.batch_size)
            g_loss = self.gan.train_on_batch(random_latent_vectors, misleading_targets)
            # Logging progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Discriminator loss: {d_loss}, Generator loss: {g_loss}")
        return True

    def generate_music(self):
        """
        Generate music using the trained generator.

        Returns:
        np.array: Generated spectrogram.
        """
        random_latent_vectors = np.random.normal(0, 1, (1, self.latent_dim))
        generated_spectrogram = self.generator.predict(random_latent_vectors)[0]
        return generated_spectrogram
    
    def save_models(self):
        """Saves the generator and discriminator models to files."""
        self.generator.save('gan_generator.h5')
        self.discriminator.save('gan_discriminator.h5')
        print("Models saved successfully.")


    def load_generator(self, generator_path='gan_generator.h5'):
        """Loads the generator model from a file."""
        self.generator = tf.keras.models.load_model(generator_path)
        print("Generator model loaded successfully.")

    def load_discriminator(self, discriminator_path='gan_discriminator.h5'):
        """Loads the discriminator model from a file."""
        self.discriminator = tf.keras.models.load_model(discriminator_path)
        print("Discriminator model loaded successfully.")

    def spectrogram_to_audio(self, spectrogram, save_path=None, play_audio=False):
        """
        Convert a spectrogram back to an audio waveform and optionally save or play it.

        Parameters:
        spectrogram (np.array): The generated spectrogram.
        save_path (str): Path to save the audio file. If None, the audio is not saved.
        play_audio (bool): Whether to play the audio using IPython display.

        Returns:
        np.array: The time-domain audio signal.
        """
        # Assuming the spectrogram is in dB, convert it back to amplitude
        spectrogram = librosa.db_to_amplitude(spectrogram)
        
        # Inverse STFT to convert back to time domain audio signal
        audio = librosa.istft(spectrogram)

        # Normalize audio to prevent potential clipping
        audio = np.clip(audio, -1, 1)

        # Save audio if a path is provided
        if save_path:
            sf.write(save_path, audio, self.sr, format='wav')
            print(f"Audio saved to {save_path}")

        # Play audio if requested
        if play_audio:
            ipd.display(ipd.Audio(audio, rate=self.sr))

        return audio


if __name__ == '__main__':

    gan = MusicGAN()
    data = gan.load_data()
    result = gan.train(data)
    if result == True:
        generated_spectrogram = gan.generate_music()
        # Convert generated spectrogram to audio, save, and play
        audio_signal = gan.spectrogram_to_audio(generated_spectrogram, save_path='generated_music.wav', play_audio=True)
        gan.save_models()

        # To generate music using a pre-trained generator
        gan.load_generator()
        generated_spectrogram = gan.generate_music()
        # Convert generated spectrogram to audio, save, and play
        audio_signal = gan.spectrogram_to_audio(generated_spectrogram, save_path='generated_music.wav', play_audio=True)
    print("Exiting")
