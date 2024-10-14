import speech_recognition as sr
import pyttsx3
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from collections import Counter
import numpy as np
import scipy.io.wavfile as wav
from io import BytesIO

r = sr.Recognizer()


def SpeakText(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()


data = []
audio_data_list = []

print("Say 'quit' to stop listening. Press any key to stop.")


def calculate_metrics(audio_data, sample_rate):
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    signal_power = np.mean(np.square(audio_array))
    noise_power = np.var(audio_array)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else float('inf')

    rmse = np.sqrt(np.mean(np.square(audio_array - np.mean(audio_array))))

    duration = len(audio_array) / sample_rate  # duration in seconds
    bitrate = (len(audio_data) * 8) / duration / 1000  # in kbps

    return snr, rmse, bitrate


try:
    while True:
        with sr.Microphone() as source:
            print("Listening...")
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.listen(source)

        print("Recognizing...")
        MyText = r.recognize_google(audio)
        MyText = MyText.lower()

        if MyText:
            print("Did you say: ", MyText)
            SpeakText(MyText)

            if MyText != 'quit':
                timestamp = datetime.now()
                audio_data = audio.get_wav_data()
                sample_rate = audio.sample_rate

                # Audio metrics
                snr, rmse, bitrate = calculate_metrics(audio_data, sample_rate)

                data.append({'timestamp': timestamp, 'text': MyText, 'snr': snr, 'rmse': rmse, 'bitrate': bitrate})
                audio_data_list.append(audio_data)
            else:
                print("Quitting...")
                break

except sr.UnknownValueError:
    print("Google Speech Recognition could not understand the audio")

except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")

except KeyboardInterrupt:
    print("\nInterrupted by user.")

df = pd.DataFrame(data)

df.to_csv('speech_data.csv', index=False)

if not df.empty:
    words = [text for text in df['text']]
    word_counter = Counter(" ".join(words).split())
    top_words = word_counter.most_common(25)
    top_words_dict = dict(top_words)

    plot_data = []
    for word, count in top_words_dict.items():
        occurrences = df[df['text'].str.contains(word, na=False)]
        timestamps = occurrences['timestamp']
        plot_data.append({'word': word, 'timestamps': timestamps, 'count': count})

    plt.figure(figsize=(12, 8))

    for item in plot_data:
        word = item['word']
        timestamps = item['timestamps']
        count = item['count']
        df_word = pd.DataFrame({'timestamp': timestamps})
        df_word['count'] = count
        plt.plot(df_word['timestamp'], df_word['count'], label=word, marker='o')

    plt.xlabel('Timestamp')
    plt.ylabel('Count')
    plt.title('Frequency of Top 15 Recognized Words Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(df['timestamp'], df['snr'], label='SNR', marker='o')
    plt.xlabel('Timestamp')
    plt.ylabel('SNR (dB)')
    plt.title('Signal-to-Noise Ratio Over Time')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(df['timestamp'], df['rmse'], label='RMSE', marker='o', color='orange')
    plt.xlabel('Timestamp')
    plt.ylabel('RMSE')
    plt.title('Root Mean Square Error Over Time')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(df['timestamp'], df['bitrate'], label='Bitrate', marker='o', color='green')
    plt.xlabel('Timestamp')
    plt.ylabel('Bitrate (kbps)')
    plt.title('Bitrate Over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

else:
    print("No data to plot.")