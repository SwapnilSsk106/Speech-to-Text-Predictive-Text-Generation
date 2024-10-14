import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class PredictiveText:
    def __init__(self, text):
        self.model = self._build_model(text)

    def _build_model(self, text):
        model = defaultdict(Counter)
        words = re.findall(r'\b\w+\b', text.lower())

        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            model[current_word][next_word] += 1

        return model

    def predict_next_word(self, input_text):
        words = re.findall(r'\b\w+\b', input_text.lower())
        if not words:
            return []

        last_word = words[-1]
        if last_word in self.model:
            next_words = self.model[last_word]
            return next_words.most_common()
        else:
            return []

    def get_word_frequencies(self):
        frequencies = defaultdict(Counter)
        for current_word, next_words in self.model.items():
            frequencies[current_word] = next_words
        return frequencies

def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def main():
    file_path = 'corpus.txt'
    corpus = load_corpus(file_path)

    predictive_text = PredictiveText(corpus)

    print("Predictive Text System")
    print("======================")

    word_frequencies = defaultdict(Counter)

    while True:
        user_input = input("\nEnter some text (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        predictions = predictive_text.predict_next_word(user_input)
        if predictions:
            print("Suggested next words:")
            for word, freq in predictions:
                print(f"- {word} (Frequency: {freq})")
                word_frequencies[user_input].update({word: freq})
        else:
            print("No predictions available.")

    df_list = []
    for input_text, freqs in word_frequencies.items():
        for word, freq in freqs.items():
            df_list.append({'input_text': input_text, 'next_word': word, 'frequency': freq})

    df = pd.DataFrame(df_list)

    if not df.empty:
        top_words_df = df.groupby('next_word')['frequency'].sum().reset_index()
        top_words_df = top_words_df.sort_values(by='frequency', ascending=False).head(10)
        top_words = top_words_df['next_word'].tolist()

        df = df[df['next_word'].isin(top_words)]

        plt.figure(figsize=(14, 8))
        sns.set(style="whitegrid")
        sns.barplot(data=df, x='input_text', y='frequency', hue='next_word', palette='viridis')

        plt.title('Frequency of Top 10 Predicted Next Words for Various Inputs')
        plt.xlabel('Input Text')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Next Word')
        plt.tight_layout()
        plt.show()

        print("\nTable of Top 10 Words by Frequency:")
        print(top_words_df)

    else:
        print("No data to plot or display.")

if __name__ == "__main__":
    main()