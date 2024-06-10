import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_word_cloud(dataframe, column_name, save_path=None):
    long_string = ' '.join(list(dataframe[column_name]))
    wordcloud = WordCloud(background_color="black", max_words=5000, contour_width=3, contour_color='steelblue')
    wordcloud.generate(long_string)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()
