# import nltk
# nltk.download('gutenberg')

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from text import text

wordcloud = WordCloud(background_color="white").generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
