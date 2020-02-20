import abc

import nltk;
import cv2
import numpy as np
import glob
import mysql.connector
from mysql.connector import Error
import os
import io
import image
from array import array
import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pymysql
import io
from PIL import Image
import cv2

"""
#for image display from database
connection=pymysql.connect(host="localhost",
                 user="root",
                 passwd="",
                 db="text2sign")
cursor=connection.cursor()
abc='a'
cursor.execute("select * from sign where letter=%s",(abc))
data2 = cursor.fetchall()
print(data2[0][0])
file_like2 = io.BytesIO(data2[0][1])

img1=Image.open(file_like2)
img1.show()
cursor.close()
connection.close()
"""
"""
try:
    connection = mysql.connector.connect(host='localhost',
                                         database='text2sign',
                                         user='root',
                                         password='')
    if connection.is_connected():
        db_Info = connection.get_server_info()
        print("Connected to MySQL Server version ", db_Info)
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)

    sql_select_Query = "select sign from sign where letter='a'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    print("Total number of rows in sign table is: ", cursor.rowcount)

    print("\nPrinting each sign record")
"""
"""
    for row in records:
        print("a")
        print("sign = ", records)
"""

"""
except Error as e:
    print("Error while connecting to MySQL", e)
finally:
    if (connection.is_connected()):
        cursor.close()
        connection.close()
        print("MySQL connection is closed")

"""
"""
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
"""
"""
text = "Backgammon is one of the oldest known board games. Its history can be traced back nearly 5,000 years to archeological discoveries in the Middle East. It is a two player game where each player has fifteen checkers which move between twenty-four points according to the roll of two dice."
sentences = nltk.sent_tokenize(text)
for sentence in sentences:
    print(sentence)
print("sentence tokenize")
for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    print(words)
print("word tokenize")
"""
"""
# stemmer lemmatizer

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

def compare_stemmer_and_lemmatizer(stemmer, lemmatizer, word, pos):
    
  Print the results of stemmind and lemmitization using the passed stemmer, lemmatizer, word and pos (part of speech)
  
    print("Stemmer:", stemmer.stem(word))
    print("Lemmatizer:", lemmatizer.lemmatize(word, pos))
    print()

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
compare_stemmer_and_lemmatizer(stemmer, lemmatizer, word = "advise", pos = wordnet.VERB)
compare_stemmer_and_lemmatizer(stemmer, lemmatizer, word = "playing", pos = wordnet.VERB)
"""
"""
# stop words

from nltk.corpus import stopwords
#print(stopwords.words("english"))

# how to remove stop words in a sentence
stop_words = set(stopwords.words("english"))
sentence = "Backgammon is one of the oldest known board games."

words = nltk.word_tokenize(sentence)
without_stop_words = [word for word in words if not word in stop_words]
print(without_stop_words)
# another way
stop_words = set(stopwords.words("english"))
sentence = "Backgammon is one of the oldest known board games."

words = nltk.word_tokenize(sentence)
without_stop_words = []
for word in words:
    if word not in stop_words:
        without_stop_words.append(word)

print(without_stop_words)
"""
"""
# Import the libraries we need
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Step 2. Design the Vocabulary
# The default token pattern removes tokens of a single character. That's why we don't have the "I" and "s" tokens in the output
count_vectorizer = CountVectorizer()
corpus = [
   'This is the first document.',
 'This document is the second document.',
  'And this is the third one.',
    'Is this the first document?' ]
# Step 3. Create the Bag-of-Words Model
bag_of_words = count_vectorizer.fit_transform(corpus)

# Show the Bag-of-Words Model as a pandas DataFrame
feature_names = count_vectorizer.get_feature_names()
a=pd.DataFrame(bag_of_words.toarray(), columns = feature_names)
print(a)
"""
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
documents = [
   'This is the first document.',
 'This document is the second document.',
  'And this is the third one.',
    'Is this the first document?' ]
tfidf_vectorizer = TfidfVectorizer()
values = tfidf_vectorizer.fit_transform(documents)

# Show the Model as a pandas DataFrame
feature_names = tfidf_vectorizer.get_feature_names()
a=pd.DataFrame(values.toarray(), columns = feature_names)
print(a)
"""
""""
import nltk
from nltk.corpus import treebank
import nltk.parse.api
import lark
sentence = "alice loves Bob"
#sentence = "abcd, xyz!"
tokens = nltk.word_tokenize(sentence)
print(tokens)
"""
# ['At', 'eight', "o'clock", 'on', 'Thursday', 'morning','Arthur', 'did', "n't", 'feel', 'very', 'good', '.']
# tagged = nltk.pos_tag(tokens)
# print(tagged[0:6])
# [('At', 'IN'), ('eight', 'CD'), ("o'clock", 'JJ'), ('on', 'IN'),('Thursday', 'NNP'), ('morning', 'NN')]
# entities = nltk.chunk.ne_chunk(tagged)
# print(entities)

"""
Tree('S', [('At', 'IN'), ('eight', 'CD'), ("o'clock", 'JJ'),
           ('on', 'IN'), ('Thursday', 'NNP'), ('morning', 'NN'),
       Tree('PERSON', [('Arthur', 'NNP')]),
           ('did', 'VBD'), ("n't", 'RB'), ('feel', 'VB'),
           ('very', 'RB'), ('good', 'JJ'), ('.', '.')])
"""

# Display a parse tree:
# t = treebank.parsed_sents('wsj_0001.mrg')[1]
# print(t.draw())


# grammar=nltk.CFG.fromstring(
#  """S -> NP VP
# VP -> V NP
# NP -> 'alice' |'Bob'
# V -> 'loves'
# """)

"""
parser=nltk.ChartParser(grammar)
trees=parser.parse_all(tokens)
for tree in trees:

    print(tree.draw())
"""
""""

import cv2
from lark import Lark

l = Lark('''start: WORD "," WORD "!"

            %import common.WORD   // imports from terminal library
            %ignore " "           // Disregard spaces in text
         ''')

print(l.parse("Hello, world!"))
"""
# main code
# string = "Hello world!"
"""
def split_str(s):
  return list(s)
print("string: ", tokens)
print("split string...")
i=0
for ch in tokens[i]:
    a=split_str(ch)
    print(a)
print(a[0][0])
"""
"""
# for image display from database
connection = pymysql.connect(host="localhost",
                             user="root",
                             passwd="",
                             db="text2sign")
cursor = connection.cursor()
abc = str('a')

for abc in range(len(abc)):
    sql="select sign from sign where letter = %s"
    cursor.execute(sql,abc)
    data2 = cursor.fetchall()
    for row in data2:
        #print(row)
        file_like2 = io.BytesIO(row[0])
        img1 = Image.open(file_like2)
        img1.show()

"""
"""
for i in file:
    img1 = Image.open(i)
    img1.show()
"""
"""
  for x in :
      print(data2)
      file_like2 = io.BytesIO(data2)

img1=Image.open(file_like2)
img1.show()

cursor.close()
connection.close()
"""
"""
for x in tokens:
  print(x)
  for i in x:
      print(i)
"""
"""
connection = pymysql.connect(host="localhost",
                             user="root",
                             passwd="",
                             db="text2sign")
cursor = connection.cursor()

for x in range(0,len(abc)):
    #print(x)
    cursor.execute("select sign from sign where letter = %s", (abc[x],))
    data2=cursor.fetchall()
    #print(data2)
    file_like2 = io.BytesIO(data2[0][0])
    img1 = Image.open(file_like2)
    img1.show()
    x += 1
"""

# perfectly execute but not execute space
import speech_recognition as sr;
# import os;
import pyaudio as py;
import glob
from PIL import Image
import os
import pygame
import pymysql
import io
from PIL import Image
import cv2

r = sr.Recognizer();
mic = sr.Microphone();
abc

def main():
    pygame.init()
    SIZE = WIDTH, HEIGHT = 720, 480
    BACKGROUND_COLOR = pygame.Color('white')
    FPS = 60
    screen = pygame.display.set_mode(SIZE)
    clock = pygame.time.Clock()
    connection = pymysql.connect(host="localhost",
                                 user="root",
                                 passwd="",
                                 db="text2sign")
    cursor = connection.cursor()
    image_no = 1
    # print(sr.Microphone.list_microphone_names())
    try:
        with mic as source:
            print("Listening.....")
            audio = r.listen(source)
            abc = r.recognize_google(audio)
            print(abc)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    #print(abc)
    for x in range(0, len(abc)):
        cursor.execute("select sign from sign where letter = %s", (abc[x],))
        data2 = cursor.fetchall()
        file_like2 = io.BytesIO(data2[0][0])

        img1 = Image.open(file_like2)
        file_like2.seek(0)
        img1.show()
        # x += 1
        # print(abc[x])
        # with open(img1, "rb") as file:
        # img1 = Image.open(file)
        imgResult = img1.resize((300, 300), resample=Image.BILINEAR)
        name = r'C:\Users\MRC\Desktop\storedimages\image' + str(image_no) + '.jpg'
        imgResult.save(name)

        image_no += 1

    images = load_images(r'C:\Users\MRC\Desktop\storedimages')
    player = AnimatedSprite(position=(100, 100), images=images)
    all_sprites = pygame.sprite.Group(player)  # Creates a sprite group and adds 'player' to it.

    running = True
    while running:

        dt = clock.tick(FPS) / 6000  # Amount of seconds between each loop.

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    player.velocity.x = 4
                elif event.key == pygame.K_LEFT:
                    player.velocity.x = -4
                elif event.key == pygame.K_DOWN:
                    player.velocity.y = 4
                elif event.key == pygame.K_UP:
                    player.velocity.y = -4
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_RIGHT or event.key == pygame.K_LEFT:
                    player.velocity.x = 0
                elif event.key == pygame.K_DOWN or event.key == pygame.K_UP:
                    player.velocity.y = 0

        all_sprites.update(dt)  # Calls the 'update' method on all sprites in the list (currently just the player).

        screen.fill(BACKGROUND_COLOR)
        all_sprites.draw(screen)
        pygame.display.update()


def load_images(path):
    images = []
    for file_name in os.listdir(path):
        image = pygame.image.load(path + os.sep + file_name).convert()
        images.append(image)
    return images


class AnimatedSprite(pygame.sprite.Sprite):
    def __init__(self, position, images):
        """
                Animated sprite object.

                Args:
                    position: x, y coordinate on the screen to place the AnimatedSprite.
                    images: Images to use in the animation.
                """
        super(AnimatedSprite, self).__init__()

        size = (500, 375)  # This should match the size of the images.

        self.rect = pygame.Rect(position, size)
        self.images = images
        self.images_right = images
        self.images_left = [pygame.transform.flip(image, True, False) for image in
                            images]  # Flipping every image.
        self.index = 0
        self.image = images[self.index]  # 'image' is the current image of the animation.

        self.velocity = pygame.math.Vector2(0, 0)

        self.animation_time = 0.1
        self.current_time = 0

    #        self.animation_frames = 6
    #       self.current_frame = 0

    def update_time_dependent(self, dt):
        """
                Updates the image of Sprite approximately every 0.1 second.

                Args:
                    dt: Time elapsed between each frame.
                """
        if self.velocity.x > 0:  # Use the right images if sprite is moving right.
            self.images = self.images_right
        elif self.velocity.x < 0:
            self.images = self.images_left

        self.current_time += dt
        if self.current_time >= self.animation_time:
            self.current_time = 0
            self.index = (self.index + 1) % len(self.images)
            self.image = self.images[self.index]
        # self.rect.move_ip(*self.velocity)

    def update(self, dt):
        """This is the method that's being called when 'all_sprites.update(dt)' is called."""
        # Switch between the two update methods by commenting/uncommenting.
        self.update_time_dependent(dt)
        # self.update_frame_dependent()


if __name__ == '__main__':
    main()

    """
        def update_frame_dependent(self):

    #        Updates the image of Sprite every 6 frame (approximately every 0.1 second if frame rate is 60).

            if self.velocity.x > 0:  # Use the right images if sprite is moving right.
                self.images = self.images_right
            elif self.velocity.x < 0:
                self.images = self.images_left

            self.current_frame += 1
            if self.current_frame >= self.animation_frames:
                self.current_frame = 0
                self.index = (self.index + 1) % len(self.images)
                self.image = self.images[self.index]

            self.rect.move_ip(*self.velocity)
    """
