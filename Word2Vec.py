# -*- coding: utf-8 -*-
from underthesea import word_tokenize
from abc import ABC
import codecs
import numpy as np
from gensim.models import Word2Vec
from gensim.models import FastText
import re
from scipy.stats import entropy
#khao báo danh sách
words=[] ## Danh sach rong = [] để chứa các từ: chiều
WordTokens = []
StopWordsInput = []  ## Danh sách stopword trong van bản
ListSentence = []    ## Danh sách chưa các câu (Documnent

class WordSegment(ABC):
    def parseword(self):
        pass   #

class ViWordSegment(WordSegment):
    # overriding abstract method
    def parseword(self):
        StopWordList = ReadStopWordList("stopwordsVi.txt")
        for word in word_tokenize(Text):
            NewWord = word.replace('.', '').replace(',', '').strip()
            WordTokens.append(NewWord)
            if not (NewWord in words) and NewWord != '':
                if not (NewWord in StopWordList):
                    words.append(NewWord.lower())
                else:
                    StopWordsInput.append(NewWord)

def ReadStopWordList(fName):
    fo = codecs.open(fName, encoding='utf-8', mode='r')
    strContain = fo.read()
    fo.close()
    return strContain.split('\r\n')


def transform_row(row):
    # Xóa số dòng ở đầu câu
    row = re.sub(r"^[0-9\.]+", "", row)

    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$", "", row)

    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ")

    row = row.strip()
    return row

# Bat dau thực hiện chương trình 
# khai bao dữ liệu để test
# corpus = ["tôi yêu công việc lập trình.",
#           "Tiếng Anh với tôi cũng rất căng.",
#           "Python là 1 ngôn ngữ lập trình",
#             "Tôi rất thích bóng đá",
#            "tôi ghét ở một mình"]

# corpus = [
#     "Đã bấy lâu nay bác tới nhà",
# "Trẻ thời đi vắng, chợ thời xa",
# "Ao sâu nước cả, khôn chài cá",
# "Vườn rộng rào thưa, khó đuổi gà",
# "Cải chửa ra cây, cà mới nụ",
# "Bầu vừa rụng rốn, mướp đương hoa",
# "Đầu trò tiếp khách, trầu không có",
# "Bác đến chơi đây ta với ta",
# ]
# corpus = [
#     'Một màu xanh xanh chấm thêm vàng vàng',
#     'Một màu xanh chấm thêm vàng cánh đồng hoang vu',
#     'Một màu nâu nâu một màu tím tím',
#     'Màu nâu tím mắt em tôi ôi đẹp dịu dàng',
#     'Một màu xanh lam chấm thêm màu chàm',
#     'Thời chinh chiến đã xa rồi sắc màu tôi',
#     'Một màu đen đen một màu trắng trắng',
#     'Chiều hoang vắng chiếc xe tang đi vội vàng'
# ]
# corpus = [
#     'Machine Learning và AI trong thời gian qua đã đạt được các thành tựu vô cùng đáng kinh ngạc',
#     'Blockchain - từ công nghệ tiền ảo đến ứng dụng tương lai',
#     'Tác hại kinh hoàng của game online với giới trẻ hiện nay',
#     'Mâu thuẫn khi chơi game và nam sinh giết hại bạn của mình cho bõ tức',
#     'Trí tuệ nhân tạo OpenAI chính thức đánh bại 5 game thủ chuyên nghiệp giỏi nhất thế giới'
#
# ]
# đọc file 
corpus = []
with open('tuyen_ngon_doc_lap.txt', 'r', encoding='utf8') as file:
     for sentences in file:
         corpus.append(transform_row(sentences))
for sentences in corpus:
    W = None
    Text = sentences
    ListSentence.append(Text)
    W = ViWordSegment()
    W.parseword()

words = list(set(words))
words.sort()
X = np.zeros([len(words), len(words)])
for sentences in corpus:
    # tương tu cũng loai bo stopword tung cau
    tokens = []
    for word in word_tokenize(sentences):
        NewWord = word.replace('.', '').replace(',', '').strip()
        if NewWord != '':
         if not (NewWord in StopWordsInput):
             tokens.append(NewWord.lower())

data = []
for sentences in corpus:

    tokens = []
    for word in word_tokenize(sentences):
        NewWord = word.replace('.', '').replace(',', '').strip()
        if NewWord != '':
         if not (NewWord in StopWordsInput):
             tokens.append(NewWord.lower())
    data.append(tokens)
print('tập data:',data)
modelW2V_Gensim = Word2Vec(data,
                                size=100,
                                min_count=2,  # số lần xuất hiện thấp nhất của mỗi từ vựng
                                window=4,  # khai báo kích thước windows size
                                sg=1,  # sg = 1 sử dụng mô hình skip-grams - sg=0 -> sử dụng CBOW
                                workers=1

                                )
modelW2V_Gensim.init_sims(replace=True)

# # Saving the model for later use. Can be loaded using Word2Vec.load()
model_name = "300features_40minwords_10context"
modelW2V_Gensim.save(model_name)
print('Tìm top-10 từ tương đồng với từ: [độc lập]')
for index, word_tuple in enumerate(modelW2V_Gensim.wv.most_similar("độc lập")):
    print('%s.%s\t\t%s\t%s' % (index, word_tuple[0], word_tuple[1],word_tuple))

