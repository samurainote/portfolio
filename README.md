# Machine Learning Projects
## 自然言語処理を中心とした機械学習およびディープラーニングの個人プロジェクト
This Repository contains following 3 kinds of projects what I have done as a self-taught ML/NLP Enthusiast.    
I've also published several articles on Medium about Data Science and Natural Language Processing.    
これまで個人で実装してきた機械学習理論や自然言語処理のコードがまとめてあります.     
機械学習や自然言語処理に関する記事をMediumに投稿しています. -> [Coldstart.nlp on Medium](https://medium.com/shortcutnlp)       

## Technical Preferences
| Title | Detail |
|:-----------:|:------------------------------------------------|
| Environment | MacOS Mojave 10.14.3 |
| Language | Python(core), R, HTML5, CSS3 |
| Library | Tensorflow, Kras, Scikit-learn, Chainer, Genism, NLTK, Scipy, Numpy, Seaborn |
| Database | MySQL, PostgreSQL |


## Implementations

### 1. Natural Language Processing 自然言語処理

- #### Seq2Seq センテンスtoセンテンス *on training*
  - [Automatic Encoder-Decoder Seq2Seq: English Chatbot](https://github.com/samurainote/chatbot_slack_keras) Seq2Seqによる英語チャットボット.
  - [Encoder-Decoder Seq2Seq with Attention: English-Japanese Machine Translation](https://github.com/samurainote/seq2seq_translate_slackbot) アテンションを用いた日英翻訳.
  - [Bidirectional LSTM: Abstract Text Summarization](https://github.com/samurainote/Text_Summarization_using_Bidirectional_LSTM) 双方向LSTMを用いた文脈を捉える抽象型文章要約.
  - [GRU: Text_Generation](https://github.com/samurainote/Text_Generation_using_GRU) GRUリカーレントネットワークを用いた文章生成.

- #### Multi-modal Deep Learning マルチモーダルディープラーニング.
  - [CNN and RNN: Image Caption Generation](https://github.com/samurainote/CaptionGeneration_CNNandLSTM_Keras) VGG16とGloveで転移学習を行なった画像キャプション生成.
  - [Convolutional Neural Network: Speech Recognition](https://github.com/samurainote/Speech_Recognition_CNN)　畳み込みニューラルネットワークを用いた音声認識.
  - [OpenCV: Text Extraction from Image](https://github.com/samurainote/OCR_Text_Detection_from_Image) 画像からテキスト情報を抽出する文字起こし機能の実装.

- #### Recommend System レコメンドエンジン × 自然言語処理
  - [Question-Answer Recommendation Engine from Kaggle](https://github.com/samurainote/NLP-meets-Recommendetion-Doc2Vec-based-Recommendation-from-kaggle) *現在進行中 on process*

- #### Classification 分類 × 自然言語処理
  - [Doc2Vec: Hate Speech Detection from Tweet](https://github.com/samurainote/Sentimentment_Analysis_for_hatespeech) Doc2Vecを用いたツイートからのヘイトスピーチ検出.
  - [Topic Model by LDA: News Contents Classification](https://github.com/samurainote/Topic_Model_LDA_for_Text_Classification_with_abcnews) 潜在的ディリクレ配分法（LDA）を用いたコンテンツ分類.
  - [TF-idf and NaiveBayes: Author Identification](https://github.com/samurainote/TF-idf_and_NaiveBayes_for_Author_Identification) Tf-idfとナイーブベイズ分類器による著者判定.
  - [Gaussian Naive Bayes for text: Spam Filter for e-mail](https://github.com/samurainote/Text_Classificasion_Spamfilter_with_GaussianNB) ナイーブベイズ分類器によるEメールスパムフィルター.
  - [CNN for Long Text: News Article Classifications](https://github.com/samurainote/CNN_Convolutional_NN_for_news_contents_classification) 長文テキストにCNNを用いたニュース記事分類.
  - [Bidirectional LSTM: Sentiment Analysis on reviews](https://github.com/samurainote/Bidirectional_LSTM_Sentiment_Analysis_imbd) 双方向LSTMを用いた映画レビューにおける感情分析.
  - [Stacked RNN vs Simple RNN: Sentiment Analysis on movie review](https://github.com/samurainote/StackedRNN_for_Sentiment_Analysis) 複数のRNNを用いた感情分類.
  - [LSTM for Tweet: Sentiment Analysis on Twitter](https://github.com/samurainote/LSTM_for_Sentiment_Analysis_with_Twitter_textdata) LSTMを用いたツイートに対する感情分析.
  - [LSTM with Chainer: Text Classification](https://github.com/samurainote/Text_Classification_LSTM_Chainer/blob/master/code/main_code.ipynb) ChainerによるLSTMを用いた文章分類のモディフィケーション.

- #### Regression 回帰 × 自然言語処理
  - [Regression: Mercari Price Suggestion Challenge](https://github.com/samurainote/mercari_price_prediction_from_kaggle): *現在進行中 on process* mercariにおける出品アイテムの価格予測.

- #### Preprocessing 前処理 × 自然言語処理
  - [Preprocessing Simplest Code-kit for NLP](https://github.com/samurainote/nlp_preprocessing_tool-kit) 自然言語処理における前処理Tips&Code.


### 2. Computer Vision by Machine Learning コンピュータビジョンと画像認識

- #### Image Processing 画像認識 × ディープラーニング
	- [OpenCV: Human Face Detection from Image](https://github.com/samurainote/Face_Detection_with_OpenCV/blob/master/Face%20Detection.ipynb) 画像から人間の顔部分を特定.
  - [Convolutional Neural Network for Image: Dog or Cat](https://github.com/samurainote/Image_Classifier_Dog_or_Cat_with_Keras/blob/master/dogvscat.ipynb) 画像から犬か猫かを見分けるバイナリ分類タスク.
  - [CNN: Sign Language Images Classification](https://github.com/samurainote/CNN_for_Sign_Language_Images) 畳み込みニューラルネットワークを用いた手話画像識別.
  - [Simple Neural Network: Hand-written Digits Classification](https://github.com/samurainote/SimpleNN_for_Handwritten_digits) ニューラルネットワークによる手書き数字認識.
  - [CNN: Hand-written Digits Classification](https://github.com/samurainote/CNN_for_Image_Processing_with_MNIST) 畳み込みニューラルネットワークによる手書き数字文字.
  - [GANs for Image: Generative Adversarial Network] *現在進行中 on process*
  - [Inception from google: Train image classifier with Inception] *現在進行中 on process*


### 3. Recommend System by Machine Learning レコメンドシステム

- #### Recommendation Engine 推薦エンジン
  - [Collaborative Filtering with SVD: Book Recommender System](https://github.com/samurainote/Book_Recommendation) 特異値分解と協調フィルタリングによる本の推薦.
  - [Content Based by Cosine Similarity: Movie Recommender System](https://github.com/samurainote/Content_based_movie_recommendation)　Tfidfとコサイン類似度による映画の推薦.


### 4. Kaggle Challenges: Prediction Tasks ディープラーニングと機械学習を用いた予測モデル

- #### Regression 回帰 × 予測
  - [Regression: HR Salary Prediction](https://github.com/samurainote/Regression_HR_Salary_Prediction/blob/master/maincode_hitters.ipynb) スポーツ選手のシーズン中のパフォーマンスからの年収予想.
  - [Simple RNN for Time-Series Data: Apple Stock Price Prediction](https://github.com/samurainote/Simple_RNN_for_Apple_stock_price_prediction) リカーレントネットワークを用いた株価予測.
  - [Regression: Boston House Price](https://github.com/samurainote/Boston_House_Price_with_Linear_Regression/blob/master/Boston_House_Price_with_Linear_Regression.ipynb) 線形回帰によるボストン地区の不動産価格予測.
  - [Regression: Multi-layerNN_for_Regression_BHP](https://github.com/samurainote/Multi-layerNN_for_Regression_BHP) 多層ニューラルネットワークによるボストン地区の不動産価格予測.

- #### Classification 分類 × 予測
  - [Classification: IBM Attrition Prediction from Kaggle](https://github.com/samurainote/ibm_attrition_classification): IBM社員データを利用した退職者予測.
  - [First Neural Network from scratch](https://github.com/samurainote/Neural_Network_from_scratch) 分類タスク向けのディープラーニングフロムスクラッチ.


### 5. Machine Learning Foundation 機械学習のワークフロー

- #### Machine Learning Workflow 機械学習ワークフロー
  - [1. Data Preparation: Scraping and Web API](https://github.com/samurainote/1._Web_Scraping), [here Medium](): スクレイピングやウェブAPIによるデータの準備.
  - [2. Data Visualization: Which Graph should I use?](https://github.com/samurainote/2._Data_Visualization), [here Medium](https://medium.com/shortcutnlp/03-data-visualization-show-your-skill-of-storytelling-from-data-a50c8818c2db): データの可視化におけるグラフの種類と使い分け.
  - [3. Data Cleaning Phase on Medium](https://github.com/samurainote/3._Data_Cleaning), [here Medium](https://medium.com/shortcutnlp/04-data-cleaning-if-you-feed-better-ml-reply-better-to-your-task-d688f9137022): 前処理におけるTips&code集.
  - [4. Feature Engineering Phase on Medium](https://github.com/samurainote/4._Feature_Engineering), [here Medium](https://medium.com/shortcutnlp/02-feature-engineering-principles-for-choosing-right-features-2503c9bd857): 次元削除と特徴量生成メソッド.
  - [5. Apply Multiple Machine Learning Model Phase on Medium](https://github.com/samurainote/5._Model_Selection), [here Medium](https://medium.com/shortcutnlp/05-model-application-how-to-compare-and-choose-the-best-ml-model-b7cfff804c08): 機械学習アルゴリズムとその比較.
  - [6. Model Validation Phase on Medium](https://github.com/samurainote/6._Model_Validation), [here Medium](https://medium.com/shortcutnlp/validation-metrics-for-machine-learning-task-%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%81%AE%E8%A9%95%E4%BE%A1%E6%8C%87%E6%A8%99%E3%81%BE%E3%81%A8%E3%82%81-ed70d5363c21): モデルの評価指標とその選び方.
  - [7. Hyperparameter Tuning Phase on Medium](https://github.com/samurainote/7._Hyperparameter_Tuning), [here Medium](https://medium.com/shortcutnlp/07-hyperparameter-tuning-a-final-method-to-improve-model-accuracy-b98ba860f2a6): ハイパーパラメータチューニング.
  - [8. Gradient Descent and Optimization Algorithm](https://github.com/samurainote/8._Gradient_Descent_and_Optimazation_Algorithm), [here Medium](https://medium.com/shortcutnlp/shortcutml-gradient-descent-algorithm-optimization-recipe-1e3edf815a5b): 勾配降下と最適化アルゴリズム.
