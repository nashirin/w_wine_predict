# w_wine_predict
scikit-learn の　wine_dataset,flaskを用いて、wineの品種当てwebアプリを作成した

app.py
nn.py
templates
 |-index.html
 |-result.html

nn.pyにてニューラルネットワーク（MLPClassifier）を使って学習させ、データをnn.pklに保存

app.pyにてflaskを使ってWebサーバ側のプログラムを実装
　predict関数ではnn.pklに保存された学習モデルを読み込み、品種のラベルを返す
　getName関数では品種のラベルから「wine 1,wine 2,wine 3」のどれかを返す


