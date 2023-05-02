from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.metrics import classification_report
from sklearn import preprocessing

wine = load_wine()
x, y = wine.data, wine.target

# 訓練データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

#データを標準化
sscaler = preprocessing.StandardScaler()
sscaler.fit(x_train)
x_train_2 = sscaler.transform(x_train)
sscaler.fit(x_test)
x_test_2 = sscaler.transform(x_test)


#確率的勾配降下法(sgd)を適用
model = MLPClassifier(solver="sgd", random_state=0, max_iter=3000)


# 学習　
model.fit(x_train_2, y_train)
pred = model.predict(x_test_2)

# 学習済みモデルの保存
joblib.dump(model, "nn.pkl", compress=True)

# 予測精度
print("result: ", model.score(x_test_2, y_test))
print(classification_report(y_test, pred))
