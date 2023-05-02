from flask import Flask, render_template, request, flash
from wtforms import Form, FloatField, SubmitField, validators
import numpy as np
import joblib

# 学習済みモデルを読み込み利用する
def predict(parameters):
    # ニューラルネットワークのモデルを読み込み
    model = joblib.load('./nn.pkl')
    params = parameters.reshape(1,-1)
    pred = model.predict(params)
    return pred

# ラベルからWineの名前を取得する
def getName(label):
    print(label)
    if label == 0:
        return "wine 1"
    elif label == 1: 
        return "wine 2"
    elif label == 2: 
        return "wine 3"
    else: 
        return "Error"

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'zJe09C5c3tMf5FnNL09C5d6SAzZoY'

# Flaskとwtformsを使い、index.html側で表示させるフォームを構築する
class WineForm(Form):   
    alcohol = FloatField("alcohol (アルコール度数)",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=10, max=15)])

    malic_acid  = FloatField("malic_acid (リンゴ酸)",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])

    ash = FloatField("ash (灰分)",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])

    alcalinity_of_ash  = FloatField("alcalinity_of_ash (灰分のアルカリ度)",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=10, max=35)])
    
    magnesium = FloatField("magnesium (マグネシウム)",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=65, max=170)])
    
    total_phenols = FloatField("total_phenols (全フェノール量)",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])
    
    flavanoids = FloatField("flavanoids (フラボノイド)",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])
    
    nonflavanoid_phenols = FloatField("nonflavanoid_phenols(非フラボノイドフェノール)",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=1)])
    
    proanthocyanins = FloatField("proanthocyanins (プロアントシアニン)",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=5)])
    
    color_intensity = FloatField("color_intensity (色の濃さ)",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=15)])
    
    hue = FloatField("hue (色相)",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=2)])
    
    od280od315_of_diluted_wines = FloatField("od280od315_of_diluted_wines (吸光度の比)",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=5)])
    
    proline = FloatField("proline (プロリン)",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=250, max=1700)])


    # html側で表示するsubmitボタンの表示
    submit = SubmitField("判定")

@app.route('/', methods = ['GET', 'POST'])
def predicts():
    form = WineForm(request.form)
    if request.method == 'POST':
        #request.method でPOSTメソッドを取り出す
        if form.validate() == False:
            flash("全て入力する必要があります。")
            return render_template('index.html', form=form)
        #index.htmlを表示する　formを変数としてtemplateに渡す
        else:            
            alcohol = float(request.form["alcohol"])            
            malic_acid  = float(request.form["malic_acid"])            
            ash = float(request.form["ash"])            
            alcalinity_of_ash  = float(request.form["alcalinity_of_ash"])
            magnesium = float(request.form["magnesium"])
            total_phenols = float(request.form["total_phenols"])
            flavanoids = float(request.form["flavanoids"])
            nonflavanoid_phenols = float(request.form["nonflavanoid_phenols"])
            proanthocyanins = float(request.form["proanthocyanins"])
            color_intensity = float(request.form["color_intensity"])
            hue = float(request.form["hue"])
            od280od315_of_diluted_wines = float(request.form["od280od315_of_diluted_wines"])
            proline = float(request.form["proline"])
            #request.form POST送信された値を取得するプロパティ
            
            from sklearn.model_selection import train_test_split
            from sklearn.datasets import load_wine
            wine = load_wine()
            x, y = wine.data, wine.target
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

            #標準化された値で学習データを作っているため、Webで打ち込んだ値を同じ関数で返す
            alcohol_2 = (alcohol-np.mean(x_train, axis=0)[0])/np.std(x_train, axis=0)[0]
            malic_acid_2 = (malic_acid-np.mean(x_train, axis=0)[1])/np.std(x_train, axis=0)[1]
            ash_2 = (ash-np.mean(x_train, axis=0)[2])/np.std(x_train, axis=0)[2]
            alcalinity_of_ash_2 = (alcalinity_of_ash-np.mean(x_train, axis=0)[3])/np.std(x_train, axis=0)[3]
            magnesium_2 = (magnesium-np.mean(x_train, axis=0)[4])/np.std(x_train, axis=0)[4]
            total_phenols_2 = (total_phenols-np.mean(x_train, axis=0)[5])/np.std(x_train, axis=0)[5]
            flavanoids_2 = (flavanoids-np.mean(x_train, axis=0)[6])/np.std(x_train, axis=0)[6]
            nonflavanoid_phenols_2 = (nonflavanoid_phenols-np.mean(x_train, axis=0)[7])/np.std(x_train, axis=0)[7]
            proanthocyanins_2 = (proanthocyanins-np.mean(x_train, axis=0)[8])/np.std(x_train, axis=0)[8]
            color_intensity_2 = (color_intensity-np.mean(x_train, axis=0)[9])/np.std(x_train, axis=0)[9]
            hue_2 = (hue-np.mean(x_train, axis=0)[10])/np.std(x_train, axis=0)[10]
            od280od315_of_diluted_wines_2 = (od280od315_of_diluted_wines-np.mean(x_train, axis=0)[11])/np.std(x_train, axis=0)[11]
            proline_2 = (proline-np.mean(x_train, axis=0)[12])/np.std(x_train, axis=0)[12]
            
            #xにWebで打ち込んだ値を修正したものを入れて、predを回す
            x = np.array([alcohol_2, malic_acid_2, ash_2, alcalinity_of_ash_2, magnesium_2, total_phenols_2,\
                      flavanoids_2, nonflavanoid_phenols_2,  proanthocyanins_2, color_intensity_2,\
                      hue_2, od280od315_of_diluted_wines_2, proline_2 ])
            pred = predict(x)
            wineName = getName(pred)

            return render_template('result.html', wineName=wineName)
        #result.htmlを表示　winenameを変数としてtemplateに渡す
    elif request.method == 'GET':

        return render_template('index.html', form=form)

if __name__ == "__main__":
    app.run()
    #Flaskのインスタンスを実行