import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


def predictProba(org, sex,age,height,weight,IMT,waist,smoke_index,CHSS,ADS,ADD,respiratory_rate,breath_holding ,exp,
                 chemical_factor,dust,work_difficult ,ALT ,AST,total_bilirubin ,direct_bilirubin ,glucose ,creatinine,
                 LLPN ,uric_acid ,triglycerides ,total_cholesterol ,alkaline_phosphatase ,gamma_glutamyl ,atherogenicity_index):
    data = np.array([[org, sex,age,height,weight,IMT,waist,smoke_index,CHSS,ADS,ADD,respiratory_rate,breath_holding ,exp,
                 chemical_factor,dust,work_difficult ,ALT ,AST,total_bilirubin ,direct_bilirubin ,glucose ,creatinine,
                 LLPN ,uric_acid ,triglycerides ,total_cholesterol ,alkaline_phosphatase ,gamma_glutamyl ,atherogenicity_index]])
    return model.predict_proba(data)


def predictDisease(org, sex,age,height,weight,IMT,waist,smoke_index,CHSS,ADS,ADD,respiratory_rate,breath_holding ,exp,
                 chemical_factor,dust,work_difficult ,ALT ,AST,total_bilirubin ,direct_bilirubin ,glucose ,creatinine,
                 LLPN ,uric_acid ,triglycerides ,total_cholesterol ,alkaline_phosphatase ,gamma_glutamyl ,atherogenicity_index):
    data = np.array([[org, sex,age,height,weight,IMT,waist,smoke_index,CHSS,ADS,ADD,respiratory_rate,breath_holding ,exp,
                 chemical_factor,dust,work_difficult ,ALT ,AST,total_bilirubin ,direct_bilirubin ,glucose ,creatinine,
                 LLPN ,uric_acid ,triglycerides ,total_cholesterol ,alkaline_phosphatase ,gamma_glutamyl ,atherogenicity_index]])
    return model.predict(data)


def load_model():
    spiro = pd.read_excel('spiro.xlsx')
    spiro_clear = spiro.drop(
        ['Альбумин, г/л', 'ЛПВП, ммоль/л', 'Мочевина, ммоль/л', 'Белок общий, г/л', 'Норма', 'Рестрикция', 'Об и рес',
         'Обструкция', 'Заболевание', 'Наруш', 'Код_наруш'], axis=1).dropna()

    dict_org = {
        'ЕПК': 1,
        'НПЗ': 2,
        'Лысогорская птицефабрика': 3,
        'Симоновская птицефабрика': 4
    }

    spiro_clear['Организация/профгруппа'] = spiro_clear['Организация/профгруппа'].replace(dict_org).astype(int)

    X = spiro_clear.drop(['Нарушение без уточнений'], axis=1)
    y = spiro_clear['Нарушение без уточнений']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(X_train, y_train)
    return rf_classifier


model = load_model()

st.title('Прогнозирование')

st.subheader("Введите Ваши данные")
org_dict = ['ЕПК',
        'НПЗ',
        'Лысогорская птицефабрика',
        'Симоновская птицефабрика']
org = st.selectbox('Организация/профгруппа', org_dict)

sex_options = ['жен', 'муж']
sex = st.selectbox('Пол', sex_options)

age = st.number_input('Возраст')
height = st.number_input('Рост, см')
weight = st.number_input('Вес, кг')
IMT = st.number_input('ИМТ')
waist = st.number_input('Обх.талии, см')
smoke_index = st.number_input('Индекс курения')
CHSS = st.number_input('ЧСС (уд/мин.)')
ADS = st.number_input('АДС(мм рт. ст)')
ADD = st.number_input('АДД(мм рт. ст)')
respiratory_rate = st.number_input('Частота дыхательных движений')
breath_holding = st.number_input('Задержка дыхания после глубокого вдоха')
exp = st.number_input('Стаж (количество лет)')
chemical_factor = st.number_input('Химический фактор')
dust = st.number_input('Пыль')
work_difficult = st.number_input('Тяжесть трудового процесса')

ALT = st.number_input('АЛТ, 1/л')
AST = st.number_input('АСТ, 1/л')
total_bilirubin = st.number_input('Билирубин общий, мкмоль/л')
direct_bilirubin = st.number_input('Билирубин прямой (связанный), мкмоль/л')
glucose = st.number_input('Глюкоза, ммоль/л')
creatinine = st.number_input('Креатинин, мкмоль/л')
LLPN = st.number_input('ЛПНП, ммоль/л')
uric_acid = st.number_input('Мочевая кислота, мкмоль/л')
triglycerides = st.number_input('Триглицериды, ммоль/л')
total_cholesterol = st.number_input('Общий холестерин, ммоль/л')
alkaline_phosphatase = st.number_input('Щелочная фосфатаза, Ед/л')
gamma_glutamyl = st.number_input('Гамма-глутамилтрансфераза, Ед./л')
atherogenicity_index = st.number_input('Индекс атерогенности')


done = st.button('Вычислить риски')

if done:

    if sex == "муж":
        sex_value = 1
    else:
        sex_value = 0

    org_value = 0
    if org == "ЕПК":
        org_value = 1
    elif org == 'НПЗ':
        org_value = 2
    elif org == 'Лысогорская птицефабрика':
        org_value = 3
    else:
        org_value = 4



    result = predictProba(org_value, sex_value, age, height, weight, IMT, waist, smoke_index, CHSS, ADS, ADD, respiratory_rate,
                     breath_holding, exp,  chemical_factor, dust, work_difficult, ALT, AST, total_bilirubin, direct_bilirubin,
                     glucose, creatinine, LLPN, uric_acid, triglycerides, total_cholesterol, alkaline_phosphatase, gamma_glutamyl, atherogenicity_index)

    rec = predictDisease(org_value, sex_value, age, height, weight, IMT, waist, smoke_index, CHSS, ADS, ADD, respiratory_rate,
                       breath_holding, exp,chemical_factor, dust, work_difficult, ALT, AST, total_bilirubin, direct_bilirubin,
                       glucose, creatinine,LLPN, uric_acid, triglycerides, total_cholesterol, alkaline_phosphatase, gamma_glutamyl,
                       atherogenicity_index)
    if rec is None:
        st.error("Не удалось рассчитать.")
    else:
        if rec == 1:
            rec_value = 'Есть риск нарушения слуха! Рекомендуется немедленное посещение врача'
        else:
            rec_value = 'Риск потери слуха маловероятен'
        st.text(rec_value)
