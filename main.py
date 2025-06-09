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


def predictProba(org, sex,age,height,weight,imt,waist,smoke_index,CHSS,ADS,ADD,respiratory_rate,breath_holding ,exp,
                 chemical_factor,dust,work_difficult ,ALT ,AST,total_bilirubin ,direct_bilirubin ,glucose ,creatinine,
                 LLPN ,uric_acid ,triglycerides ,total_cholesterol ,alkaline_phosphatase ,gamma_glutamyl ,atherogenicity_index):
    data = np.array([[org, sex,age,height,weight,imt,waist,smoke_index,CHSS,ADS,ADD,respiratory_rate,breath_holding ,exp,
                 chemical_factor,dust,work_difficult ,ALT ,AST,total_bilirubin ,direct_bilirubin ,glucose ,creatinine,
                 LLPN ,uric_acid ,triglycerides ,total_cholesterol ,alkaline_phosphatase ,gamma_glutamyl ,atherogenicity_index]])
    return model.predict_proba(data)


def predictDisease(org, sex,age,height,weight,imt,waist,smoke_index,CHSS,ADS,ADD,respiratory_rate,breath_holding ,exp,
                 chemical_factor,dust,work_difficult ,ALT ,AST,total_bilirubin ,direct_bilirubin ,glucose ,creatinine,
                 LLPN ,uric_acid ,triglycerides ,total_cholesterol ,alkaline_phosphatase ,gamma_glutamyl ,atherogenicity_index):
    data = np.array([[org, sex,age,height,weight,imt,waist,smoke_index,CHSS,ADS,ADD,respiratory_rate,breath_holding ,exp,
                 chemical_factor,dust,work_difficult ,ALT ,AST,total_bilirubin ,direct_bilirubin ,glucose ,creatinine,
                 LLPN ,uric_acid ,triglycerides ,total_cholesterol ,alkaline_phosphatase ,gamma_glutamyl ,atherogenicity_index]])
    return model.predict(data)


def calculate_bmi(weight, height):
    if pd.isna(weight) or pd.isna(height):
        return None
    try:
        return round(weight / (height / 100) ** 2, 2)
    except ZeroDivisionError:
        print(f"Ошибка: Рост равен 0 для строки {weight}, {height}")
        return None

def classify_bmi(bmi_value):
    if bmi_value is None:
        return 'Недопределено'
    elif bmi_value < 18.5:
        return 'Недостаточный'
    elif bmi_value < 25:
        return 'Нормальный'
    elif bmi_value < 30:
        return 'Избыточный вес'
    elif bmi_value < 35:
        return 'Ожирение I степени'
    elif bmi_value < 40:
        return 'Ожирение II степени'
    else:
        return 'Ожирение III степени'

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

st.title('Прогнозирование наличия бронхолёгочных патологий')

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

col1, col2 = st.columns(2)
with col1:
    weight_input = st.number_input('Вес (кг)', min_value=0.0, value=70.0, step=0.1)
with col2:
    height_input = st.number_input('Рост (см)', min_value=50.0, max_value=250.0, value=170.0, step=1.0)


# Кнопка для расчета ИМТ
calculate_button = st.button('Рассчитать ИМТ')

if calculate_button:
    bmi_result = calculate_bmi(weight_input, height_input)

    if bmi_result is None:
        st.error("Не удалось рассчитать ИМТ. Проверьте введенные данные.")
    else:
        st.write(f"Ваш ИМТ: {bmi_result:.2f}")

    # Отображаем классификацию по ИМТ
    classification = classify_bmi(bmi_result)
    imt = classification
    bmi = bmi_result
    st.write(f"Классификация: {classification}")

st.subheader("Введите Ваши данные последнего физикального обследования:")

waist = st.number_input('Обх.талии, см')
smoke_index = st.number_input('Индекс курения (можно расчитать на любом сайте с калькулятором индекса курения)')
CHSS = st.number_input('ЧСС (уд/мин.)')
ADS = st.number_input('АДС(мм рт. ст)')
ADD = st.number_input('АДД(мм рт. ст)')
respiratory_rate = st.number_input('Частота дыхательных движений')
breath_holding = st.number_input('Задержка дыхания после глубокого вдоха')
exp = st.number_input('Стаж (количество лет)')
chemical_factor = st.checkbox('Химический фактор')
dust = st.checkbox('Пыль')
work_difficult = st.checkbox('Тяжесть трудового процесса')

st.subheader("Введите Ваши параметры биохимического анализа крови:")

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

    res_chemical_factor = 1 if chemical_factor else 0
    res_dust = 1 if dust else 0
    res_work_difficult = 1 if work_difficult else 0



    result = predictProba(org_value, sex_value, age, height, weight, imt, waist, smoke_index, CHSS, ADS, ADD, respiratory_rate,
                     breath_holding, exp,  chemical_factor, dust, work_difficult, ALT, AST, total_bilirubin, direct_bilirubin,
                     glucose, creatinine, LLPN, uric_acid, triglycerides, total_cholesterol, alkaline_phosphatase, gamma_glutamyl, atherogenicity_index)

    rec = predictDisease(org_value, sex_value, age, height, weight, imt, waist, smoke_index, CHSS, ADS, ADD, respiratory_rate,
                       breath_holding, exp,chemical_factor, dust, work_difficult, ALT, AST, total_bilirubin, direct_bilirubin,
                       glucose, creatinine,LLPN, uric_acid, triglycerides, total_cholesterol, alkaline_phosphatase, gamma_glutamyl,
                       atherogenicity_index)
    if rec is None:
        st.error("Не удалось рассчитать.")
    else:
        if rec == 1:
            rec_value = 'Есть вероятность наличия бронхолёгочной патологии. Необходимо проконсультироваться со специалистом!'
        else:
            rec_value = 'Риск бронхолёгочной патологии маловероятен.'
        st.text(rec_value)
