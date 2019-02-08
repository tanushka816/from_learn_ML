'''
Нужно подобрать оптимальное значение k для алгоритма kNN
Набор данных - Wine
https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
Требуется предсказать сорт винограда, из которого изготовлено вино,
используя результаты химических анализов
'''
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale

import warnings  # меня бесили эти ворнинги!
warnings.filterwarnings("ignore")

data = pandas.read_csv('wine.data.csv', names=['Class', 'Alcohol',
                                                'Malic acid', 'Ash',	'Alcalinity of ash',
                                                'Magnesium', 'Total phenols', 'Flavanoids',
                                                'Nonflavanoid phenols', 'Proanthocyanins',
                                                'Color intensity',  'Hue', 'OD280/OD315 of diluted wines',	'Proline' ])
y = data[['Class']]  # целевой класс
X = data.drop(['Class'], axis='columns')  # признаки
opt_list = []  # лист результатов оценки

# Генератор разбиений sklearn.model_selection.KFold
# задает набор разбиений на обучение и валидацию.
# Число блоков в кросс-валидации определяется параметром n_splits
# Порядок следования объектов в выборке может быть неслучайным,
# это может привести к смещенности кроссвалидационной оценки.
# Чтобы устранить такой эффект, объекты выборки случайно перемешивают перед разбиением на блоки.
# Для перемешивания параметр shuffle=True
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for k in range(1, 51):
    # Поиск точности классификации на кросс-валидации
    # для метода k ближайших соседей
    # (sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50
    # В качестве меры качества - доля верных ответов scoring='accuracy'
    neigh = KNeighborsClassifier(n_neighbors=k)
    opt_list.append(cross_val_score(neigh, X, y, cv=kf, scoring='accuracy'))

opt_what = pandas.DataFrame(opt_list, range(1, 51)).mean(axis=1).sort_values(ascending=False)
top_opt = opt_what.head(1)
print(1, top_opt.index[0])  # k при котором получилось оптимальное качество
print(2, top_opt.values[0])  # само качество
##  k = 1, качество = 0.7
##  скорее всего метод плохо работает на имеющихся признаках.
##  Можно ли получить большее качество за счет приведения признаков к одному масштабу?
##################
X = scale(X)  # масштабирование признаков с помощью функции sklearn.preprocessing.scale
#  после масштабирования получается матрица,  в которой
# каждый столбец имеет нулевое среднее значение и единичное стандартное отклонение
# снова кросс-валидация
opt_list_scale = []
for k in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=k)
    opt_list_scale.append(cross_val_score(neigh, X, y, cv=kf, scoring='accuracy'))

opt_dataf_scale = pandas.DataFrame(opt_list_scale, range(1, 51)).mean(axis=1).sort_values(ascending=False)
top_opt_scale = opt_dataf_scale.head(1)
print(1, top_opt_scale.index[0])  # k при котором получилось оптимальное качество
print(2, top_opt_scale.values[0])  # само качество
##  k = 29, качество = 0.98
##  то, что оптимальный результат достигается при k>1 более логичный результат, чем до масштабирования





