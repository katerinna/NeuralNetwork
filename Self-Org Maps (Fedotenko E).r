# ¬з€ты данные о трафике ћассачусетского железнодорожного сообщени€
# в качестве факторной переменной возьмЄм классификацию дней недели, где 
# N - обычный будний день, Y - выходные или праздничные дни

install.packages('kohonen')
library('kohonen')

# чтение данных из таблицы

tab <- read.table("Riders.csv", header=T,sep = ",",as.is=T, dec = ".")

# избавл€емс€ от текстовых переменных и задаЄм факторную переменную level

x <- tab[,4:11]
level <- factor(tab[,12])

set.seed(1234)

#строим самоорганизующуюс€ карту  охонена дл€ кластеризации объектов
som.x <- som(scale(x), grid = somgrid(5,5,'hexagonal'))
som.x
plot(som.x, main = 'Kohonen SOM')
plot(som.x, type = 'changes', main = 'Kohonen SOM')


# далее проведЄм алгоритм самообучени€ карт  охонена,
# разбив выборку на тренировочную, куда войдут 70 наблюдений, и тестовую, куда войдут 20

train <- sample(nrow(x), 70)
x_train <- scale(x[train,])
x_test <- scale(x[-train,],
                center = attr(x_train, "scaled:center"),
                scale = attr(x_train, "scaled:center"))

train_data <- list(measurements = x_train,
                   level = level[train])

test_data <- list(measurements = x_test,
                  level = level[-train])

# результаты изобразим на карте:
som.x <- supersom(train_data, grid = somgrid(5,5,'hexagonal'))
plot(som.x, main = 'Kohonen SOM')

# данные теперь классифицируютс€ на будние и выходные дни

# предскажем на основе полученных правил, какими будут €вл€тьс€ 
# дни с заданными параметрами других 20 дней

som_predict <- predict(som.x, newdata = test_data)
table(level[-train], som_predict$predictions[['level']])

#     N  Y
#  N  5  0
#  Y  0  14

# построенна€ нейронна€ сеть предсказала данные без ошибок

# посмотрим на номера классов
class_train <- data.frame(level = level[train], class=x_train)
class_test <- data.frame(level = som_predict$predictions[['level']], class=x_test)

class_train
class_test 