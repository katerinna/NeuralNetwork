# ����� ������ � ������� ��������������� ���������������� ���������
# � �������� ��������� ���������� ������ ������������� ���� ������, ��� 
# N - ������� ������ ����, Y - �������� ��� ����������� ���

install.packages('kohonen')
library('kohonen')

# ������ ������ �� �������

tab <- read.table("Riders.csv", header=T,sep = ",",as.is=T, dec = ".")

# ����������� �� ��������� ���������� � ����� ��������� ���������� level

x <- tab[,4:11]
level <- factor(tab[,12])

set.seed(1234)

#������ ������������������ ����� �������� ��� ������������� ��������
som.x <- som(scale(x), grid = somgrid(5,5,'hexagonal'))
som.x
plot(som.x, main = 'Kohonen SOM')
plot(som.x, type = 'changes', main = 'Kohonen SOM')


# ����� ������� �������� ������������ ���� ��������,
# ������ ������� �� �������������, ���� ������ 70 ����������, � ��������, ���� ������ 20

train <- sample(nrow(x), 70)
x_train <- scale(x[train,])
x_test <- scale(x[-train,],
                center = attr(x_train, "scaled:center"),
                scale = attr(x_train, "scaled:center"))

train_data <- list(measurements = x_train,
                   level = level[train])

test_data <- list(measurements = x_test,
                  level = level[-train])

# ���������� ��������� �� �����:
som.x <- supersom(train_data, grid = somgrid(5,5,'hexagonal'))
plot(som.x, main = 'Kohonen SOM')

# ������ ������ ���������������� �� ������ � �������� ���

# ���������� �� ������ ���������� ������, ������ ����� �������� 
# ��� � ��������� ����������� ������ 20 ����

som_predict <- predict(som.x, newdata = test_data)
table(level[-train], som_predict$predictions[['level']])

#     N  Y
#  N  5  0
#  Y  0  14

# ����������� ��������� ���� ����������� ������ ��� ������

# ��������� �� ������ �������
class_train <- data.frame(level = level[train], class=x_train)
class_test <- data.frame(level = som_predict$predictions[['level']], class=x_test)

class_train
class_test 