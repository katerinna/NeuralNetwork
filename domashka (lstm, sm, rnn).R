# необходимые для работы пакеты

install.packages('BatchGetSymbols')
install.packages('plotly')
install.packages('tensorflow')
install.packages('keras')
install.packages('ggplot2')
install.packages('minimax')


library(plotly)
library(BatchGetSymbols)
library('keras')
library('tensorflow')
library('minimax')

# загрузка данных по индексу Nikkei - Япония 
tickers <- c('%5EN225')
first.date <- Sys.Date() - 360*5
last.date <- Sys.Date()

yts <- BatchGetSymbols(tickers = tickers,
                       first.date = first.date,
                       last.date = last.date,
                       cache.folder = file.path(tempdir(),
                                                'BGS_Cache') )

y <-  yts$df.tickers$price.close
myts <-  data.frame(index = yts$df.tickers$ref.date, price = y, vol = yts$df.tickers$volume)
myts <-  myts[complete.cases(myts), ]
myts <-  myts[-seq(nrow(myts) - 1200), ]
myts$index <-  seq(nrow(myts))

# стандартизация 

msd.price <-  c(mean(myts$price), sd(myts$price))
msd.vol <-  c(mean(myts$vol), sd(myts$vol))
myts$price <-  (myts$price - msd.price[1])/msd.price[2]
myts$vol <-  (myts$vol - msd.vol[1])/msd.vol[2]
summary(myts)

# деление на тестовую и тренировочную

datalags = 20
train <-  myts[seq(1000 + datalags), ]
test <-  myts[1000 + datalags + seq(200 + datalags), ]
batch.size <- 50

# подготовка входных данных

x.train <-  array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags, 2))
y.train = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags, 1))
dim(x.train)
dim(y.train)
x.test <-  array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags, 2))
y.test <-  array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags, 1))

# вывод интерактивного графика котировок
plot_ly(myts, x = ~index, y = ~price, type = "scatter", mode = "markers", color = ~vol)
# автокорреляционная функция
acf(myts$price, lag.max = 1200)

#########################################################################
# LSTM / adam / mse
model <- keras_model_sequential()  %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

model %>%
  compile(loss = 'mse', optimizer = 'adam')
model %>% fit(x.train, y.train, epochs = 40, batch_size = batch.size)

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]

lstm_adam_mse = 0.171

#########################################################################
# LSTM / adam / mae
model <- keras_model_sequential()  %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

model %>%
  compile(loss = 'mae', optimizer = 'adam')
model %>% fit(x.train, y.train, epochs = 25, batch_size = batch.size)

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]

lstm_adam_mae = 0.34

#########################################################################
# LSTM / adam / mape
model <- keras_model_sequential()  %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

model %>%
  compile(loss = 'mape', optimizer = 'adam')
model %>% fit(x.train, y.train, epochs = 30, batch_size = batch.size)

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]

lstm_adam_mape = 1

#########################################################################
# LSTM / rmsprop / mse
model <- keras_model_sequential()  %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

model %>%
  compile(loss = 'mse', optimizer = 'rmsprop')
model %>% fit(x.train, y.train, epochs = 25, batch_size = batch.size)

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]

lstm_rmsprop_mse = 0.17

#########################################################################
# LSTM / rmsprop / mae
model <- keras_model_sequential()  %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

model %>%
  compile(loss = 'mae', optimizer = 'rmsprop')
model %>% fit(x.train, y.train, epochs = 25, batch_size = batch.size)

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]

lstm_rmsprop_mae = 0.34

#########################################################################
# LSTM / rmsprop / mape
model <- keras_model_sequential()  %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

model %>%
  compile(loss = 'mape', optimizer = 'rmsprop')
model %>% fit(x.train, y.train, epochs = 20, batch_size = batch.size)

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]

lstm_rmsprop_mape = 1

##################################################################
########################### SM ###################################
##################################################################

y <-  yts$df.tickers$price.close
myts <-  data.frame(index = yts$df.tickers$ref.date, price = y, vol = yts$df.tickers$volume)
myts <-  myts[complete.cases(myts), ]
myts <-  myts[-seq(nrow(myts) - 1200), ]
myts$index <-  seq(nrow(myts))

# Стандартизация минимакс
myts <- data.frame(index = rminimax(myts$index), price = rminimax(myts$price), vol= rminimax(myts$vol))
myts

# Деление на тестовую и тренировочную
datalags = 20
train <-  myts[seq(1000 + datalags), ]
test <-  myts[1000 + datalags + seq(200 + datalags), ]
batch.size <- 50

# Создание массивов
x.train <-  array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags))
y.train = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags))

x.test <-  array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags))
y.test <-  array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags))

#########################################################################
# SM / relu / adam / mse

model <- keras_model_sequential() %>%
  layer_dense(units = 1, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'sigmoid')

model %>% compile(
  optimizer = 'adam',
  loss = 'mse')
model%>% fit(x.train, y.train, epochs = 10, batch_size = 50)

pred_out <- model %>% predict(x.test, batch_size = 50) %>% .[,1]

sm_relu_adam_mse = 0.082
sm_relu_adam_mse

#########################################################################
# SM / relu / adam / mae

model <- keras_model_sequential() %>%
  layer_dense(units = 1, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'sigmoid')

model %>% compile(
  optimizer = 'adam',
  loss = 'mae')
model%>% fit(x.train, y.train, epochs = 10, batch_size = 50)

pred_out <- model %>% predict(x.test, batch_size = 50) %>% .[,1]

sm_relu_adam_mae = 0.248

#########################################################################
# SM / relu / adam / mae

model <- keras_model_sequential() %>%
  layer_dense(units = 1, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'sigmoid')

model %>% compile(
  optimizer = 'adam',
  loss = 'mae')
model%>% fit(x.train, y.train, epochs = 10, batch_size = 50)

pred_out <- model %>% predict(x.test, batch_size = 50) %>% .[,1]

sm_relu_adam_mae = 0.248

#########################################################################
# SM / relu / adam / mape

model <- keras_model_sequential() %>%
  layer_dense(units = 1, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'sigmoid')

model %>% compile(
  optimizer = 'adam',
  loss = 'mape')
model%>% fit(x.train, y.train, epochs = 30, batch_size = 50)

pred_out <- model %>% predict(x.test, batch_size = 50) %>% .[,1]

sm_relu_adam_mape = 1

#########################################################################
# SM / relu / rmsprop / mae

model <- keras_model_sequential() %>%
  layer_dense(units = 1, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'sigmoid')

model %>% compile(
  optimizer = 'rmsprop',
  loss = 'mae')
model%>% fit(x.train, y.train, epochs = 30, batch_size = 50)

pred_out <- model %>% predict(x.test, batch_size = 50) %>% .[,1]

sm_relu_rmsprop_mae = 0.248

#########################################################################
# SM / relu / rmsprop / mse

model <- keras_model_sequential() %>%
  layer_dense(units = 1, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'sigmoid')

model %>% compile(
  optimizer = 'rmsprop',
  loss = 'mae')
model%>% fit(x.train, y.train, epochs = 10, batch_size = 50)

pred_out <- model %>% predict(x.test, batch_size = 50) %>% .[,1]

sm_relu_rmsprop_mse = 0.248

#########################################################################
# SM / relu / rmsprop / mape

model <- keras_model_sequential() %>%
  layer_dense(units = 1, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'sigmoid')

model %>% compile(
  optimizer = 'rmsprop',
  loss = 'mape')
model%>% fit(x.train, y.train, epochs = 10, batch_size = 50)

pred_out <- model %>% predict(x.test, batch_size = 50) %>% .[,1]

sm_relu_rmsprop_mape = 1

#########################################################################
# SM / sigmoid / adam / mse

model <- keras_model_sequential() %>%
  layer_dense(units = 1, activation = 'sigmoid') %>%
  layer_dense(units = 5, activation = 'sigmoid')

model %>% compile(
  optimizer = 'adam',
  loss = 'mse')
model%>% fit(x.train, y.train, epochs = 10, batch_size = 50)

pred_out <- model %>% predict(x.test, batch_size = 50) %>% .[,1]

sm_sigmoid_adam_mse = 0.082
sm_sigmoid_adam_mse

#########################################################################
# SM / sigmoid / adam / mae

model <- keras_model_sequential() %>%
  layer_dense(units = 1, activation = 'sigmoid') %>%
  layer_dense(units = 5, activation = 'sigmoid')

model %>% compile(
  optimizer = 'adam',
  loss = 'mae')
model%>% fit(x.train, y.train, epochs = 10, batch_size = 50)

pred_out <- model %>% predict(x.test, batch_size = 50) %>% .[,1]

sm_sigmoid_adam_mae = 0.248

#########################################################################
# SM / sigmoid / adam / mae

model <- keras_model_sequential() %>%
  layer_dense(units = 1, activation = 'sigmoid') %>%
  layer_dense(units = 5, activation = 'sigmoid')

model %>% compile(
  optimizer = 'adam',
  loss = 'mae')
model%>% fit(x.train, y.train, epochs = 10, batch_size = 50)

pred_out <- model %>% predict(x.test, batch_size = 50) %>% .[,1]

sm_sigmoid_adam_mae = 0.248

#########################################################################
# SM / sigmoid / adam / mape

model <- keras_model_sequential() %>%
  layer_dense(units = 1, activation = 'sigmoid') %>%
  layer_dense(units = 5, activation = 'sigmoid')

model %>% compile(
  optimizer = 'adam',
  loss = 'mape')
model%>% fit(x.train, y.train, epochs = 30, batch_size = 50)

pred_out <- model %>% predict(x.test, batch_size = 50) %>% .[,1]

sm_sigmoid_adam_mape = 1

#########################################################################
# SM / sigmoid / rmsprop / mae

model <- keras_model_sequential() %>%
  layer_dense(units = 1, activation = 'sigmoid') %>%
  layer_dense(units = 5, activation = 'sigmoid')

model %>% compile(
  optimizer = 'rmsprop',
  loss = 'mae')
model%>% fit(x.train, y.train, epochs = 30, batch_size = 50)

pred_out <- model %>% predict(x.test, batch_size = 50) %>% .[,1]

sm_sigmoid_rmsprop_mae = 0.248

#########################################################################
# SM / sigmoid / rmsprop / mse

model <- keras_model_sequential() %>%
  layer_dense(units = 1, activation = 'sigmoid') %>%
  layer_dense(units = 5, activation = 'sigmoid')

model %>% compile(
  optimizer = 'rmsprop',
  loss = 'mae')
model%>% fit(x.train, y.train, epochs = 10, batch_size = 50)

pred_out <- model %>% predict(x.test, batch_size = 50) %>% .[,1]

sm_sigmoid_rmsprop_mse = 0.248

#########################################################################
# SM / sigmoid / rmsprop / mape

model <- keras_model_sequential() %>%
  layer_dense(units = 1, activation = 'sigmoid') %>%
  layer_dense(units = 5, activation = 'sigmoid')

model %>% compile(
  optimizer = 'rmsprop',
  loss = 'mape')
model%>% fit(x.train, y.train, epochs = 10, batch_size = 50)

pred_out <- model %>% predict(x.test, batch_size = 50) %>% .[,1]

sm_sigmoid_rmsprop_mape = 1


#####################################################################
############################### RNN #################################
#####################################################################

#########################################################################
# RNN / sigmoid / adam / mse


model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 1000, output_dim = 50) %>%
  layer_simple_rnn(units = 50, return_sequences = TRUE) %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "adam",
  loss = "mse",
)

model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)

rnn_sigmoid_adam_mse = 0.084

#########################################################################
# RNN / sigmoid / adam / mae

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 1000, output_dim = 50) %>%
  layer_simple_rnn(units = 50, return_sequences = TRUE) %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "adam",
  loss = "mae",
)

model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)

rnn_sigmoid_adam_mae = 0.25

#########################################################################
# RNN / sigmoid / adam / mape

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 1000, output_dim = 50) %>%
  layer_simple_rnn(units = 50, return_sequences = TRUE) %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "adam",
  loss = "mape",
)

model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)

rnn_sigmoid_adam_mape = 1

#########################################################################
# RNN / sigmoid / rmsprop / mse

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 1000, output_dim = 50) %>%
  layer_simple_rnn(units = 50, return_sequences = TRUE) %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
)

model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)

rnn_sigmoid_rmsprop_mse = 0.084

#########################################################################
# RNN / sigmoid / rmsprop / mae

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 1000, output_dim = 50) %>%
  layer_simple_rnn(units = 50, return_sequences = TRUE) %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "mae",
)

model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)

rnn_sigmoid_rmsprop_mae = 0.251

#########################################################################
# RNN / sigmoid / rmsprop / mape

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 1000, output_dim = 50) %>%
  layer_simple_rnn(units = 50, return_sequences = TRUE) %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "mape",
)

model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)

rnn_sigmoid_rmsprop_mape = 1

#########################################################################
# RNN / relu / adam / mse


model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 1000, output_dim = 50) %>%
  layer_simple_rnn(units = 50, return_sequences = TRUE) %>%
  layer_dense(units = 1, activation = "relu")
model %>% compile(
  optimizer = "adam",
  loss = "mse",
)

model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)

rnn_relu_adam_mse = 0.086

#########################################################################
# RNN / relu / adam / mae

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 1000, output_dim = 50) %>%
  layer_simple_rnn(units = 50, return_sequences = TRUE) %>%
  layer_dense(units = 1, activation = "relu")
model %>% compile(
  optimizer = "adam",
  loss = "mae",
)

model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)

rnn_relu_adam_mae = 0.252

#########################################################################
# RNN / relu / adam / mape

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 1000, output_dim = 50) %>%
  layer_simple_rnn(units = 50, return_sequences = TRUE) %>%
  layer_dense(units = 1, activation = "relu")
model %>% compile(
  optimizer = "adam",
  loss = "mape",
)

model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)

rnn_relu_adam_mape = 1

#########################################################################
# RNN / relu / rmsprop / mse

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 1000, output_dim = 50) %>%
  layer_simple_rnn(units = 50, return_sequences = TRUE) %>%
  layer_dense(units = 1, activation = "relu")
model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
)

model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)

rnn_relu_rmsprop_mse = 0.085

#########################################################################
# RNN / relu / rmsprop / mae

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 1000, output_dim = 50) %>%
  layer_simple_rnn(units = 50, return_sequences = TRUE) %>%
  layer_dense(units = 1, activation = "relu")
model %>% compile(
  optimizer = "rmsprop",
  loss = "mae",
)

model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)

rnn_relu_rmsprop_mae = 0.253

#########################################################################
# RNN / relu / rmsprop / mape

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 1000, output_dim = 50) %>%
  layer_simple_rnn(units = 50, return_sequences = TRUE) %>%
  layer_dense(units = 1, activation = "relu")
model %>% compile(
  optimizer = "rmsprop",
  loss = "mape",
)

model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)

rnn_relu_rmsprop_mape = 1


#########################################################################
#########################################################################
############################## TABL #####################################
#########################################################################
#########################################################################

answer <- data.frame(c("LSTM", "LSTM", "SM", "SM", "SM", "SM", "RNN", "RNN", "RNN", "RNN"),
                     c("adam", "rmsprop", "adam", "adam", "rmsprop", "rmsprop", "adam", "adam", "rmsprop", "rmsprop"),
                     c("none", "none","relu", "sigmoid","relu", "sigmoid","relu", "sigmoid","relu", "sigmoid"),
                     c(lstm_adam_mae, lstm_rmsprop_mae, 
                       sm_relu_adam_mae, sm_sigmoid_adam_mae, sm_relu_rmsprop_mae, sm_sigmoid_rmsprop_mae,
                       rnn_relu_adam_mae, rnn_sigmoid_adam_mae, rnn_relu_rmsprop_mae, rnn_sigmoid_rmsprop_mae),
                     c(lstm_adam_mse, lstm_rmsprop_mse, 
                       sm_relu_adam_mse, sm_sigmoid_adam_mse, sm_relu_rmsprop_mse, sm_sigmoid_rmsprop_mse,
                       rnn_relu_adam_mse, rnn_sigmoid_adam_mse, rnn_relu_rmsprop_mse, rnn_sigmoid_rmsprop_mse),
                     c(lstm_adam_mape, lstm_rmsprop_mape, 
                       sm_relu_adam_mape, sm_sigmoid_adam_mape, sm_relu_rmsprop_mape, sm_sigmoid_rmsprop_mape,
                       rnn_relu_adam_mape, rnn_sigmoid_adam_mape, rnn_relu_rmsprop_mape, rnn_sigmoid_rmsprop_mape)
                     )

colnames(answer) <- c("model", "optimizer","activation", "mae","mse","mape")

###########
lstm_adam_mse = 0.171
lstm_adam_mae = 0.34
lstm_adam_mape = 1
lstm_rmsprop_mse = 0.17
lstm_rmsprop_mae = 0.34
lstm_rmsprop_mape = 1
sm_relu_adam_mse = 0.082
sm_relu_adam_mae = 0.248
sm_relu_adam_mape = 1
sm_relu_rmsprop_mae = 0.248
sm_relu_rmsprop_mse = 0.248
sm_relu_rmsprop_mape = 1
sm_sigmoid_adam_mse = 0.082
sm_sigmoid_adam_mae = 0.248
sm_sigmoid_adam_mape = 1
sm_sigmoid_rmsprop_mae = 0.248
sm_sigmoid_rmsprop_mse = 0.248
sm_sigmoid_rmsprop_mape = 1
rnn_relu_adam_mse = 0.086
rnn_relu_adam_mae = 0.252
rnn_relu_adam_mape = 1
rnn_relu_rmsprop_mse = 0.085
rnn_relu_rmsprop_mae = 0.253
rnn_relu_rmsprop_mape = 1
rnn_sigmoid_adam_mse = 0.084
rnn_sigmoid_adam_mae = 0.25
rnn_sigmoid_adam_mape = 1
rnn_sigmoid_rmsprop_mse = 0.084
rnn_sigmoid_rmsprop_mae = 0.251
rnn_sigmoid_rmsprop_mape = 1
###########


# Лучшая модель -  LSTM / adam / mse
model <- keras_model_sequential()  %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

model %>%
  compile(loss = 'mse', optimizer = 'rmsprop')
model %>% fit(x.train, y.train, epochs = 10, batch_size = batch.size)

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]

plot_ly(myts, x = ~index, y = ~price, type = "scatter", mode = "markers", color = ~vol) %>%
  add_trace(y = c(rep(NA, 1000), pred_out), x = myts$index, name = "LSTM prediction", color = 'black')
plot(y.test - pred_out, type = 'line')
plot(x = y.test, y = pred_out)

