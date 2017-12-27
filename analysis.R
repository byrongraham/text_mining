library(tidyverse)
library(readxl)
library(tidytext)
library(tm)
library(caret)
library(gbm)
library(wordcloud)

grant_data<- read_excel("C:/Users/Byron Graham/Documents/EPSRC_NLP/Grants_database_v2.xls")
survey_data<- read_excel("C:/Users/Byron Graham/Documents/dataset27.xls")


grant_data2 <- grant_data %>% 
  select(GrantRefNumber, Summary)

survey_data2 <- survey_data %>% 
  select(GrantRefNumber, industryinvrecode) %>% 
  left_join(grant_data2, by = "GrantRefNumber")

text <- survey_data2$Summary

text_clean = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", text)
text_clean = gsub("@\\w+", "", text_clean)
text_clean = gsub("[[:punct:]]", "", text_clean)
text_clean = gsub("[[:digit:]]", "", text_clean)
text_clean = gsub("http\\w+", "", text_clean)

text_clean = removePunctuation(text_clean)
#character vector and stem
#text_clean <- unlist(strsplit(text_clean, split = ' '))

#text_clean = stemDocument(text_clean)
#text_clean = stemCompletion(text_clean)

text_corpus = Corpus(VectorSource(text_clean))
text_corpus = tm_map(text_corpus, tolower)
text_corpus = tm_map(text_corpus, removeWords, c(stopwords("english")))
text_corpus = tm_map(text_corpus, stripWhitespace)
#text_corpus = tm_map(text_corpus, PlainTextDocument)
text_corpus = tm_map(text_corpus, stemDocument, language = "english")  
#check some output
#text_corpus[[10]][1]

#tdm = TermDocumentMatrix(text_corpus)
dtm<-DocumentTermMatrix(text_corpus)
m<-as.matrix(dtm)
#m = as.matrix(tdm)

frame<-data.frame(m)

# inspect most popular words
findFreqTerms(dtm, lowfreq=30)


#produce some wordclouds
freq = data.frame(sort(colSums(as.matrix(dtm)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], max.words=50, colors=brewer.pal(1, "Dark2"))


#remove sparse terms
#wf =rowSums(m)
#m1 = m[wf>quantile(wf,probs=0.9),]



#remove all 0 columns
#m1 = m1[,colSums(m1)!=0]






#remove sparse terms
dtm2 <- removeSparseTerms(dtm, 0.98)



dtm_m <- as.matrix(dtm2)
model_data<-data.frame(dtm_m)



#combine back in the industry involvement
model_data$ind_inv <- as.factor(survey_data2$industryinvrecode)

#build a classifier
sum(is.na(model_data))
#model_data<-complete.cases(model_data)
model_data<-na.omit(model_data)


#split the data
index <- createDataPartition(model_data$ind_inv, p=0.8, list=FALSE)
trainSet <- model_data[index,]
testSet<- model_data[-index,]


#feature selection in caret-recursive feature elinimation


#model building
#names(getModelInfo())

predictors <- names(select(model_data, -ind_inv))

outcome <- names(select(model_data, ind_inv))


model_gbm<-train(model_data[,predictors], model_data[,outcome], method='gbm')

####tuning parameters
fitControl<-trainControl(
  method='repeatedcv',
  number=5,
  repeats=1)


#######custom tune grid
grid <- expand.grid(n.trees=c(10,20,50,100,500,1000), 
                    shrinkage=c(0.01,0.05,0.1,0.5), 
                    n.minobsinnode=c(3,5,10), 
                    interaction.depth=c(1,5,10))



model_glm<-train(trainSet[,predictors],
                 trainSet[,outcome],
                 method='glm',
                 trControl=fitControl)


model_gbm<-train(trainSet[,predictors],
                 trainSet[,outcome],
                 method='gbm',
                 verbose=FALSE,
                 trControl=fitControl,
                 tuneGrid = grid)


model_rf<-train(trainSet[,predictors],
                 trainSet[,outcome],
                 method='rf',
                 verbose=FALSE,
                 trControl=fitControl)



#summarise the model
print(model_glm)
print(model_gbm)
print(model_rf)
plot(model_gbm)
varImp(model_gbm)

plot(varImp(object=model_gbm), main="GBM - Variable Importance")



#predictions glm
predictionsglm<-predict.train(object=model_glm,
                           testSet[,predictors],
                           type="raw")

table(predictionsglm)

confusionMatrix(predictionsglm, testSet[,outcome])


#predictions gbm
predictionsgbm<-predict.train(object=model_gbm,
                           testSet[,predictors],
                           type="raw")

table(predictions)

confusionMatrix(predictions, testSet[,outcome])

#predictions rf
predictionsrf<-predict.train(object=model_rf,
                           testSet[,predictors],
                           type="raw")

table(predictionsrf)

confusionMatrix(predictionsrf, testSet[,outcome])


# Print the dimensions of tweets_m
dim(tdm_m)

#view matrix

tdm_m[1:5, 1:5]

#document term matrix might be better - alos convert to data frame for processing




#########convert to a data frame
# convert the sparse term-document matrix to a standard data frame
mydata.df <- as.data.frame(inspect(dtm2))

data<- as.data.frame(dtm)

# inspect dimensions of the data frame
nrow(mydata.df)
ncol(mydata.df)

#######cluster analysis
mydata.df.scale <- scale(mydata.df)
d <- dist(mydata.df.scale, method = "euclidean") # distance matrix
fit <- hclust(d, method="ward")
plot(fit) # display dendogram?

groups <- cutree(fit, k=5) # cut tree into 5 clusters
# draw dendogram with red borders around the 5 clusters
rect.hclust(fit, k=5, border="red")



###########use tf-idf
data(crude)
dtm <- DocumentTermMatrix(crude, control = list(weighting = weightTfIdf))

data(crude)
dtm <- DocumentTermMatrix(crude,
                          control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE),
                                         stopwords = TRUE))


#more detail on predictive modelling:
https://rstudio-pubs-static.s3.amazonaws.com/132792_864e3813b0ec47cb95c7e1e2e2ad83e7.html

#good tutorial example
http://web.letras.up.pt/bhsmaia/EDV/apresentacoes/Bradzil_Classif_withTM.pdf


#use tfidf
review_dtm_tfidf <- DocumentTermMatrix(review_corpus, control = list(weighting = weightTfIdf))
review_dtm_tfidf = removeSparseTerms(review_dtm_tfidf, 0.95)
review_dtm_tfidf


freq = data.frame(sort(colSums(as.matrix(review_dtm_tfidf)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], max.words=100, colors=brewer.pal(1, "Dark2"))





#neural net using MXNet
# Installation - Windows
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
library(mxnet)



cran <- getOption("repos")
cran["dmlc"] <- "https://s3-us-west-2.amazonaws.com/apache-mxnet/R/CRAN/"
options(repos = cran)
install.packages("mxnet",dependencies = T)
library(mxnet)
#https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/understanding-deep-learning-parameter-tuning-with-mxnet-h2o-package-in-r/tutorial/

#convert data to matrix and numeric
train_m<-as.matrix(select(trainSet, -ind_inv))
target<-as.numeric(trainSet$ind_inv)


test_m<-as.matrix(select(testSet, -ind_inv))
target_test<-as.numeric(testSet$ind_inv)


#multilayered perceptron model
mx.set.seed(1)

mlpmodel <-mx.mlp(data=train_m,
                  label=target,
                  hidden_node=10,
                  out_node = 2,
                  out_activation="softmax",
                  num.round=100,
                  array.batch.size=20,
                  learning.rate=0.03,
                  eval.metric=mx.metric.accuracy,
                  eval.data=list(data=test_m, label=target_test))
mlpmodel

summary(mlpmodel)

#computation graph
graph.viz(mlpmodel$symbol)

#create predictions
preds = predict(mlpmodel, test_m)
pred_label = max.col(t(preds))-1
table(pred_label, target_test)

https://github.com/apache/incubator-mxnet/issues/1546






mlpmodel <-mx.mlp(data=train_m,
                  label=target,
                  hidden_node=10,
                  out_node = 2,
                  out_activation="softmax",
                  num.round=100,
                  array.batch.size=20,
                  learning.rate=0.03,
                  eval.metric=mx.metric.accuracy)















