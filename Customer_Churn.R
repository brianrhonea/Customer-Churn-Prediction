rm(list=ls())
cat("\014")

library(ggplot2)
library(rpart)
library(caret)
library(rpart.plot)
library(dplyr)
library(ROSE)
library(corrplot)
library(stats)
library(cluster)
library(smotefamily)
library(ROCit)

train<- read.csv("train.csv", stringsAsFactors = F)
test<-read.csv("test.csv", stringsAsFactors = F)


set.seed(123) # for reproducible results
train.split <- sample(1:nrow(train), nrow(train)*(2/3))
churned.train <- train[train.split,]   # 8333 rows
churned.test <- train[-train.split,] #4167 
#ROSE
# balanced data set with both over and under sampling
balanced_data <- ovun.sample(churned ~., data=churned.train,
                              seed=1, method="over")$data
table(balanced_data$churned)

#feature selection 

#Reform data
balanced_data$signup_date<- as.Date(balanced_data$signup_date, format = "%M/%D/%Y")
churned.test$signup_date<- as.Date(churned.test$signup_date, format = "%M/%D/%Y")
balanced_data$location<-as.factor(balanced_data$location)
balanced_data$payment_plan<-as.factor(balanced_data$payment_plan)
balanced_data$location<-as.factor(balanced_data$location)
balanced_data$churned<-as.factor(balanced_data$churned)



###supervised decision tree#####

#first create categorical varaibles as factors
# cp = 0: minimum improvement in complexity parameter for splitting
fit.big <- rpart(churned ~ ., 
                 data=balanced_data,
                 control=rpart.control(xval=10, minsplit=2, cp=0))

# object fit.big$frame has a row for each node in the tree
nrow(fit.big$frame) # 247 nodes

plot(fit.big)

# extract the vector of predicted values for churn for every row
churn.pred <- predict(fit.big, balanced_data, type="class")
# extract the actual value of churn for every row
churn.actual <- balanced_data$churned
# confusion matrix for training data cellco.train 
confusionMatrix(table(churn.pred,churn.actual), positive='1')

# confusion matrix for hold out data in cellco.test
churn.pred <- predict(fit.big, churned.test, type="class")
churn.actual <- churned.test$churned
confusionMatrix(table(churn.pred,churn.actual), positive='1')


bestcp <- fit.big$cptable[which.min(fit.big$cptable[,"xerror"]),"CP"]
bestcp   

# The lowest error occurs at CP = bestcp
# We can use this for post-pruning
fit.post <- prune.rpart(fit.big, cp=bestcp)
nrow(fit.post$frame)  

# plot the tree - NOTE. same as pre-pruned tree fit.small2 
prp(fit.post, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    main="Post-prune Tree with best cp")  

# compute the confusion matrices and accuracy 
confusionMatrix(table(predict(fit.post, balanced_data, type="class"),
                      balanced_data$churned), positive='1')

confusionMatrix(table(predict(fit.post, churned.test, type="class"),
                      churned.test$churned), positive='1')



################  Principal Component Analysis################  
### perform PCA for all numerical variables in the training data ###
#creating PC with rows calories:cups. Omitting all not needed columns
#if you haven't applied na.omit can add it here

pca_out_1 <- prcomp(balanced_data[, -c(1,3,4,5,7,8,9,20)])

# summary of PCs
summary(pca_out_1)

# let's look at the weights of these components, stored in "rotation" of the output
pca_out_1$rotation

# get principal component scores from "x" of the output 
scores<-pca_out_1$x

# correlations are zero between different pcs, as each PC is orthogonal to another
cor(scores)

# scale before constructing pca
#standardized results using "scale." parameter
pca_out_2 <- prcomp(balanced_data[, -c(1,3,4,5,7,8,9,20)], scale.=T)
summary(pca_out_2)
pca_out_2$rotation


### lets visualize ###
library(ggplot2)

# scree plot: plot of the proportion of variance explained (PVE) by each PC
# first let us create vector of variances for each generated principal component
pca.var <- pca_out_2$sdev^2
# create proportion of variance explained by each principal component
pca.pve <- data.frame( pve = pca.var/sum(pca.var), component = c(1:12) )
# plot
#look for "elbow" which varies between persons views using base
plot(pca.pve$pve)
#look for "elbow" which varies between persons views using ggplot
g<- ggplot(pca.pve, aes(component, pve))
g + geom_point() + labs(title = "Scree Plot", x="Component", y="PVE")+ scale_x_continuous(breaks = seq(1,12, by= 1))

# plot the weight of the original variables in each PC
rot <- as.data.frame( pca_out_2$rotation ) #transforming data to dataframe to track loadings
rot$feature<-rownames(rot)

# ordered in decreasing loading (i.e., weight) for PC1
rot$feature <- factor(rot$feature, levels = rot$feature[order(rot$PC1, decreasing = T)])
#plot PC1
g <- ggplot(rot, aes(feature, PC1))
g + geom_bar(stat= "identity", position = "identity", fill = "green") + theme(axis.text.x = element_text(size = 10, angle = 45, hjust=1))


# ordered in decreasing loading for PC2
rot$feature2<-rownames(rot)
# Create a new dataframe sorted by PC2
sorted_rot <- rot[order(rot$PC2, decreasing = TRUE), ]

# Assign rownames based on the sorted order
sorted_rot$feature2 <- factor(rownames(sorted_rot), levels = rownames(sorted_rot))

# Assign sorted_rot back to rot if you want to replace the original dataframe
rot <- sorted_rot

#plot PC2 should show calories
g2<-ggplot(rot,aes(feature2, PC2))
g2 + geom_bar(stat="identity", position= "identity", fill= "blue") +theme(axis.text.x = element_text(size = 10, angle = 45, hjust=1))


################  PCA + Classification ################  
# append pca scores to train.df 
balanced_data<-cbind(balanced_data, pca_out_2$x)
# calculate pca scores for the testing data using predict()  
testScores<-as.data.frame( predict(pca_out_2, balanced_data[, -c(1,3,4,5,7,8,9,20)]))
test.df <- cbind(balanced_data, testScores)
rm(testScores)
# use glm() (general linear model) with family = "binomial" to fit a logistic 
logit.reg <- glm(churned ~ PC1 + PC2 + PC3 + PC4 + PC5, 
                 data = balanced_data, family = "binomial")

# use predict() with type = "response" to compute predicted probabilities. 
logitPredict <- predict(logit.reg, test.df, type = "response")
# we choose 0.5 as the cutoff here for 1 vs. 0 classes
logitPredictClass <- ifelse(logitPredict >0.5, 1, 0)

# Evaluate classifier performance on testing data
library(caret)
actual <- test.df$churned
predict<- logitPredictClass
confusionMatrix(table(predict, actual),positive = "1") #balanced - 0.9231, accuracy - 0.913   

#try after class check comparisons vs PC and x values and evaluate performance
logit.reg2 <- glm(churned ~ PC1 + PC2 , 
                  data = balanced_data, family = "binomial")
logitPredict2 <- predict(logit.reg2, test.df, type = "response")
logitPredictClass2 <- ifelse(logitPredict2 >0.5, 1, 0)
library(caret)
actual2 <- test.df$churned
predict2<- logitPredictClass2
confusionMatrix(table(predict2, actual2),positive = "1") #balanced - 0.8346, accuracy-0.8261
#using less PC values makes it less accurate.


