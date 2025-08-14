
library("ggplot2")
library("shapviz")
library("xgboost")
library("caret")
library("dplyr")
library("metafor")
library("metagear")
library("DT")
library("ggrepel")
library("rcartocolor")
library("tidyr")
library("metaforest")
library("dismo")
library("randomForest")
library("kernlab")
library("caret")
library("readxl")




Che_Tianhao <- read_excel("E:/Desktop/Machine_learning_builds_data.xlsx")


Cmeta <- na.omit(Che_Tianhao)


C <- sample(2,nrow(Cmeta),replace = TRUE,prob = c(0.7,0.3))
C_train <- Cmeta[C==1,]
C_test <- Cmeta[C==2,]


moderators <- c("Lon","Lat","Continent","Area",
                "Tmin","Tmax","TD","MAP","CZ",
                "VC","NDVI","EVI","LST","ET",
                "ST","Sand","Clay","pH","BD",
                "DEM","Slope","SD",
                "PD","HDI","GDP")


train <- xgb.DMatrix(data.matrix(C_train[moderators]),
                      label=C_train$SOC)
test <- xgb.DMatrix(data.matrix(C_test[moderators]),
                     label=C_test$SOC)

XGB <- xgb.train(params = list(learning_rate=0.1),
                 data = train,
                 eta=0.2,
                 nrounds = 40)


C_train$XGB <- predict(XGB,train)
C_test$XGB <- predict(XGB,test)


RMSEFun=function(sim,obs){
  round(sqrt(mean((sim-obs)^2,)),3)
}
R2Fun=function(sim,obs){
  round(summary(lm(sim~+obs))$r.squared,4)
}         
select=dplyr::select
C_test%>%select(SOC,XGB)%>%gather(key="ML",value = 'value',-SOC)%>%
  group_by(ML)%>%summarise(RMSE=RMSEFun(value,SOC),
                           R2=R2Fun(value,SOC))
C_train%>%select(SOC,XGB)%>%gather(key="ML",value = 'value',-SOC)%>%
  group_by(ML)%>%summarise(RMSE=RMSEFun(value,SOC),
                           R2=R2Fun(value,SOC))


C_train%>%ggplot(aes(x=SOC,y=XGB))+
  geom_point()+
  scale_x_continuous(limits = c(-2,4))+
  scale_y_continuous(limits = c(-2,4))+
  coord_fixed(ratio = 1)+
  geom_abline(intercept = 0, slope = 1,size=0.5,linetype=2)
C_test%>%ggplot(aes(x=SOC,y=XGB))+
  geom_point()+
  scale_x_continuous(limits = c(-2,4))+
  scale_y_continuous(limits = c(-2,4))+
  coord_fixed(ratio = 1)+
  geom_abline(intercept = 0, slope = 1,size=0.5,linetype=2)


SSP126 <- read_excel("E:/Desktop/SSP126.xlsx")
SSP245 <- read_excel("E:/Desktop/SSP245.xlsx")
SSP585 <- read_excel("E:/Desktop/SSP585.xlsx")


SSP126_predict <- xgb.DMatrix(data.matrix(SSP126[moderators]),
                     label=SSP126$SOC)
SSP245_predict <- xgb.DMatrix(data.matrix(SSP245[moderators]),
                    label=SSP245$SOC)
SSP585_predict <- xgb.DMatrix(data.matrix(SSP585[moderators]),
                              label=SSP585$SOC)


SSP126$XGB <- predict(XGB,SSP126_predict)
SSP245$XGB <- predict(XGB,SSP245_predict)
SSP585$XGB <- predict(XGB,SSP585_predict)


write.csv(C_train,file = "C_train_SOC_XGB.csv",na="NA",row.names=FALSE)
write.csv(C_test,file = "C_test_SOC_XGB.csv",na="NA",row.names=FALSE)
write.csv(SSP126,file = "SSP126_SOC_XGB.csv",na="NA",row.names=FALSE)
write.csv(SSP245,file = "SSP245_SOC_XGB.csv",na="NA",row.names=FALSE)
write.csv(SSP585,file = "SSP585_SOC_XGB.csv",na="NA",row.names=FALSE)


shap <- shapviz(XGB,
               X_pred=data.matrix(Cmeta[moderators]),
               X=Cmeta)


sv_importance(shap, kind="beeswarm")+theme_bw()

sv_importance(shap,show_numbers = TRUE)+theme_bw()


+theme_classic()


sv_dependence(shap,
              "Tmax",
              alpha=0.5,
              size=1.5,
              color_var = NULL)+theme_bw()

sv_dependence(shap,
              v=c("CZ"))+theme_bw()


library(xgboost)
library(dplyr)


bootstrap_ci <- function(data, moderators, n_bootstrap = 100) {
  n <- nrow(data)
  predictions <- matrix(NA, nrow = n, ncol = n_bootstrap)
  
  for (i in 1:n_bootstrap) {
 
    bootstrap_sample <- data[sample(n, replace = TRUE), ]
    
  
    train_bootstrap <- xgb.DMatrix(data.matrix(bootstrap_sample[moderators]), label = bootstrap_sample$SOC)
    model_bootstrap <- xgb.train(params = list(learning_rate = 0.1), data = train_bootstrap, eta = 0.2, nrounds = 40)
    
  
    predictions[, i] <- predict(model_bootstrap, data.matrix(data[moderators]))
  }
  

  ci_lower <- apply(predictions, 1, quantile, probs = 0.025)
  ci_upper <- apply(predictions, 1, quantile, probs = 0.975)
  
  return(data.frame(prediction = rowMeans(predictions), ci_lower = ci_lower, ci_upper = ci_upper))
}


SSP126_ci <- bootstrap_ci(SSP126, moderators)
SSP126$prediction <- SSP126_ci$prediction
SSP126$ci_lower <- SSP126_ci$ci_lower
SSP126$ci_upper <- SSP126_ci$ci_upper


SSP245_ci <- bootstrap_ci(SSP245, moderators)
SSP245$prediction <- SSP245_ci$prediction
SSP245$ci_lower <- SSP245_ci$ci_lower
SSP245$ci_upper <- SSP245_ci$ci_upper


SSP585_ci <- bootstrap_ci(SSP585, moderators)
SSP585$prediction <- SSP585_ci$prediction
SSP585$ci_lower <- SSP585_ci$ci_lower
SSP585$ci_upper <- SSP585_ci$ci_upper


write.csv(SSP126, file = "SSP126_SOC_XGB_CI.csv", na = "NA", row.names = FALSE)
write.csv(SSP245, file = "SSP245_SOC_XGB_CI.csv", na = "NA", row.names = FALSE)
write.csv(SSP585, file = "SSP585_SOC_XGB_CI.csv", na = "NA", row.names = FALSE)