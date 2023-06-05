setwd("/home/stefanljubovic/Documents/RVPII/datasets")

library(sparklyr)
library(dplyr)
library(ggplot2)
library(dbplot)
library(DBI)
library(caret)

sc <- spark_connect(master = "local", version = "3.3.2")

spark_get_java()

parking_violations_raw <- spark_read_csv(sc, 
                               name = "parking_violations", 
                               path = ".", 
                               header = T, 
                               memory = T)

parking_violations <- parking_violations_raw %>%
  select(Vehicle_Make, Registration_State, Violation_In_Front_Of_Or_Opposite, Plate_Type,Violation_Code
         ,Vehicle_Year, Violation_Location, Law_Section) %>%
  mutate(Vehicle_Year = as.double(Vehicle_Year),
         Violation_Code = as.double(Violation_Code)) %>%
  filter(!is.na(Vehicle_Make) & 
           !is.na(Registration_State) &
           !is.na(Violation_In_Front_Of_Or_Opposite) &
           !is.na(Plate_Type) &
           !is.na(Violation_Code) &
           !regexp_replace(Registration_State, "[0-9]", "") != Registration_State &
           !is.na(Violation_Location) &
           !is.na(Law_Section) &
           !is.na(Vehicle_Year) &
           Vehicle_Year != 0 )




parking_violations <- parking_violations %>% 
  mutate(Year_Check = case_when(
    Vehicle_Year > 2010 ~ "old",
    TRUE ~ "new"))


formula <- Year_Check ~ 
  Violation_Code +
  Vehicle_Year +
  Violation_In_Front_Of_Or_Opposite +
  Violation_Location +
  Law_Section

parking_violations.split <- sdf_random_split(parking_violations, training = 0.8, test = 0.2, seed = 6)
parking_violations.training <- parking_violations.split$training
parking_violations.test <- parking_violations.split$test

it <- c(1, 2, 5, 8, 10, 20, 50)
tacnost <-numeric(length(it))

for (i in seq_along(it)) {
  counter <- counter + 1
  logreg <- ml_logistic_regression(
    x = parking_violations.training,
    formula = formula,
    family = "binomial",
    max_iter = it[i],
    threshold = 0.5
  )
  logreg.perf <- ml_evaluate(logreg,parking_violations.test)
  tacnost[i] <- logreg.perf$accuracy()
}


##### Prikaz preciznosti na grafiku

data.frame(i=it,t=tacnost) %>%
  ggplot(aes(i,t)) +
  geom_line() +
  scale_x_continuous(breaks=it) +
  labs(x="Broj iteracija", y="Tacnost")




k <- 4

# Initialize accuracy vectors
accuracies_log_reg <- c()
accuracies_svm <- c()
accuracies_knn <- c()

# Create partition sizes
partition_sizes <- rep(1/k, times = k)
fold_names <- paste0("fold", as.character(1:k))
names(partition_sizes) <- fold_names

# Split the dataset into partitions
violations_partitions <- parking_violations %>%
  sdf_random_split(weights = partition_sizes, seed = 86)

# Perform cross-validation
for (i in 1:k) {
  
  # Get training and test data
  training <- sdf_bind_rows(violations_partitions[-i])
  test <- violations_partitions[[i]]
  
  # Train and evaluate logistic regression
  log_reg <- ml_logistic_regression(x = training, formula = formula, family = "binomial", max_iter = 20, threshold = 0.5)
  evaluate_log_reg <- ml_evaluate(log_reg, test)
  accuracies_log_reg[i] <- evaluate_log_reg$accuracy()
  
  # Train and evaluate SVM
  svm <- ml_linear_svc(x = training, formula = formula, max_iter = 20, standardization = TRUE)
  evaluate_svm <- ml_evaluate(svm, test)
  accuracies_svm[i] <- evaluate_svm$Accuracy
  
  # Train and evaluate k-nearest neighbors classifier
  knn <- ml_knn_classifier(x = training, formula = formula, k = 5)
  evaluate_knn <- ml_evaluate(knn, test)
  accuracies_knn[i] <- evaluate_knn$Accuracy
  
}

# Print the mean accuracy for each model
print(paste("Logistic Regression Mean Accuracy:", mean(accuracies_log_reg)))
print(paste("SVM Mean Accuracy:", mean(accuracies_svm)))
print(paste("K-Nearest Neighbors Classifier Mean Accuracy:", mean(accuracies_knn)))


#Clusterisation

violations.clusters <- parking_violations %>% 
  filter(Vehicle_Year > 1960)

clust <- ml_kmeans(violations.clusters, ~ Vehicle_Year + Violation_Code, k=3, max_iter = 10, init_mode = "random")

clust$model$summary$cluster()

ml_evaluate(clust,parking_violations %>% select(Vehicle_Year,Violation_Code))

cluster.centers.df <- data.frame(
  year = clust$centers$Vehicle_Year,
  code = clust$centers$Violation_Code
)


#violations.clusters %>%
  #dbplot_raster(Vehicle_Year,Violation_Code,resolution=20) +
  #geom_point(data=clust$centers,
             #mapping=aes(x=Violation_Code,y=Vehicle_Year),color="yellow",size=5,pch="x")
cluster.values <- clust$model$summary$cluster() %>% collect()
violations.clusters<- violations.clusters %>% collect()
violations.clusters$clust <- as.factor(cluster.values$prediction)

ggplot(data = violations.clusters,
       aes(x = Vehicle_Year, y = Violation_Code, colour = "red")) +
  geom_jitter(size = 2) +
  geom_point(data = clust.centers.df,
             aes(x = year, y = code),
             color = "brown",
             size = 4,
             shape = 15)



