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

accuracies_log_reg <- c()
accuracies_svm <- c()
accuracies_dec_tree <- c()

violations_partitions <- vector("list", length = k)


parking_violations.split <- sdf_random_split(parking_violations, first = 0.5, second = 0.5, seed = 6)
parking_violations.first <- sdf_random_split(parking_violations.split$first, first = 0.5, second = 0.5, seed = 6)
parking_violations.second <- sdf_random_split(parking_violations.split$second, first = 0.5, second = 0.5, seed = 6)
violations_partitions[[1]] <- parking_violations.first$first
violations_partitions[[2]] <- parking_violations.first$second
violations_partitions[[3]] <- parking_violations.second$first
violations_partitions[[4]] <- parking_violations.second$second

for (i in 1:k) {
  
  training <- sdf_bind_rows(violations_partitions[-i])
  test <- violations_partitions[[i]]
  
  log_reg <- ml_logistic_regression(x = training, formula = formula, family = "binomial", max_iter = 20, threshold = 0.5)
  evaluate_log_reg <- ml_evaluate(log_reg, test)
  accuracies_log_reg[i] <- evaluate_log_reg$accuracy()
  
  svm <- ml_linear_svc(x = training, formula = formula, max_iter = 20, standardization = TRUE)
  evaluate_svm <- ml_evaluate(svm, test)
  accuracies_svm[i] <- evaluate_svm$Accuracy
  
  dec_tree <- ml_decision_tree_classifier(x = training, formula = formula, max_depth = 5, min_instances_per_node = 1000, impurity = "gini")
  evaluate_dec_tree <- ml_evaluate(dec_tree, test)
  accuracies_dec_tree[i] <- evaluate_dec_tree$Accuracy
  
}

print(paste("Logistic Regression Mean Accuracy:", mean(accuracies_log_reg)))
print(paste("SVM Mean Accuracy:", mean(accuracies_svm)))
print(paste("Decision Tree Classifier Mean Accuracy:", mean(accuracies_dec_tree)))

#Clusterisation1

parking.clusters <- parking_violations %>% 
  filter(Vehicle_Year >= 2000 & Vehicle_Year<=2005)  %>%
 group_by(Violation_Code, Vehicle_Year) %>%
  summarise(Count = n())

year.code <- ml_kmeans(
  parking.clusters,
  ~ Vehicle_Year + Violation_Code, 
  k = 5, 
  max_iter = 10, 
  init_mode = "k-means||")

year.code
year.code$model$summary$cluster()

ml_evaluate(year.code, parking_violations %>% select(Vehicle_Year , Violation_Code))

cluster.values <- year.code$model$summary$cluster() %>% collect()
parking.clusters <- parking.clusters %>% collect()
parking.clusters$clust <- as.factor(cluster.values$prediction)

cluster.centers.df <- data.frame(
  year = year.code$centers$Vehicle_Year,
  code = year.code$centers$Violation_Code
)

ggplot(data = parking.clusters,
       aes(x = Violation_Code, y = Vehicle_Year, colour = clust)) +
  geom_jitter(size = 2) +
  geom_point(data = cluster.centers.df,
             aes(x = code, y = year),
             color = "brown",
             size = 4,
             shape = 15)


#Clusterisation2


parking.clusters2 <- parking_violations %>%
  group_by(Registration_State,Violation_Code) %>%
  summarise(People_In_State = n(), .groups = "keep") %>%
  filter(People_In_State  > 50000)


year.code2 <- ml_kmeans(
  parking.clusters2,
  ~ People_In_State + Violation_Code, 
  k = 3, 
  max_iter = 10, 
  init_mode = "k-means||")

year.code2
year.code2$model$summary$cluster()

ml_evaluate(year.code2, parking.clusters2 %>% select(People_In_State , Violation_Code))

cluster.values2 <- year.code2$model$summary$cluster() %>% collect()
parking.clusters2 <- parking.clusters2 %>% collect()
parking.clusters2$clust <- as.factor(cluster.values2$prediction)

cluster.centers.df2 <- data.frame(
  location = year.code2$centers$People_In_State,
  code = year.code2$centers$Violation_Code
)

ggplot(data = parking.clusters2,
       aes(x = People_In_State, y = Violation_Code, colour = clust)) +
  geom_jitter(size = 2) +
  geom_point(data = cluster.centers.df2,
             aes(x = location, y = code),
             color = "brown",
             size = 4,
             shape = 15) +
  scale_x_continuous(labels = scales::comma)








