setwd("/home/stefanljubovic/Documents/RVPII/datasets")

library(sparklyr)
library(dplyr)
library(ggplot2)
library(DBI)
install.packages("stringr")
library(stringr)

sc <- spark_connect(master = "local", version = "3.3.2")

spark_get_java()


parking_violations_raw <- spark_read_csv(sc, 
                               name = "parking_violations", 
                               path = ".", 
                               header = T, 
                               memory = T)

parking_violations <- parking_violations_raw %>%
  select(Vehicle_Make, Vehicle_Body_Type, Registration_State, Violation_In_Front_Of_Or_Opposite, Plate_Type,Violation_Code
         ,Vehicle_Year, Violation_Location, Law_Section) %>%
  mutate(Vehicle_Year = as.double(Vehicle_Year),
         Violation_Code = as.double(Violation_Code)) %>%
  filter(!is.na(Vehicle_Make) & 
           !is.na(Vehicle_Body_Type) & 
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

parking_violations.split <- sdf_random_split(parking_violations, training = 0.75, test = 0.25, seed = 5)
parking_violations.training <- parking_violations.split$training
parking_violations.test <- parking_violations.split$test

iters <- c(3, 5, 10, 15, 20, 30, 50, 70)
model.accuracies <- c(length(iters))
counter <- 0

for (iter in iters) {
  counter <- counter + 1
  log.reg <- ml_logistic_regression(
    x = parking_violations,
    formula = formula,
    family = "binomial",
    max_iter = iter,
    threshold = 0.5
  )
  
  log.reg.perfs <- ml_evaluate(log.reg, parking_violations.test)
  model.accuracies[counter] <- log.reg.perfs$accuracy()
}


##### Prikaz preciznosti na grafiku

df <- data.frame(max_iterations = iters, accuracy = model.accuracies)
df

g <- ggplot(df, aes(max_iterations, accuracy)) +
  geom_line() +
  geom_point()

g



# Perform K-fold cross-validation
k <- 4

# Initialize accuracy vectors
accuracies_log_reg <- c()
accuracies_svm <- c()
accuracies_dec_tree <- c()

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
  
  # Train and evaluate linear regression
  log_reg <- ml_logistic_regression(x = training, formula = formula, family = "binomial", max_iter = 20, threshold = 0.5)
  evaluate_log_reg <- ml_evaluate(log_reg, test)
  accuracies_log_reg[i] <- evaluate_log_reg$accuracy()
  
  
  # Train and evaluate SVM
  svm <- ml_linear_svc(x = training, formula = formula, max_iter = 20, standardization = TRUE)
  evaluate_svm <- ml_evaluate(svm, test)
  accuracies_svm[i] <- evaluate_svm$Accuracy
  
  # Train and evaluate decision tree classifier
  dec_tree <- ml_decision_tree_classifier(x = training, formula = formula, max_depth = 5, min_instances_per_node = 1000, impurity = "gini")
  evaluate_dec_tree <- ml_evaluate(dec_tree, test)
  accuracies_dec_tree[i] <- evaluate_dec_tree$Accuracy
  
}

# Print the mean accuracy for each model
print(paste("Logistic Regression Mean Accuracy:", mean(accuracies_log_reg)))
print(paste("SVM Mean Accuracy:", mean(accuracies_svm)))
print(paste("Decision Tree Classifier Mean Accuracy:", mean(accuracies_dec_tree)))



