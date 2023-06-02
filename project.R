setwd("/home/stefanljubovic/Documents/RVPII/datasets")

install.packages("sparklyr")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("DBI")

library(sparklyr)
library(dplyr)
library(ggplot2)
library(DBI)

sc <- spark_connect(master = "local", version = "3.3.2")

spark_get_java()


parking_violations_raw <- spark_read_csv(sc, 
                               name = "parking_violations", 
                               path = ".", 
                               header = T, 
                               memory = T)


parking_violations <- parking_violations_raw %>%
  select(-NTA, -BIN, -BBL, -Census_Tract, -Community_Council, -Longitude, -Latitude, -Double_Parking_Violation, -Hydrant_Violation,
         -No_Standing_or_Stopping_Violation, -Violation_Description, -Violation_Post_Code,-Feet_From_Curb,-Meter_Number) %>%
  filter(!is.na(Vehicle_Make) & !
           is.na(Vehicle_Body_Type))

parking_violations
library(DBI)
dbGetQuery(sc, "select count(*) from parking_violations")

parking_violations.split <- sdf_random_split(parking_violations, training = 0.75, test = 0.25, seed = 5)

parking_violations.training <- parking_violations.split$training
parking_violations.test <- parking_violations.split$test



