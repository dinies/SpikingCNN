#! /bin/bash

rm ../logs/log.csv
rm ../weights/weight_*
rm ../datasets/ClassifierSet/TestingData.csv
rm ../datasets/ClassifierSet/TrainingData.csv

rm ../datasets/LearningSet/FaceDone/*
rm ../datasets/LearningSet/MotorDone/*
rm ../datasets/LearningSet/FaceToDo/*
rm ../datasets/LearningSet/MotorToDo/*
cp ../datasets/LearningSet/Motor/* ../datasets/LearningSet/MotorToDo/
cp ../datasets/LearningSet/Face/* ../datasets/LearningSet/FaceToDo/

rm ../datasets/TrainingSet/FaceDone/*
rm ../datasets/TrainingSet/MotorDone/*
rm ../datasets/TrainingSet/FaceToDo/*
rm ../datasets/TrainingSet/MotorToDo/*
cp ../datasets/TrainingSet/Motor/* ../datasets/TrainingSet/MotorToDo/
cp ../datasets/TrainingSet/Face/* ../datasets/TrainingSet/FaceToDo/


rm ../datasets/TestingSet/FaceDone/*
rm ../datasets/TestingSet/MotorDone/*
rm ../datasets/TestingSet/FaceToDo/*
rm ../datasets/TestingSet/MotorToDo/*
cp ../datasets/TestingSet/Motor/* ../datasets/TestingSet/MotorToDo/
cp ../datasets/TestingSet/Face/* ../datasets/TestingSet/FaceToDo/


