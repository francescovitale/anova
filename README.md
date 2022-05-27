# Python implementation of one- and two-way ANOVA for one- and two-factor full-factorial design of experiments

This script allows performing one- and two-way ANOVA with a fully generalized structure. It requires input csv data (with cells separated through the ";" character) contained in the "Data" directory; the header of the csv file must encode the levels of the factors of the design of experiment. If the experiment has two factors, each sample group pair label must be specified as two values linked with an underscore character "_".


To run the script, at least two input arguments are required: the first lists the factor(s) (if two factors are analyzed, they must be separated by an underscore character "_").

Following is a sample script execution:

testbed_analysis WS_NC Accuracy.csv
