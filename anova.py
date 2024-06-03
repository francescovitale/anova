import pandas as pd
import numpy as np
import math
import random
import os
from random import randrange
from scipy import stats
import matplotlib.pyplot as plt
#import statsmodels.api as sm
import scikit_posthocs as sp
import pylab as py
import sys

data_dir = "Data/"

def load_data(input_data_files):

	data = {}

	for input_data_file in input_data_files:
		loaded_data = pd.read_csv(data_dir + input_data_file)
		data[input_data_file.split(".")[0]] = loaded_data 

	return data

def compute_model(experimental_data, factor_names):

	model = {}

	# get the number of factors 

	n_factors = len(factor_names)

	if n_factors == 1: # one-factor full-factorial...
		factor_levels = []
		factor_levels_cardinality = 0
		
		for column in experimental_data.columns.values:
			column_factor_level = column
			factor_levels.append(column_factor_level)
		
		factor_levels_cardinality = len(factor_levels)
		n_repetitions = len(experimental_data[experimental_data.columns.values[0]])
		
		grand_mean = 0
		for column in experimental_data:
			grand_mean = grand_mean + sum(experimental_data[column])
		grand_mean = grand_mean/(factor_levels_cardinality*n_repetitions)
		
		# compute factor effect
		factor_effects = {}
		
		for factor_level in factor_levels:
			effect_sum = 0
			for column in experimental_data:
				column_value = column
				if factor_level == column_value:
					effect_sum = effect_sum + experimental_data[column].sum()
			effect_sum = effect_sum/(n_repetitions) - grand_mean
			factor_effects[factor_names[0] + "_" + factor_level] = effect_sum
		
	if n_factors == 2: # two-factor full-factorial
		# get the factor levels
		factor_levels = {}
		factor_levels_cardinality = {}
		for factor_name in factor_names:
			factor_levels[factor_name] = []

		for column in experimental_data.columns.values:
			column_factor_levels = column.split("_")
			for idx,column_factor_level in enumerate(column_factor_levels):
					if column_factor_level not in factor_levels[factor_names[idx]]:
						factor_levels[factor_names[idx]].append(column_factor_level)
		for factor in factor_levels:
			factor_levels_cardinality[factor] = len(factor_levels[factor])
		n_repetitions = len(experimental_data[experimental_data.columns.values[0]])
		# compute grand mean
		grand_mean = 0
		for column in experimental_data:
			grand_mean = grand_mean + sum(experimental_data[column])

		total_levels_mul = 1
		for factor in factor_levels_cardinality:
			total_levels_mul = total_levels_mul * factor_levels_cardinality[factor]

		grand_mean = grand_mean/(total_levels_mul*n_repetitions)

		# compute individual factor effects
		factor_effects = {}

		for factor in factor_levels:
			
			for factor_level in factor_levels[factor]:
				effect_sum = 0
				for column in experimental_data:
					split_column_values = column.split("_")
					for split_column_value in split_column_values:
						if factor_level == split_column_value:
							#print("summing column " + column + " for factor level " + factor_level)
							effect_sum = effect_sum + experimental_data[column].sum()
				if factor == factor_names[0]:
					effect_sum = effect_sum/(factor_levels_cardinality[factor_names[1]]*n_repetitions) - grand_mean
				else:
					effect_sum = effect_sum/(factor_levels_cardinality[factor_names[0]]*n_repetitions) - grand_mean

				factor_effects[factor + "_" + factor_level] = effect_sum

		# compute combined factor effects

		for column in experimental_data:

			
			split_column_values = column.split("_")

			found_factor_levels = []

			combined_effect_name = ""
			for idx,split_column_value in enumerate(split_column_values):
				for factor in factor_levels:
					for factor_level in factor_levels[factor]:
						if split_column_value == factor_level:
							combined_effect_name = combined_effect_name + factor + "_" + factor_level
							
				if idx < len(split_column_values)-1:
					combined_effect_name = combined_effect_name + "_"
			
			tokens = combined_effect_name.split("_")
			factor_effects[combined_effect_name] = experimental_data[column].sum()/n_repetitions - factor_effects[tokens[0] + "_" + tokens[1]] - factor_effects[tokens[2] + "_" + tokens[3]] - grand_mean
			
	  
	
	model["grand_mean"] = grand_mean
	for factor_effect in factor_effects:
		model[factor_effect] = factor_effects[factor_effect]
	
				
	return model

def compute_residuals(experimental_data, model, factor_names):

	predicted_responses = []
	residuals = []

	n_factors = len(factor_names)
	
	if n_factors == 1: # one-factor full-factorial
		for column in experimental_data:
			level = column
			factor = factor_names[0] + "_" + level
			predicted_response = model["grand_mean"] + model[factor]
			predicted_responses.append(predicted_response)
			for point in experimental_data[column]:
				residuals.append(point-predicted_response)

	if n_factors == 2: # two-factor full-factorial
		for column in experimental_data:
			levels = column.split("_")
			first_factor = factor_names[0] + "_" + levels[0]
			second_factor = factor_names[1] + "_" + levels[1]

			predicted_response = model["grand_mean"]+model[first_factor]+model[second_factor]+model[first_factor + "_" + second_factor]
			
			predicted_responses.append(predicted_response)
			for point in experimental_data[column]:
				residuals.append(point-predicted_response)

	return residuals, predicted_responses

def check_normality(residuals):
	statistic, p_value = stats.shapiro(residuals)
	#print("The computed p-value using the normality test is equal to: " + str(p_value))
	if p_value>0.05:
		return True
	else:
		return False

def check_omoscedasticity(residuals, predicted_responses):
	statistic, p_value = stats.bartlett(residuals, predicted_responses)
	#print("The computed p-value using the Bartlett test is equal to: " + str(p_value))
	if p_value>0.05:
		return True
	else:
		return False

def compute_SS(residuals, model, experimental_data, factor_names):
	n_factors = len(factor_names)

	if n_factors == 1: # one-factor full-factorial
		SS = {}
		df = {}
		
		factor_levels = []
		factor_levels_cardinality = 0
		
		for column in experimental_data.columns.values:
			column_factor_level = column
			factor_levels.append(column_factor_level)
		
		factor_levels_cardinality = len(factor_levels)
		n_repetitions = len(experimental_data[experimental_data.columns.values[0]])
		squared_residuals = [residual ** 2 for residual in residuals]
		squared_experimental_data = []
		for factor_level in experimental_data:
			squared_experimental_data.append([experimental_point ** 2 for experimental_point in experimental_data[factor_level]])
		squared_experimental_data = sum(squared_experimental_data,[])
		factor_effects = []
		for factor_level in factor_levels:
			factor_effects.append(model[factor_names[0] + "_" + factor_level])
		squared_factor_effects = [effect ** 2 for effect in factor_effects]
	
		SS["Y"] = sum(squared_experimental_data)
		SS["0"] = factor_levels_cardinality*n_repetitions*(model["grand_mean"]**2)
		SS["A"] = n_repetitions * sum(squared_factor_effects)
		SS["E"] = sum(squared_residuals)
		SS["T"] = SS["Y"] - SS["0"]

		df["A"] = (factor_levels_cardinality - 1)
		df["E"] = factor_levels_cardinality*(n_repetitions-1)
	if n_factors == 2: # two-factor full-factorial

		SS = {}
		df = {}

		# get the factor levels
		factor_levels = {}
		factor_levels_cardinality = {}
		for factor_name in factor_names:
			factor_levels[factor_name] = []

		for column in experimental_data.columns.values:
			column_factor_levels = column.split("_")
			for idx,column_factor_level in enumerate(column_factor_levels):
					if column_factor_level not in factor_levels[factor_names[idx]]:
						factor_levels[factor_names[idx]].append(column_factor_level)
		for factor in factor_levels:
			factor_levels_cardinality[factor] = len(factor_levels[factor])
		n_repetitions = len(experimental_data[experimental_data.columns.values[0]])
		squared_residuals = [residual ** 2 for residual in residuals]
		squared_experimental_data = []
		for factor_level in experimental_data:
			squared_experimental_data.append([experimental_point ** 2 for experimental_point in experimental_data[factor_level]])
		squared_experimental_data = sum(squared_experimental_data,[])
		
		first_factor_effects = []
		for factor_level in factor_levels[factor_names[0]]:
			first_factor_effects.append(model[factor_names[0] + "_" + factor_level])
		second_factor_effects = []
		for factor_level in factor_levels[factor_names[1]]:
			second_factor_effects.append(model[factor_names[1] + "_" + factor_level])
		combined_factor_effects = []
		for factor_level in factor_levels[factor_names[0]]:
			for second_factor_level in factor_levels[factor_names[1]]:
				combined_factor_effects.append(model[factor_names[0] + "_" + factor_level + "_" + factor_names[1] + "_" + second_factor_level])

		print(first_factor_effects)

		squared_first_factor_effects = [effect ** 2 for effect in first_factor_effects]
		squared_second_factor_effects = [effect ** 2 for effect in second_factor_effects]
		squared_combined_effects = [effect ** 2 for effect in combined_factor_effects]
		
		
		SS["Y"] = sum(squared_experimental_data)
		
		total_levels_mul = 1
		for factor in factor_levels_cardinality:
			total_levels_mul = total_levels_mul * factor_levels_cardinality[factor]
			
			

		SS["0"] = total_levels_mul*n_repetitions*(model["grand_mean"]**2)

		SS["A"] = factor_levels_cardinality[factor_names[1]] * n_repetitions * sum(squared_first_factor_effects)

		SS["B"] = factor_levels_cardinality[factor_names[0]] * n_repetitions * sum(squared_second_factor_effects)

		SS["AB"] = n_repetitions*sum(squared_combined_effects)

		SS["E"] = sum(squared_residuals)

		SS["T"] = SS["Y"] - SS["0"]

		df["A"] = (factor_levels_cardinality[factor_names[0]] - 1)
		df["B"] = (factor_levels_cardinality[factor_names[1]] - 1)
		df["AB"] = df["A"]*df["B"]
		df["E"] = factor_levels_cardinality[factor_names[0]]*factor_levels_cardinality[factor_names[1]]*(n_repetitions-1)

	
	return SS, df

def compute_factors_importance(SS):
	factors_importance = {}
	
	n_factors = len(factor_names)

	if n_factors == 1: # one-factor full-factorial
		factors_importance["A"] = SS["A"]/SS["T"]
		
	if n_factors == 2:
		factors_importance["A"] = SS["A"]/SS["T"]
		factors_importance["B"] = SS["B"]/SS["T"]
		factors_importance["AB"] = SS["AB"]/SS["T"]

	return factors_importance


def check_statistical_significance(experimental_data, factor_names, conf_level):

   
	
	statistical_significance_result = {}
	p_values = {}
	n_factors = len(factor_names)
	
	if n_factors == 1:
	
		factor_groups = []
		for column in experimental_data:
			factor_groups.append(experimental_data[column])
			
		test_statistic, p_value = stats.f_oneway(*factor_groups)
		p_values[factor_names[0]] = p_value
		
		if p_value > conf_level:
			statistical_significance_result[factor_names[0]] = False
		else:
			statistical_significance_result[factor_names[0]] = True

	if n_factors == 2:
		
		first_factor_groups = []
		second_factor_groups = []
		combined_factor_groups = []
		# get the factor levels
		factor_levels = {}
		factor_levels_cardinality = {}
		for factor_name in factor_names:
			factor_levels[factor_name] = []

		for column in experimental_data.columns.values:
			column_factor_levels = column.split("_")
			for idx,column_factor_level in enumerate(column_factor_levels):
					if column_factor_level not in factor_levels[factor_names[idx]]:
						factor_levels[factor_names[idx]].append(column_factor_level)
		for factor in factor_levels:
			factor_levels_cardinality[factor] = len(factor_levels[factor])
		
		reference_factor = factor_names[0]
		for level in factor_levels[reference_factor]:
			factor_group = []
			for column in experimental_data:
				column_split_values = column.split("_")
				if level == column_split_values[0]:
					#print("concatenating column group " + column)
					factor_group = [*factor_group, *experimental_data[column]]
			first_factor_groups.append(factor_group)
			
		# get the second factor groups
		reference_factor = factor_names[1]
		for level in factor_levels[reference_factor]:
			factor_group = []
			for column in experimental_data:
				column_split_values = column.split("_")
				if level == column_split_values[1]:
					#print("concatenating column group " + column)
					factor_group = [*factor_group, *experimental_data[column]]
			second_factor_groups.append(factor_group)

		# get the combined factor groups
		for column in experimental_data:
			combined_factor_groups.append(experimental_data[column])
			
		test_statistic_first_factor, p_value_first_factor = stats.f_oneway(*first_factor_groups)

		test_statistic_second_factor, p_value_second_factor = stats.f_oneway(*second_factor_groups)
		
		test_statistic_combined_factor, p_value_combined_factor = stats.f_oneway(*combined_factor_groups)	
		

		p_values["first_factor"] = p_value_first_factor
		p_values["second_factor"] = p_value_second_factor
		p_values["combined_factor"] = p_value_combined_factor

		if p_values["first_factor"] < conf_level:
			statistical_significance_result["first_factor"] = True
		else:
			statistical_significance_result["first_factor"] = False
		if p_values["second_factor"] < conf_level:
			statistical_significance_result["second_factor"] = True
		else:
			statistical_significance_result["second_factor"] = False
		if p_values["combined_factor"] < conf_level:
			statistical_significance_result["combined_factor"] = True
		else:
			statistical_significance_result["combined_factor"] = False	
		
		pass
	

	return statistical_significance_result, p_values

def check_statistical_significance_non_parametric(experimental_data, factor_names, conf_level):

	statistical_significance_result = {}
	p_values = {}

	n_factors = len(factor_names)

	if n_factors == 1: # one-factor full-factorial

		factor_groups = []

		for column in experimental_data:
			factor_groups.append(experimental_data[column])

		if len(factor_groups) > 2:
			test_statistic, p_value = stats.friedmanchisquare(*factor_groups)
		else:
			test_statistic, p_value = stats.wilcoxon(*factor_groups)

		p_values[factor_names[0]] = p_value
	
		if p_value > conf_level:
			statistical_significance_result[factor_names[0]] = False
		else:
			statistical_significance_result[factor_names[0]] = True
	if n_factors == 2: # two-factor full-factorial
		first_factor_groups = []
		second_factor_groups = []
		combined_factor_groups = []
		# get the factor levels
		factor_levels = {}
		factor_levels_cardinality = {}
		for factor_name in factor_names:
			factor_levels[factor_name] = []

		for column in experimental_data.columns.values:
			column_factor_levels = column.split("_")
			for idx,column_factor_level in enumerate(column_factor_levels):
					if column_factor_level not in factor_levels[factor_names[idx]]:
						factor_levels[factor_names[idx]].append(column_factor_level)
		for factor in factor_levels:
			factor_levels_cardinality[factor] = len(factor_levels[factor])

		# get the first factor groups
		reference_factor = factor_names[0]
		for level in factor_levels[reference_factor]:
			factor_group = []
			for column in experimental_data:
				column_split_values = column.split("_")
				if level == column_split_values[0]:
					#print("concatenating column group " + column)
					factor_group = [*factor_group, *experimental_data[column]]
			first_factor_groups.append(factor_group)

		# get the second factor groups
		reference_factor = factor_names[1]
		for level in factor_levels[reference_factor]:
			factor_group = []
			for column in experimental_data:
				column_split_values = column.split("_")
				if level == column_split_values[1]:
					#print("concatenating column group " + column)
					factor_group = [*factor_group, *experimental_data[column]]
			second_factor_groups.append(factor_group)

		# get the combined factor groups
		for column in experimental_data:
			combined_factor_groups.append(experimental_data[column])
	
		if len(first_factor_groups) > 2:
			test_statistic_first_factor, p_value_first_factor = stats.friedmanchisquare(*first_factor_groups)
		else:
			test_statistic_first_factor, p_value_first_factor = stats.wilcoxon(*first_factor_groups)
			
		if len(second_factor_groups) > 2:
			test_statistic_first_factor, p_value_second_factor = stats.friedmanchisquare(*second_factor_groups)
		else:
			test_statistic_first_factor, p_value_second_factor = stats.wilcoxon(*second_factor_groups)
			
		if len(combined_factor_groups) > 2:
			test_statistic_first_factor, p_value_combined_factor = stats.friedmanchisquare(*combined_factor_groups)
		else:
			test_statistic_first_factor, p_value_combined_factor = stats.wilcoxon(*combined_factor_groups)		
		

		p_values["first_factor"] = p_value_first_factor
		p_values["second_factor"] = p_value_second_factor
		p_values["combined_factor"] = p_value_combined_factor

		if p_values["first_factor"] < conf_level:
			statistical_significance_result["first_factor"] = True
		else:
			statistical_significance_result["first_factor"] = False
		if p_values["second_factor"] < conf_level:
			statistical_significance_result["second_factor"] = True
		else:
			statistical_significance_result["second_factor"] = False
		if p_values["combined_factor"] < conf_level:
			statistical_significance_result["combined_factor"] = True
		else:
			statistical_significance_result["combined_factor"] = False


	return statistical_significance_result, p_values

def compute_sample_mean(input_data):
	return sum(input_data)/len(input_data)

def compute_sample_variance(input_data, sample_mean):
	const_diff_data = [x - sample_mean for x in input_data]
	const_diff_data_squared = [x ** 2 for x in const_diff_data]
	return sum(const_diff_data_squared)/(len(input_data)-1)

def compute_samples_mean_variance(experimental_data):
	groups_mean = {}
	groups_variance = {}

	for column in experimental_data:
		column_data = experimental_data[column].tolist()
		groups_mean[column] = round(compute_sample_mean(column_data),5)
		groups_variance[column] = round(compute_sample_variance(column_data, groups_mean[column]),5)

	return groups_mean, groups_variance
		
def print_ANOVA_results(dataset_name, groups_mean, groups_variance, model, normality_check_result, omoscedasticity_check_result, SS, statistical_significance_result, p_values, conf_level, factor_names, min_p_value_idx, min_p_value, factors_importance):

	file = open(dataset_name + ".txt", "w")

	n_factors = len(factor_names)
	if n_factors == 1: # one-factor full-factorial
		for group in zip(groups_mean, groups_variance):
			print("The sample mean and variance of group " + group[0] + " are: (" + str(groups_mean[group[0]]) + "," + str(groups_variance[group[0]]) + ")")
			file.write("The sample mean and variance of group " + group[0] + " are: (" + str(groups_mean[group[0]]) + "," + str(groups_variance[group[0]]) + ")\n")
		for component in model:
			if component == "grand_mean":
				print("The grand mean of the model is: " + str(model[component]))
				file.write("The grand mean of the model is: " + str(model[component]) + "\n")
			else:
				print("The factor effect " + component + " of the model is: " + str(model[component]))
				file.write("The factor effect " + component + " of the model is: " + str(model[component]) + "\n")
		if normality_check_result == True:
			print("Residuals are normal")
			file.write("Residuals are normal\n")
		else:
			print("Residuals are not normal")
			file.write("Residuals are not normal\n")
		for ss in SS:
			print("The sum of squares SS" + ss + " is equal to: " + str(SS[ss]))
			file.write("The sum of squares SS" + ss + " is equal to: " + str(SS[ss]) + "\n")
		for factor in factors_importance:
			if factor == "A":
				print("The importance of factor " + factor_names[0] + " is equal to: " + str(factors_importance["A"]))
				file.write("The importance of factor " + factor_names[0] + " is equal to: " + str(factors_importance["A"]) + "\n")

		for factor in statistical_significance_result:
			if(statistical_significance_result[factor] == True):
				print("Sample groups linked to factor " + factor_names[0] + " show statistically significant differences with 95% confidence (p-value = " + str(p_values[factor]) + ")")
				file.write("Sample groups linked to factor " + factor_names[0] + " show statistically significant differences with 95% confidence (p-value = " + str(p_values[factor]) + ")\n")
			else:
				print("Sample groups linked to factor " + factor_names[0] + " do not show statistically significant differences with 95% confidence (p-value = " + str(p_values[factor]) + ")")
				file.write("Sample groups linked to factor " + factor_names[0] + " do not show statistically significant differences with 95% confidence (p-value = " + str(p_values[factor]) + ")\n")
		
	if n_factors == 2: # two-factor full-factorial
		for group in zip(groups_mean, groups_variance):
			print("The sample mean and variance of group " + group[0] + " are: (" + str(groups_mean[group[0]]) + "," + str(groups_variance[group[0]]) + ")")
			file.write("The sample mean and variance of group " + group[0] + " are: (" + str(groups_mean[group[0]]) + "," + str(groups_variance[group[0]]) + ")\n")
		for component in model:
			if component == "grand_mean":
				print("The grand mean of the model is: " + str(model[component]))
				file.write("The grand mean of the model is: " + str(model[component]) + "\n")
			else:
				print("The factor effect " + component + " of the model is: " + str(model[component]))
				file.write("The factor effect " + component + " of the model is: " + str(model[component]) + "\n")
		if normality_check_result == True:
			print("Residuals are normal")
			file.write("Residuals are normal\n")
		else:
			print("Residuals are not normal")
			file.write("Residuals are not normal\n")
		if omoscedasticity_check_result == True:
			print("Residuals are omoscedastic")
			file.write("Residuals are omoscedastic\n")
		else:
			print("Residuals are not omoscedastic")
			file.write("Residuals are not omoscedastic\n")

		for ss in SS:
			print("The sum of squares SS" + ss + " is equal to: " + str(SS[ss]))
			file.write("The sum of squares SS" + ss + " is equal to: " + str(SS[ss]) + "\n")
		
		for factor in factors_importance:
			if factor == "A":
				print("The importance of factor " + factor_names[0] + " is equal to: " + str(factors_importance["A"]))
				file.write("The importance of factor " + factor_names[0] + " is equal to: " + str(factors_importance["A"]) + "\n")
			if factor == "B":
				print("The importance of factor " + factor_names[1] + " is equal to: " + str(factors_importance["B"]))
				file.write("The importance of factor " + factor_names[1] + " is equal to: " + str(factors_importance["B"]) + "\n")
			if factor == "AB":
				print("The importance of factor " + factor_names[0] + " and factor " + factor_names[1] + " is equal to: " + str(factors_importance["AB"]))
				file.write("The importance of factor " + factor_names[0] + " and factor " + factor_names[1] + " is equal to: " + str(factors_importance["AB"]) + "\n")
		
		for factor in statistical_significance_result:
			if factor == "first_factor":
				if(statistical_significance_result[factor] == True):
					print("Sample groups linked to factor " + factor_names[0] + " show statistically significant differences with " + str((1-conf_level)*100) + "% confidence (p-value = " + str(round(p_values[factor],4)) + ")")
					file.write("Sample groups linked to factor " + factor_names[0] + " show statistically significant differences with " + str((1-conf_level)*100) + "% confidence (p-value = " + str(round(p_values[factor],4)) + ")\n")
				else:
					print("Sample groups linked to factor " + factor_names[0] + " do not show statistically significant differences with " + str((1-conf_level)*100) + "% confidence (p-value = " + str(round(p_values[factor],4)) + ")")
					file.write("Sample groups linked to factor " + factor_names[0] + " do not show statistically significant differences with " + str((1-conf_level)*100) + "% confidence (p-value = " + str(round(p_values[factor],4)) + ")\n")
			elif factor == "second_factor":
				if(statistical_significance_result[factor] == True):
					print("Sample groups linked to factor " + factor_names[1] + " show statistically significant differences with " + str((1-conf_level)*100) + "% confidence (p-value = " + str(round(p_values[factor],4)) + ")")
					file.write("Sample groups linked to factor " + factor_names[1] + " show statistically significant differences with " + str((1-conf_level)*100) + "% confidence (p-value = " + str(round(p_values[factor],4)) + ")\n")
				else:
					print("Sample groups linked to factor " + factor_names[1] + " do not show statistically significant differences with " + str((1-conf_level)*100) + "% confidence (p-value = " + str(round(p_values[factor],4)) + ")")
					file.write("Sample groups linked to factor " + factor_names[1] + " do not show statistically significant differences with " + str((1-conf_level)*100) + "% confidence (p-value = " + str(round(p_values[factor],4)) + ")\n")
			elif factor == "combined_factor":
				if(statistical_significance_result[factor] == True):
					print("Sample groups linked to factor " + factor_names[0] + " and factor " + factor_names[1] + " combined show statistically significant differences with " + str((1-conf_level)*100) + "% confidence (p-value = " + str(round(p_values[factor],4)) + ")")
					file.write("Sample groups linked to factor " + factor_names[0] + " and factor " + factor_names[1] + " combined show statistically significant differences with " + str((1-conf_level)*100) + "% confidence (p-value = " + str(round(p_values[factor],4)) + ")\n")
				else:
					print("Sample groups linked to factor " + factor_names[0] + " and factor " + factor_names[1] + " combined do not show statistically significant differences with " + str((1-conf_level)*100) + "% confidence (p-value = " + str(round(p_values[factor],4)) + ")")
					file.write("Sample groups linked to factor " + factor_names[0] + " and factor " + factor_names[1] + " combined do not show statistically significant differences with " + str((1-conf_level)*100) + "% confidence (p-value = " + str(round(p_values[factor],4)) + ")\n")
			
		print("Sample level pair linked to factor " + factor_names[0] + " with the lowest paired p-value computed through a post-hoc test is the pair (" + min_p_value_idx["first_factor"][0] + "," + min_p_value_idx["first_factor"][1] + ") with p-value: " + str(round(min_p_value["first_factor"],4)))
		file.write("Sample level pair linked to factor " + factor_names[0] + " with the lowest paired p-value computed through a post-hoc test is the pair (" + min_p_value_idx["first_factor"][0] + "," + min_p_value_idx["first_factor"][1] + ") with p-value: " + str(round(min_p_value["first_factor"],4))+"\n")
		print("Sample level pair linked to factor " + factor_names[1] + " with the lowest paired p-value computed through a post-hoc test is the pair (" + min_p_value_idx["second_factor"][0] + "," + min_p_value_idx["second_factor"][1] + ") with p-value: " + str(round(min_p_value["second_factor"],4)))
		file.write("Sample level pair linked to factor " + factor_names[1] + " with the lowest paired p-value computed through a post-hoc test is the pair (" + min_p_value_idx["second_factor"][0] + "," + min_p_value_idx["second_factor"][1] + ") with p-value: " + str(round(min_p_value["second_factor"],4)) + "\n")

	file.close()
	return None
	
def apply_post_hoc_analysis(experimental_data, factor_names):
	
	n_factors = len(factor_names)

	if n_factors == 2:
		p_values = {}
		
		factor_levels = {}
		factor_levels_cardinality = {}
		for factor_name in factor_names:
			factor_levels[factor_name] = []
			
			

		for column in experimental_data.columns.values:
			column_factor_levels = column.split("_")
			for idx,column_factor_level in enumerate(column_factor_levels):
				if column_factor_level not in factor_levels[factor_names[idx]]:
					factor_levels[factor_names[idx]].append(column_factor_level)
		for factor in factor_levels:
			factor_levels_cardinality[factor] = len(factor_levels[factor])
		n_repetitions = len(experimental_data[experimental_data.columns.values[0]])
		
		
		
		first_factor_groups = {}
		second_factor_groups = {}
		
		for factor_level in factor_levels[factor_names[0]]:
			first_factor_groups[factor_level] = []
			for column in experimental_data.columns.values:
				column_factor_levels = column.split("_")
				if factor_level == column_factor_levels[0]:
					first_factor_groups[factor_level].append(experimental_data[column].tolist())
		
		for factor_level in factor_levels[factor_names[1]]:
			second_factor_groups[factor_level] = []
			for column in experimental_data.columns.values:
				column_factor_levels = column.split("_")
				if factor_level == column_factor_levels[1]:
					second_factor_groups[factor_level].append(experimental_data[column].tolist())
		
		
		for level in first_factor_groups:
			temp = []
			for list in first_factor_groups[level]:
				temp = [*temp, *list]
			first_factor_groups[level] = temp
		for level in second_factor_groups:
			temp = []
			for list in second_factor_groups[level]:
				temp = [*temp, *list]
			second_factor_groups[level] = temp

		first_factor_data = np.array([*first_factor_groups.values()])
		second_factor_data = np.array([*second_factor_groups.values()])

		p_values["first_factor"] = sp.posthoc_nemenyi_friedman(first_factor_data.T)
		p_values["second_factor"] = sp.posthoc_nemenyi_friedman(second_factor_data.T)

		min_p_value = {}
		min_p_value_idx = {}

		# first factor minimum p-value
		min_p_value["first_factor"] = 1.0
		min_p_value_idx["first_factor"] = [0, 0]
		for row in p_values["first_factor"]:
			for idx,p_val in enumerate(p_values["first_factor"][row]):
				if p_val < min_p_value["first_factor"]:
					min_p_value["first_factor"] = p_val
					min_p_value_idx["first_factor"] = [row, idx]

		# second factor minimum p-value
		min_p_value["second_factor"] = 1.0
		min_p_value_idx["second_factor"] = [0, 0]
		for row in p_values["second_factor"]:
			for idx,p_val in enumerate(p_values["second_factor"][row]):
				if p_val < min_p_value["second_factor"]:
					min_p_value["second_factor"] = p_val
					min_p_value_idx["second_factor"] = [row, idx]

		for idx,level in enumerate(factor_levels[factor_names[0]]):
			if idx == min_p_value_idx["first_factor"][0]:
				min_p_value_idx["first_factor"][0] = level
			if idx == min_p_value_idx["first_factor"][1]:
				min_p_value_idx["first_factor"][1] = level
				
		for idx,level in enumerate(factor_levels[factor_names[1]]):
			if idx == min_p_value_idx["second_factor"][0]:
				min_p_value_idx["second_factor"][0] = level
			if idx == min_p_value_idx["second_factor"][1]:
				min_p_value_idx["second_factor"][1] = level

		return min_p_value_idx, min_p_value
	else:
		return None, None


input_data_files = []
factor_names = []
conf_level = 0.05

try:
	if len(sys.argv) < 2:
		raise Exception

	for idx,data_filename in enumerate(sys.argv):
		if idx == 1:
			factors = sys.argv[idx]
			factors = factors.split("_")
			for factor in factors:
				factor_names.append(factor)
		if idx > 1:
			input_data_files.append(data_filename)
except Exception:
	print("Not enough input arguments provided. Please, specify at least the factors (in a single argument, separating the names with a '_'), and one response data file (in csv format).")
	sys.exit()

experimental_data = load_data(input_data_files)


for dataset in experimental_data:


	groups_mean, groups_variance = compute_samples_mean_variance(experimental_data[dataset])
	
	model = compute_model(experimental_data[dataset], factor_names)
	
	residuals, predicted_responses = compute_residuals(experimental_data[dataset], model, factor_names)
	
	normality_check_result = check_normality(residuals)
	
	omoscedasticity_check_result = check_omoscedasticity(residuals, predicted_responses)
	
	SS, df = compute_SS(residuals, model, experimental_data[dataset], factor_names)
	
	factors_importance = compute_factors_importance(SS)
	
	if normality_check_result == True and omoscedasticity_check_result == True:
		statistical_significance_result, p_values = check_statistical_significance(experimental_data[dataset],factor_names, conf_level)
	else:
		statistical_significance_result, p_values = check_statistical_significance_non_parametric(experimental_data[dataset],factor_names, conf_level)

	min_p_value_idx, min_p_value = apply_post_hoc_analysis(experimental_data[dataset], factor_names)
	
	print_ANOVA_results(dataset,groups_mean, groups_variance, model, normality_check_result, omoscedasticity_check_result, SS, statistical_significance_result, p_values, conf_level, factor_names, min_p_value_idx, min_p_value, factors_importance)
	
	













