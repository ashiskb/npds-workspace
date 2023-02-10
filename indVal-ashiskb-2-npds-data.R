#Please set the current working directory to the source directory
#check getwd() returns what you need

#loading the packages
library(labdsv)

dataset_root = '/home/ashiskb/Documents/data/NPDS-data/'

#import the data from csv files: {data file, group file}
#Import the first csv file 'df_npds_data.csv' containing both samples (rows) with features(columns)
#First column in the file is called 'product'. That's not part of the features (i.e., species)
#where: 
#product: str, class names
#Make sure, no feature column is all zeros, i.e., each feature needs to be present (=True/1) at least one of the sample
npds_data <- read.csv(paste(dataset_root,'indval-workspace/df_npds_data.csv',sep=''),sep=',')#,row.names=1) #main csv file


#Import the second csv file 'df_npds_groups.csv' containing class name to integer mapping
# because indval can not handle categorical/non-numeric values as the 2nd argument to indval()
#Make sure, it has the same number of rows as the first csv file
#And, it maps categorical class names to integers
npds_groups <- read.csv(paste(dataset_root,'indval-workspace/df_npds_groups.csv',sep=''),sep=',')




#calculate the indicator values for species:
#first argument (feature only): npds_data[,-1] >> meaning all columns except the first column. Note. R indexing starts with 1
#2nd arg (labels/env/species): npds_groups[,2] >> the numeric/integer labels
iva <- indval(npds_data[,-1], npds_groups[,2], too.many=131)


##Now, use this to gather only the most important information.
## group is the 'class names', and 
#  freq is the number of times the feature was present among the samples.
gr <- iva$maxcls[iva$pval<=0.05] #group
iv <- iva$indcls[iva$pval<=0.05] #indicator value
pv <- iva$pval[iva$pval<=0.05]  #p-value
fr <- apply(npds_data[,-1]>0, 2, sum)[iva$pval<=0.05] #frequency

#Now, let's create the summary
indvalsummary <- data.frame(group=gr, indval=iv, pvalue=pv, freq=fr)
indvalsummary <- indvalsummary[order(indvalsummary$group, -indvalsummary$indval),]

#Sort by indval
indvalsummary_sorted_by_indval_desc <- indvalsummary[order(-indvalsummary$indval),]

#Let us see the result
#indvalsummary


##Export result to a csv file
write.csv(indvalsummary_sorted_by_indval_desc, paste(dataset_root,'indval-workspace/indvalsummary-npds-on-131-features.csv',sep=''))
