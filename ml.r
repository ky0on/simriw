set.seed(308)

#load
library(data.table)
cat('loading...')
dir <- 'simdata'
files <- list.files(path=dir, pattern='*.csv')
temp <- lapply(paste0(dir, '/', files), fread, sep=',')
df <- rbindlist(temp)

#train
library(Cubist)
library(mlbench)
cat('training...\n')
model <- cubist(x=df[, c('DL', 'TMP', 'RAD', 'DVI'), with=F], y=df[['DVR']])
# summary(model)

#plot
# library(gridExtra)
# p1 <- dotplot(model, what='splits', main='Conditions')
# p2 <- dotplot(model, what='coefs', main='Coefs')
# grid.arrange(p1,p2)
