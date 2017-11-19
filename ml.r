set.seed(308)

#load
library(data.table)
cat('loading...\n')
dir <- 'simdata'
files <- list.files(path=dir, pattern='*.csv')
temp <- lapply(paste0(dir, '/', files), fread, sep=',')
df <- rbindlist(temp)

#train
library(Cubist)
library(mlbench)
library(dplyr)
cat('training...\n')
sdf <- sample_n(df, 500000)
x <- sdf[, c('DL', 'TMP', 'RAD', 'DVI'), with=F]
y <- sdf[['DVR']]
model <- cubist(x, y)
sink('output/summary.txt'); print(summary(model)); sink()

#parameter tuning with caret
# library(doParallel)
# require(caret)
# cl <- makeCluster(8); registerDoParallel(cl)
# df <- sample_n(df, 100)
# x <- df[, c('DL', 'TMP', 'RAD', 'DVI'), with=F]
# y <- df[['DVR']]
# fit <- train(x, y, 'cubist', committees=1)
# fit
# fit$times$everything
# stopCluster(cl); registerDoSEQ()

#plot
# library(gridExtra)
# p1 <- dotplot(model, what='splits', main='Conditions')
# p2 <- dotplot(model, what='coefs', main='Coefs')
# grid.arrange(p1,p2)
