#init
set.seed(308)
out <- './output/'

#load
library(data.table)
dir <- 'simdata'
files <- list.files(path=dir, pattern='*.csv')
temp <- lapply(paste0(dir, '/', files), fread, sep=',')
df <- rbindlist(temp)

#dataset
library(dplyr)
# sdf <- sample_n(df, 500000)
sdf <- sample_n(df, 5000)
x <- sdf[, c('DL', 'TMP', 'RAD', 'DVI'), with=F]
y <- sdf[['DVR']]

#train (cubist)
# library(Cubist)
# model <- cubist(x, y)
# sink(paste0(out, 'cubist.txt')); print(summary(model)); sink()
# saveRDS(model, paste0(out, 'cubist.obj'))

#train (mob)
library(party)
ctrl <- party::mob_control(
                           alpha=0.05,
                           bonferroni=TRUE,
                           minsplit=500,
                           objfun=deviance,
                           verbose=FALSE)
model <- party::mob(DVR ~ DL+TMP|DVI,
                    data=sdf,
                    control=ctrl,
                    model=linearModel)
#todo: try non-linear
sink(paste0(out, 'mob.txt')); print(model, summary(model)); sink()
pdf(paste0(out, 'mob.pdf'), width=12); plot(model); dev.off()
saveRDS(model, paste0(out, 'mob.obj'))

#train (M5P)
#  - M5P controls: WOW(M5P)
# library(RWeka)
# library(partykit)
# model <- M5P(DVR~TMP+DL+DVI,
#              data=sdf,
#              control=Weka_control(N=F, M=200))
# sink(paste0(out, 'm5p.txt')); print(model, summary(model)); sink()
# pdf(paste0(out, 'm5p.pdf')); plot(as.party.Weka_tree(model)); dev.off()
# saveRDS(model, paste0(out, 'm5p.obj'))

#todo: plot prediction
