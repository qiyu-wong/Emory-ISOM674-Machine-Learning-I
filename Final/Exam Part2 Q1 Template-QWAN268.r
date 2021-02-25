
# Question 1, Tree models for the Spam data

suppressWarnings(if(!require("tree")) { install.packages("tree"); require("tree") })

set.seed(20201207)
DataOrig <- read.table("spambasedata-Orig.csv",sep=",",header=T,
                       stringsAsFactors=F)

ord <- sample(nrow(DataOrig))
DataOrig <- DataOrig[ord,]

# Change IsSpam to a factor

DataOrig$IsSpam <- factor(DataOrig$IsSpam)

# Doing a 60-20-20 split
TrainInd <- ceiling(nrow(DataOrig)*0.6)
TrainDF <- DataOrig[1:TrainInd,]
tmpDF <- DataOrig[-(1:TrainInd),]
ValInd <- ceiling(nrow(tmpDF)*0.5)
ValDF <- tmpDF[1:ValInd,]
TestDF <- tmpDF[-(1:ValInd),]

remove(TrainInd,tmpDF,ValInd,ord)

# Question 1a --------------------------------------------------------------
# Write an R function to evaluate the log likelihood

loglike <- function(PHat,Y) {
  # Phat is the vector of predicted probabilities from your model
  # Y is the vector of 0's and 1's indicating if the e-mail is a spam email.
  eps <- 1e-10
  PHat <- (PHat+eps)/(1+2*eps)
  #YOUR CODE HERE
  LogLike <- sum(as.numeric(Y)*PHat-log(1+exp(PHat)))
  return(LogLike)
}

# Question 1b --------------------------------------------------------------
# Fit the "big" tree

# YOUR CODE HERE
Fm <- IsSpam ~ .
tc <- tree.control(nrow(TrainDF),minsize=2,mincut=1,mindev=0)
out <- tree(Fm,data=TrainDF,control=tc,method="deviance")

# Question 1c --------------------------------------------------------------
# Compute the performance for trees sizes

# YOUR CODE HERE
ll <-vector()
for (num in 2:20){
  out1 <- prune.tree(out,best=num)
  Yhat <- predict(out1,newdata=ValDF,type="vector")
  ll[num-1] <- loglike(Yhat[,2],ValDF$IsSpam)
}

# Question 1d to 1h --------------------------------------------------------------

# YOUR CODE HERE. Clearly label your answers to each question

# Question 1d --------------------------------------------------------------
ll_p <- data.frame(N = 2:20, ll = ll)
plot(ll_p)

# Question 1e --------------------------------------------------------------
BestN <- ll_p[which.max(ll_p$ll),1]
BestN
#20 is the best size
out1 <- prune.tree(out,best=BestN)

Yhat <- predict(out1,newdata=ValDF,type="vector")
loglike(Yhat[,2],ValDF$IsSpam)
#LL is -199.7814

# Question 1f --------------------------------------------------------------
source("C:/Users/10331/OneDrive/Desktop/RocPlot.r")
ROCPlot(Yhat[,2],ValDF$IsSpam)
round(ROCPlot(Yhat[,2],ValDF$IsSpam)$AUC,3)
#AUC is 0.964

# Question 1g --------------------------------------------------------------
Yhat_t <- predict(out1,newdata=TestDF,type="vector")
loglike(Yhat_t[,2],TestDF$IsSpam)
#LL is -209.4273

# Question 1h --------------------------------------------------------------
ROCPlot(Yhat_t[,2],TestDF$IsSpam)
round(ROCPlot(Yhat_t[,2],TestDF$IsSpam)$AUC,3)
#AUC is 0.958

