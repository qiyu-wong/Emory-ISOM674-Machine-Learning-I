# Create Datasets for Neural Net ------------------------------------------

load("C:/Users/10331/OneDrive/Desktop/SamePersonMatchingDataFiles.RData",verbose=T)

# If previously fit models exist, load them in.
# This file will overwrite any models that have the same names as those created here

TrainDF <- data.frame(Y=YTrMtch,X=XTrMtch[,1:100])
ValDF <- data.frame(Y=YValMtch,X=XValMtch[,1:100])

write.table(TrainDF,file="C:/Users/10331/OneDrive/Desktop/TrainDF.csv",sep=",",row.names=F,col.names=T)
write.table(ValDF,file="C:/Users/10331/OneDrive/Desktop/ValDF.csv",sep=",",row.names=F,col.names=T)


# -------------------------------------------------------------------------
# Read in the test Neural Net Output and Compute the ROC and AUC

TrainOutput <- read.table("C:/Users/10331/OneDrive/Desktop/TrainDFOutput.csv",header=T,sep=",",quote=NULL,stringsAsFactors = F)
ValOutput <- read.table("C:/Users/10331/OneDrive/Desktop/ValDFOutput.csv",header=T,sep=",",quote=NULL,stringsAsFactors = F)
names(ValOutput)
# Compute the ROC and the AUC

# Your Code Here
source("C:/Users/10331/OneDrive/Desktop/RocPlot.r")
ROCPlot(ValOutput$ValP,ValOutput$Y)
ROCPlot(ValOutput$ValP,ValOutput$Y)$AUC
#AUC is 0.901

# -------------------------------------------------------------------------
# Read in the Narrow Neural Net Output and Compute the ROC and AUC

TrainOutput <- read.table("C:/Users/10331/OneDrive/Desktop/TrainDFOutput2.csv",header=T,sep=",",quote=NULL,stringsAsFactors = F)
ValOutput <- read.table("C:/Users/10331/OneDrive/Desktop/ValDFOutput2.csv",header=T,sep=",",quote=NULL,stringsAsFactors = F)
names(ValOutput)
# Compute the ROC and the AUC

# Your Code Here
ROCPlot(ValOutput$ValP,ValOutput$Y)
ROCPlot(ValOutput$ValP,ValOutput$Y)$AUC
#AUC is 0.883
