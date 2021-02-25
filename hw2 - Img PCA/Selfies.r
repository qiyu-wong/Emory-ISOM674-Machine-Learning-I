
# Read in the data

FileName <- "SelfieImageData.csv"
Labs <- scan(file=FileName,what="xx",nlines=1,sep="|")
DataAsChars <- matrix(scan(file=FileName,what="xx",sep="|",skip=1),byrow=T,ncol=length(Labs))
colnames(DataAsChars) <- Labs
dim(DataAsChars)

# size in memory in MBs
as.double(object.size(DataAsChars)/1024/1024)

ImgData <- matrix(as.integer(DataAsChars[,-1]),nrow=nrow(DataAsChars))
colnames(ImgData) <- Labs[-1]
rownames(ImgData) <- DataAsChars[,1]
# size in memory in MBs
#as.double(object.size(ImgData)/1024/1024)

# Take a look
ImgData[1:8,1:8]
dim(ImgData)

# Free up some memory just in case
remove(DataAsChars)

# Show each Image
for(whImg in 1:nrow(ImgData)) {
  Img <- matrix(ImgData[whImg,],byrow=T,ncol=sqrt(ncol(ImgData)))
  Img <- apply(Img,2,rev)
  par(pty="s",mfrow=c(1,1))
  image(z=t(Img),col = grey.colors(255),useRaster=T)
  Sys.sleep(1)
}

# -------------------------------------- Q1  ---------------------------------------
dim(ImgData) # 28


# ----------------------------- Q2 average face  ------------------------------------
Img_avg <- colMeans(ImgData) # get the average of all the columns (pixels)
Img_avg <- apply(ImgData, 2, FUN=mean)

Img <- matrix(Img_avg,byrow=T,ncol=sqrt(ncol(ImgData)))
Img <- apply(Img,2,rev)
par(pty="s",mfrow=c(1,1))
image(z=t(Img),col = grey.colors(255),useRaster=T)

# -------------------------------------- Q3  ---------------------------------------
# 28

# ------------------------------- Q5 scree plot  ------------------------------------
Img_centered <- sweep(ImgData, 2, Img_avg) # get centered ImgData
Img_centered_t <- t(Img_centered)

small_matrix <- Img_centered %*% Img_centered_t # generate Xc * Xc_t
SDecomp <- eigen(small_matrix) # get eigenvector and eigenvalue

par(mfrow=c(1,1))
plot(SDecomp$values/27,
     col.ticks = NULL)
title("jhan372: Scree Plot")


# -------------------------------- Q6 -----------------------------------------------
sprintf("%.2f", max(SDecomp$values/27))
# 65608278.25

# -------------------------------- Q7 -----------------------------------------------

norm_vec <- function(x) sqrt(sum(x^2)) # Eculidean norm
eigen_vectors<-data.frame(matrix(ncol=28,nrow = 203401))

# first way 
xctv <- t(Img_centered) %*% SDecomp$vectors
norm <- apply(xctv, 2, norm_vec)
eigen_vectors <- sweep(xctv, 2, norm, FUN="/")

# second way
for (i in 1:28){
  eigen_vectors[,i] <- Img_centered_t %*% SDecomp$vectors[,i] / norm_vec(Img_centered_t %*% SDecomp$vectors[,i])
}

eigen_values <- SDecomp$values/27

# see how many eigenvalues needed to explain 85% variance --> 12
sum(cumsum(eigen_values)/sum(eigen_values) < 0.85) +1

# -------------------------------- Q8 -----------------------------------------------
PCompTrain20d <- ImgData %*% as.matrix(eigen_vectors[,1:20]) # dim = 28*20
ReconTrain20d <- PCompTrain20d %*% t(eigen_vectors[,1:20]) # dim = 28*203401

which(rownames(ImgData) == 'jhan372') # return 10

par(mfrow=c(1,1))
Img <- matrix(ReconTrain20d[10,],byrow=T,ncol=sqrt(ncol(ReconTrain20d)))
Img <- apply(Img,2,rev)
image(z=t(Img),col = grey.colors(255),useRaster=T)

# -------------------------------- Q10 -----------------------------------------------
par(mfrow=c(1,1))
vec <- eigen_vectors[,1]
vec <- (vec-min(vec))/(max(vec)-min(vec))
vec <- vec*255 
range(vec)

vecImage <- matrix(vec,byrow=T,ncol=sqrt(length(vec)))
vecImage <- t(apply(vecImage,2,rev))
image(z=vecImage,col = grey.colors(255),useRaster=T)
title("Eigenface for\nEigenvector 8")

# -------------------------------- Q11 -----------------------------------------