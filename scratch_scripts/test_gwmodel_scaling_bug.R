X_data <- c(
  0.3745401188473625, 0.5986584841970366, 0.05808361216819946, 0.7080725777960455, 
  0.8324426408004217, 0.18340450985343382, 0.43194501864211576, 0.13949386065204183, 
  0.45606998421703593, 0.5142344384136116, 0.9507143064099162, 0.15601864044243652, 
  0.8661761457749352, 0.020584494295802447, 0.21233911067827616, 0.3042422429595377, 
  0.2912291401980419, 0.29214464853521815, 0.7851759613930136, 0.5924145688620425, 
  0.7319939418114051, 0.15599452033620265, 0.6011150117432088, 0.9699098521619943, 
  0.18182496720710062, 0.5247564316322378, 0.6118528947223795, 0.3663618432936917, 
  0.19967378215835974, 0.046450412719997725
)
X <- matrix(X_data, nrow=10, ncol=3)
colnames(X) <- c('A', 'B', 'C')
library(GWmodel)
coords <- cbind(0:9, 0:9)
sdf <- SpatialPointsDataFrame(coords, as.data.frame(X))

# GWmodel 2.4-2 gwpca has a bug in its scale handling logic.
# Line 133 of `gwpca.r` reads:
#    if (scaling) 
#        x <- scale(x, center = T, scale = T)
#    else x <- scale(x, center = T)
#
# Because `scale` in R defaults to `scale=TRUE`, the `else` branch still scales the data globally!

# To demonstrate how gwpca internally computes local svd, we hook into it:
global_i <<- 1
my_wpca <- function (x, wt, nu = 0, nv = 2) {
    local.center <- function(x, wt) sweep(x, 2, colSums(sweep(x, 1, wt, '*'))/sum(wt))
    res <- svd(sweep(local.center(x, wt), 1, sqrt(wt), '*'), nu = nu, nv = nv)
    print(paste('d^2 for call', global_i, ':'))
    print(res[['d']]^2)
    global_i <<- global_i + 1
    res
}

assignInNamespace('wpca', my_wpca, ns='GWmodel')
gwpca(sdf, vars=c('A', 'B', 'C'), bw=5, k=2, kernel='bisquare', adaptive=FALSE, scaling=FALSE, cv=FALSE)
