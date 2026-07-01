# Script to generate an adaptive external reference fixture for pytest.
# Run this on a machine with the 'GWmodel' and 'jsonlite' packages installed.
# It will generate 'gwpca_adaptive_reference_fixture.json' for Python-side
# comparison tests.

library(GWmodel)
library(jsonlite)

# 1. Define the reference dataset used in spml/tests/test_decomposition.py.
# The vector is reshaped using column-major matrix order, then standardized
# before fitting so it matches the Python fixture.
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
colnames(X) <- c("A", "B", "C")
X <- scale(X)

# Geometry: points on a line from (0,0) to (9,9)
coords <- cbind(0:9, 0:9)
sdf <- SpatialPointsDataFrame(coords, as.data.frame(X))

# 2. Run the external reference implementation.
# Python configuration:
# GWPCA(n_components=2, bandwidth=5, fixed=False, kernel="bisquare",
#       include_focal=True)
# Here, the reference implementation takes the adaptive bandwidth as an
# integer number of neighbors.
model <- gwpca(sdf, vars=c("A", "B", "C"), bw=8, k=2, kernel="bisquare", adaptive=TRUE, scaling=FALSE)

# 3. Extract the outputs
loadings <- model$loadings
variance <- model$var
local_pv <- model$local.PV

# Save to JSON
output <- list(
  loadings = loadings,
  variance = variance,
  local_pv = local_pv
)

write_json(output, "gwpca_adaptive_reference_fixture.json", digits = 8)
cat("Successfully generated gwpca_adaptive_reference_fixture.json\n")
