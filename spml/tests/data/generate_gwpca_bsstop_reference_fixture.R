# Script to generate a second GWPCA external reference fixture from an
# official example workflow.
#
# Run this on a machine with the 'GWmodel' and 'jsonlite' packages
# installed. It will generate 'gwpca_bsstop_fixture.json' for use in
# Python-side parity tests.

library(GWmodel)
library(jsonlite)

data(bsstop)

# Follow the example documented in the reference manual:
# take the continuous variables, standardize them, and run GWPCA on the
# stop coordinates.
X <- scale(as.matrix(bsstop[, 5:14]))
colnames(X) <- colnames(bsstop)[5:14]
coords <- as.matrix(cbind(bsstop$XCOO, bsstop$YCOO))
sdf <- SpatialPointsDataFrame(coords, as.data.frame(X))

# The manual demonstrates gwpca(..., bw = 1000000, k = 10) on this dataset.
model <- gwpca(
  sdf,
  vars = colnames(sdf@data),
  bw = 1000000,
  k = 10,
  kernel = "bisquare",
  adaptive = FALSE,
  scaling = FALSE
)

output <- list(
  columns = colnames(sdf@data),
  coords = unname(coords),
  data = unname(as.matrix(sdf@data)),
  loadings = model$loadings,
  variance = model$var,
  local_pv = model$local.PV
)

write_json(output, "gwpca_bsstop_fixture.json", digits = 8)
cat("Successfully generated gwpca_bsstop_fixture.json\n")
