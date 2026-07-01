# PR Notes

## Summary
These commits focus on refactoring the `GWPCA` output data structures for better readability and downstream usability, improving test robustness, and resolving package rename conflicts with the upstream repository.

## Changes Included
- **Merge upstream main, resolve spatialml->spml rename conflicts**: Cleanly resolved the file path and naming conflicts arising from the upstream project renaming from `spatialml` to `spml`.
- **Refactor GWPCA components to MultiIndex DataFrame and apply string labels**: Changed the raw numpy array output of `components_` to a labeled pandas `MultiIndex` DataFrame, making it significantly easier to interpret local component loadings. Also applied string labels to variance attributes.
- **Fix CI type checking for pandas Indexing**: Resolved typing issues caught by CI related to pandas indexing mechanics.
- **test: abstract R-specific terminology from GWPCA tests and loosen tolerances**: test mathematical correctness rather than just mocking an external package. Loosened exactness tolerances slightly (`atol=1e-5`) to account for differences across architectures.
