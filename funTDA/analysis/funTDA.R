# ================================================================
# Load libraries
# ================================================================
library(TDA)
library(fda)
library(ggplot2)
library(ggrepel)
library(patchwork)
set.seed(123)

# ================================================================
# Path to files
# ================================================================
ph_folder <- "output/PH" 
files <- list.files(ph_folder, pattern = "_PH.csv$", full.names = TRUE)

#extract the numeric ID from filename
get_number <- function(fname){
 as.numeric(sub(".*adjmatrix([0-9]+)_PH.*", "\\1", fname))
}

#order files numerically
file_numbers <- sapply(files, get_number)
files <- files[order(file_numbers)]

# ================================================================
# Compute persistence landscapes
# ================================================================
DataSymp <- c()
DataAsymp <- c()
symp_labels <- c()
asymp_labels <- c()
tseq <- seq(0, 1, length.out = 500)

read_landscape <- function(file_path, dim = 1, tseq = NULL) {
  stopifnot(!is.null(tseq))
  PH <- read.csv(file_path)
  colnames(PH) <- c("Birth", "Death", "dimension")
  PH <- PH[c("dimension", "Birth", "Death")]
  PH <- as.matrix(PH)
  x <- landscape(PH, dimension = dim, KK = 1, tseq)
  return(x)
}

for(f in files){
  x1 <- as.vector(read_landscape(f, dim = 1, tseq = tseq))

  # detect group from filename
  fname <- basename(f)
  id <- get_number(fname)
  # files is already sorted ascending by id (see above), so appending with
  # rbind (rather than prepending with cbind) keeps each group's rows in
  # ascending ID order; labels are derived from that same real ID rather
  # than hardcoded, so a row's label always matches its source file.
  if(grepl("^Asymptomatic", fname)){
    DataAsymp <- rbind(DataAsymp, x1)
    asymp_labels <- c(asymp_labels, paste0("A", id))
  } else if(grepl("^Symptomatic", fname)){
    DataSymp <- rbind(DataSymp, x1)
    symp_labels <- c(symp_labels, paste0("S", id))
  } else {
    warning(paste("File not assigned to group:", fname))
  }
}

DataBoth <- rbind(DataAsymp, DataSymp)

# ================================================================
# Create FD objects
# ================================================================
create_fd <- function(Data){
  x <- seq(0, 1, length.out = ncol(Data))
  basis <- create.bspline.basis(c(0,1), nbasis = ncol(Data)-1, norder = 2)
  fdobj <- smooth.basis(x, t(Data), basis)
  return(fdobj$fd)
}

fdSymp <- create_fd(DataSymp)
fdAsymp <- create_fd(DataAsymp)
fdBoth <- create_fd(DataBoth)

plot(mean.fd(fdSymp), main="Mean Symptomatic Landscape")
plot(mean.fd(fdAsymp), main="Mean Asymptomatic Landscape")

# ================================================================
# Functional PCA
# ================================================================
pca <- pca.fd(fdBoth, nharm=2)
m <- c(asymp_labels, symp_labels)
col.group <- c(rep("black", length(asymp_labels)), rep("blue", length(symp_labels)))
group <- c(rep("Asymptomatic", length(asymp_labels)), rep("Symptomatic", length(symp_labels)))

pca_df <- data.frame(
  PC1 = pca$scores[, 1],
  PC2 = pca$scores[, 2],
  ID = m,
  Group = group
)

  ggplot(pca_df, aes(x = PC1, y = PC2, color = Group, label = ID)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
    geom_point(size = 4, alpha = 0.9) +
    ggrepel::geom_text_repel(size = 4, show.legend = FALSE, max.overlaps = Inf) +
    scale_color_manual(values = c("Asymptomatic" = "black", "Symptomatic" = "blue")) +
    labs(
      x = "PC1 Score",
      y = "PC2 Score",
      title = ""
    ) +
    coord_fixed() +  
    theme_classic(base_size = 16) +
    theme(
      legend.position = "right",
      legend.title = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "bold")
    )


# ================================================================
# Permutation hypothesis test
# ================================================================
#Test of asymptomatic and symptomatic
ptest1 <- tperm.fd(fdAsymp, fdSymp, nperm = 10000, q = 0.05, plotres = TRUE)
ptest1$pval

#Test of symptomatic and symptomatic
splits <- combn(1:9, 4)
n_splits <- ncol(splits)
pvals <- numeric(n_splits)

for(i in 1:n_splits){
  
  group1_idx <- splits[, i]
  group2_idx <- setdiff(1:9, group1_idx)
  
  fdSymp1 <- create_fd(DataSymp[group1_idx, ])
  fdSymp2 <- create_fd(DataSymp[group2_idx, ])
  
  ptest <- tperm.fd(fdSymp1, fdSymp2,
                    nperm = 10000,
                    q = 0.05,
                    plotres = FALSE)
  
  pvals[i] <- ptest$pval
  print(i)
}

mean_p <- mean(pvals)
sd_p   <- sd(pvals)

mean_p
sd_p

#Test of asymptomatic and asymptomatic
# combn(1:8, 4) enumerates all 70 4-element subsets of 8 items. Since the
# complement of a 4-element subset of 8 is also a 4-element subset, every
# 4-vs-4 partition appears twice here (once as (group1, group2), once as
# (group2, group1) when its complement is later drawn as group1_idx) — so
# these 70 runs cover only 35 unique partitions, each tested twice. Not
# changed here to keep reproducing the baseline numbers; the analogous
# combn(1:9, 4) loop above does not have this issue (a 4-vs-5 split's
# 5-element complement is never itself drawn as a 4-element subset).
splits <- combn(1:8, 4)
n_splits <- ncol(splits)
pvals <- numeric(n_splits)

for(i in 1:n_splits){
  
  group1_idx <- splits[, i]
  group2_idx <- setdiff(1:8, group1_idx)
  
  fdSymp1 <- create_fd(DataAsymp[group1_idx, ])
  fdSymp2 <- create_fd(DataAsymp[group2_idx, ])
  
  ptest <- tperm.fd(fdSymp1, fdSymp2,
                    nperm = 10000,
                    q = 0.05,
                    plotres = FALSE)
  
  pvals[i] <- ptest$pval
  print(i)
}

mean_p <- mean(pvals)
sd_p   <- sd(pvals)

mean_p
sd_p
