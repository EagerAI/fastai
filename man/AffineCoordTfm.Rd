% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/vision_augment.R
\name{AffineCoordTfm}
\alias{AffineCoordTfm}
\title{AffineCoordTfm}
\usage{
AffineCoordTfm(
  aff_fs = NULL,
  coord_fs = NULL,
  size = NULL,
  mode = "bilinear",
  pad_mode = "reflection",
  mode_mask = "nearest",
  align_corners = NULL
)
}
\arguments{
\item{aff_fs}{aff_fs}

\item{coord_fs}{coord_fs}

\item{size}{size}

\item{mode}{mode}

\item{pad_mode}{pad_mode}

\item{mode_mask}{mode_mask}

\item{align_corners}{align_corners}
}
\description{
Combine and apply affine and coord transforms
}